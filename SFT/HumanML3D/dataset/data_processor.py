import numpy as np
import json 
import os
import random
import torch
from torch.nn.utils import rnn
import re

class TextProcessor():
    def __init__(self, args=None):
        self.context_style = 'gemma'
        self.args = args
        self.max_tgt_len = 2048
        # import pdb; pdb.set_trace()
        self.args.prompt_w_response = args.prompt_w_response if hasattr(args, 'prompt_w_response') else False
        self.wo_answer = args.wo_answer if hasattr(args, 'wo_answer') else False
        self.ignore_incorrect = args.ignore_incorrect if hasattr(args, 'ignore_incorrect') else False
        self.wo_assess_refine = args.wo_assess_refine if hasattr(args, 'wo_assess_refine') else False
        self.unified_mogen_planner_templates = {
            "<prompt_analysis_plan>": "I need to analyze the goal and identify the critical aspects to generate the motion.",
            "<generation_plan>": "I need to generate a motion according to the previous analysis.",
            "<assessment_plan>": "I need to assess the alignment between the generated motion and the goal text, then provide specific refinement instructions for improvement.",
            "<refinement_plan>": "I need to refine the motion according to the instruction.",
            "<finish_thinking_plan>": "I need to finish thinking as the satisfactory motion is generated.",
        }
        self.test_conversation_templates = {
            "t2m_motionagent": {
                "system_prompt": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
                "user_prompt": "### Instruction:\nGenerate a motion matching the following input human motion description\n\n### Input:\n<Goal_Caption>"
            },
            "t2m": {
                "system_prompt": "You are an assistant who helps users understand or generate 3D human motion representations.",
                "user_prompt": "Generate a motion that depicts <Goal_Caption>."
            },
            "prompt_analysis": {
                "system_prompt": "You are an assistant who helps users understand or generate 3D human motion representations.",
                "user_prompt": "Determine the critical aspects to address prior to generating a motion that represents <Goal_Caption>."
            },
            "refine_instruction":{
                "system_prompt": "You are an assistant who helps users understand or generate 3D human motion representations.",
                "user_prompt": "Motion: <Generated_Motion>. Goal text: <Goal_Caption>. Determine how to refine the generated motion for better alignment."
            },
            "instructed_motion_refinement":{
                "system_prompt": "You are an assistant who helps users understand or generate 3D human motion representations.",
                "user_prompt": "Motion: <Generated_Motion>. Goal text: <Goal_Caption>. Refine the motion according to the instruction: <Refinement_Instruction>."
            },
            "unified_mogen_cot_v2": {
                "system_prompt": "You are an assistant who helps users understand or generate 3D human motion representations.",
                "user_prompt": "Given a goal text describing a human motion, your job includes the following parts: (1) analyzing the goal text and clearly describing the reasoning process to identify key elements; (2) performing multiple rounds of generation and self-assessment until a satisfactory motion is achieved; (3) outputting the final motion based on the reasoning process. Response format: Show initial analysis and generation-assessment cycles indide <think> and </think> tags; Place the final motion inside <answer> and </answer> tags.\nGoal Text: <Goal_Caption>"
            },
            "unified_mogen_cot_v3": {
                "system_prompt": "You are an assistant who helps users understand or generate 3D human motion representations.",
                "user_prompt": "Given a text outlining a human motion objective, employ a step-by-step thought process to realize the motion: (1) analyze the text, providing a clear explanation of the reasoning to identify essential elements; (2) conduct several rounds of motion generation and self-assessment until the motion is satisfactory. Wrap all responses in <think> and </think> tags, and formulate a plan before each step.\nGoal Text: <Goal_Caption>"
            },
            "m2t": {
                "system_prompt": "You are an assistant who helps users understand or generate 3D human motion representations.",
                "user_prompt": "<Goal_Motion>\nProvide a detailed description of the motion."
            },
            "unified_mogen_cot_v3_wo_assess_refine":{
                "system_prompt": "You are an assistant who helps users understand or generate 3D human motion representations.",
                "user_prompt": "User: Given a goal text describing a human motion, your job includes the following parts: (1) analyzing the goal text and clearly describing the reasoning process to identify key elements; (2) outputting the motion based on the reasoning process. Put all your response in <think> and </think> tags, and make a plan before each step.\nGoal Text: <Goal_Caption>"
            }
        }
        self.gen_forcing =True
    

    def process_batch_data(self, inputs, tokenizer):

        batch_size = len(inputs['system_prompt'])
        batch_input_ids, batch_target_ids = [], []

        for b in range(batch_size):

            input_info = {key: value[b] for key, value in inputs.items()}
            task = input_info['task']

            if 'unified_mogen_cot' in task:
                one_input_ids, one_target_ids = self.build_context_unified_mogen_cot(input_info, tokenizer)
            else:
                one_input_ids, one_target_ids = self.build_context_naive(input_info, tokenizer)

            assert '_Motion>' not in tokenizer.decode(one_input_ids) and '_motion>' not in tokenizer.decode(one_input_ids), f"{task}\n{tokenizer.decode(one_input_ids)}"

            batch_input_ids.append(torch.LongTensor(one_input_ids))
            batch_target_ids.append(torch.LongTensor(one_target_ids))



        input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
        assert input_ids.size() == target_ids.size()

        input_ids = input_ids[:, :self.max_tgt_len]
        target_ids = target_ids[:, :self.max_tgt_len]

        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
        assert attention_mask.size() == input_ids.size()
        return input_ids, target_ids, attention_mask

    def build_context_unified_mogen_cot(self, input_info, tokenizer):
        system_prompt = input_info['system_prompt']
        user_prompt  = input_info['user_prompt'] if 'user_prompt' in input_info else input_info['instruction']
        # think = input_info['think']
        answer = input_info['solution'] if 'solution' in input_info else input_info['response']

        motions = input_info['motion_info'] if 'motion_info' in input_info else input_info['motions']
        motions = {k: v['motion_tokens'] for k,v in motions.items()}
        # get motion strings
        motions_ = {}
        for motion_key, motion_tokens in motions.items():
            motion_string = tokenizer.decode(motion_tokens)
            motions_[motion_key] = motion_string
        motions = motions_

        texts = input_info['texts']
        max_intermediate_gens = 4
        
        num_intermediate_motions = random.choice([0, 1, 2, 3, 4])
        if num_intermediate_motions > 0:
            intermediate_motion_keys = sorted(random.sample([f'<Intermediate_Motion_{i}>' for i in range(4)], num_intermediate_motions))
            clean_intermediate_motion_keys = [k for k in intermediate_motion_keys if (texts[k.replace('<', '<Assess_')]['refinement'] not in ['No refinement needed.', 'No refinements needed.']) and len(texts[k.replace('<', '<Assess_')]['refinement'].split(' ')) > 3 and ('No refinement needed' not in texts[k.replace('<', '<Assess_')]['refinement']) and ('No refinement needed' not in texts[k.replace('<', '<Assess_')]['refinement']) and ('No refinements needed' not in texts[k.replace('<', '<Assess_')]['refinement'])]       # 这里是为了过滤掉perfect的中间结果
            intermediate_motion_keys = clean_intermediate_motion_keys
        else:
            intermediate_motion_keys = []
        response_breakdowns = []
        response_breakdowns.append(['tag', '<think>'])
        response_breakdowns.append(['plan', '<prompt_analysis_plan>'])
        response_breakdowns.append(['analyze', '<Prompt_Analysis>'])
        response_breakdowns.append(['plan','<generation_plan>'])
        if self.wo_assess_refine and self.wo_answer:
            response_breakdowns.append(['generate', '<Goal_Motion>'])
        else:
            for k in intermediate_motion_keys:
                response_breakdowns.append(['generate', k])
                response_breakdowns.append(['plan','<assessment_plan>'])
                response_breakdowns.append(['assess',k.replace('<', '<Assess_')])
                response_breakdowns.append(['plan','<refinement_plan>'])
            response_breakdowns.append(['generate', '<Goal_Motion>'])
            response_breakdowns.append(['plan','<assessment_plan>'])
            response_breakdowns.append(['assess', '<Assess_Goal_Motion>'])
            response_breakdowns.append(['plan','<finish_thinking_plan>'])
        response_breakdowns.append(['tag', '</think>'])
        if not self.wo_answer:
            response_breakdowns.append(['tag', '<answer>'])
            response_breakdowns.append(['generate', '<Goal_Motion>'])
            response_breakdowns.append(['tag', '</answer>'])
        response_breakdowns.append(['tag', '<eos>'])

        plan_begin_tag_ids = tokenizer('[plan]', add_special_tokens=False).input_ids
        plan_end_tag_ids = tokenizer('[/plan]', add_special_tokens=False).input_ids
        analyze_begin_tag_ids = tokenizer('[analyze]', add_special_tokens=False).input_ids
        analyze_end_tag_ids = tokenizer('[/analyze]', add_special_tokens=False).input_ids
        assess_begin_tag_ids = tokenizer('[assess]', add_special_tokens=False).input_ids
        assess_end_tag_ids = tokenizer('[/assess]', add_special_tokens=False).input_ids
        generate_begin_tag_ids = tokenizer('[generate]', add_special_tokens=False).input_ids
        generate_end_tag_ids = tokenizer('[/generate]', add_special_tokens=False).input_ids
        motion_begin_tag_ids = tokenizer('<Motion>', add_special_tokens=False).input_ids
        motion_end_tag_ids = tokenizer('</Motion>', add_special_tokens=False).input_ids
        nextline_tag_ids = tokenizer('\n', add_special_tokens=False).input_ids

        if self.context_style == 'gemma':
            bos_id = tokenizer.bos_token_id
            one_input_ids, one_target_ids = [], []
            one_input_ids.append(bos_id)
            one_target_ids.append(-100)
            one_full_text = tokenizer.decode(bos_id)
            
            for text_key, text in texts.items():
                if text_key in system_prompt:
                    system_prompt = system_prompt.replace(text_key, text).replace('..', '.')
                if text_key in user_prompt:
                    user_prompt = user_prompt.replace(text_key, text).replace('..', '.')

            input_text = system_prompt + '\n\n' + "User: " + user_prompt + '\n\n'
            if self.args.prompt_w_response:
                input_text += ' Response:'
            
            one_full_text += input_text
            full_prompt_ids = tokenizer(input_text, add_special_tokens=False).input_ids
            one_input_ids += full_prompt_ids
            one_target_ids += [-100] * len(full_prompt_ids)

            thinking_stage = True
            for step in response_breakdowns:
                mode, value = step

                if mode == 'tag':
                    if value == '<answer>':
                        thinking_stage = False
                    step_ids = tokenizer(value, add_special_tokens=False).input_ids
                    one_input_ids += step_ids
                    one_full_text += value
                    one_target_ids += step_ids
                elif mode == 'plan':
                    if self.args.wo_plan:
                        continue
                    plan_value = value
                    for plan_k, plan_v in self.unified_mogen_planner_templates.items():
                        plan_value = plan_value.replace(plan_k, plan_v)
                    # plan_value = '[plan]' + plan_value + '[/plan]\n'
                    step_ids = plan_begin_tag_ids + tokenizer(plan_value, add_special_tokens=False).input_ids + plan_end_tag_ids + nextline_tag_ids
                    one_input_ids += step_ids
                    one_full_text += '[plan]' + plan_value + '[/plan]\n'
                    one_target_ids += step_ids
                elif mode == 'analyze':
                    analysis_value = value
                    for analysis_k, analysis_v in texts.items():
                        if analysis_k in analysis_value:
                            analysis_value = analysis_value.replace(analysis_k, analysis_v)
                    # analysis_value = '[analyze]' + analysis_value + '[/analyze]\n'
                    step_ids = analyze_begin_tag_ids + tokenizer(analysis_value, add_special_tokens=False).input_ids + analyze_end_tag_ids + nextline_tag_ids
                    one_input_ids += step_ids
                    one_full_text += '[analyze]' + analysis_value + '[/analyze]\n'
                    one_target_ids += step_ids
                elif mode == 'assess':
                    assess_value = value
                    for assess_k, assess_v in texts.items():
                        if assess_k in assess_value:
                            assessment_value = assess_v['assessment']
                            refinement_value = assess_v['refinement']
                            assess_v_ = f'Assessment: {assessment_value}\n'
                            assess_v_ += f'Refinement: {refinement_value}'
                            assess_value = assess_value.replace(assess_k, assess_v_)
                    # assess_value = '[assess]' + assess_value + '[/assess]\n'
                    step_ids = assess_begin_tag_ids + tokenizer(assess_value, add_special_tokens=False).input_ids + assess_end_tag_ids + nextline_tag_ids
                    one_input_ids += step_ids
                    one_full_text += '[assess]' + assess_value + '[/assess]\n'
                    one_target_ids += step_ids
                elif mode == 'generate':
                    motion_string = motions[value]
                    exact_motion_ids = motion_begin_tag_ids + tokenizer(motion_string, add_special_tokens=False).input_ids + motion_end_tag_ids

                    if thinking_stage:
                        exact_step_ids = generate_begin_tag_ids + exact_motion_ids + generate_end_tag_ids
                        one_full_text += '[generate]' + '<Motion>' + motion_string + '</Motion>' + '[/generate]\n'
                    else:
                        exact_step_ids = exact_motion_ids
                        one_full_text += '<Motion>' + motion_string + '</Motion>'
                    # one_input_ids += exact_step_ids
                    # one_input_ids += nextline_tag_ids
                    
                    if value == '<Goal_Motion>' or not self.args.ignore_incorrect:
                        one_input_ids += exact_step_ids
                        one_target_ids += exact_step_ids
                        if thinking_stage:
                            one_input_ids += nextline_tag_ids
                            one_target_ids += nextline_tag_ids
                    else:
                        one_input_ids += exact_step_ids
                        one_target_ids += [-100] * len(exact_step_ids)      # 我们不能让模型学习错误的motion
                        if thinking_stage:
                            one_input_ids += nextline_tag_ids
                            one_target_ids += [-100] * len(nextline_tag_ids)

        assert one_full_text == tokenizer.decode(one_input_ids)
        assert len(one_input_ids) == len(one_target_ids)
        if not self.wo_assess_refine:
            pattern=r".*?<think>\s*\[plan\].*?\[/plan\]\s*\[analyze\].*?\[/analyze\]\s*\[plan\].*?\[/plan\]\s*(?:(?:\[generate\]<Motion>(?:<Motion_\d{1,3}>)+</Motion>\[/generate\]\s*\[plan\].*?\[/plan\]\s*\[assess\].*?\[/assess\])\s*\[plan\].*?\[/plan\]\s*)*</think>\s*<answer><Motion>(?:<Motion_\d{1,3}>)+</Motion></answer><eos>"
            if self.wo_answer:
                pattern=r".*?<think>\s*\[plan\].*?\[/plan\]\s*\[analyze\].*?\[/analyze\]\s*\[plan\].*?\[/plan\]\s*(?:(?:\[generate\]<Motion>(?:<Motion_\d{1,3}>)+</Motion>\[/generate\]\s*\[plan\].*?\[/plan\]\s*\[assess\].*?\[/assess\])\s*\[plan\].*?\[/plan\]\s*)*</think>\s*<eos>"
            assert re.search(pattern, tokenizer.decode(one_input_ids), re.DOTALL) is not None
            assert len(one_full_text.split('No refinement needed')) <= 2
            assert len(one_full_text.split('No refinements needed')) <= 2
        assert '_Template' not in one_full_text

        if not self.wo_answer:
            split_motion_st = [t for t in tokenizer.decode(one_input_ids).split('<Motion>')[1:] if '</Motion>' in t]
            gen_motions = [t.split('</Motion>')[0] for t in split_motion_st]
            assert gen_motions[-1] == gen_motions[-2]
        return one_input_ids, one_target_ids

    def build_context_naive(self, input_info, tokenizer):
        system_prompt = input_info['system_prompt']
        user_prompt  = input_info['user_prompt'] if 'user_prompt' in input_info else input_info['instruction']
        # think = input_info['think']
        answer = input_info['solution'] if 'solution' in input_info else input_info['response']

        motions = input_info['motion_info'] if 'motion_info' in input_info else input_info['motions']
        motions = {k: v['motion_tokens'] for k,v in motions.items()}

        texts = input_info['texts']


        # 忽略thinking
        response = answer

        for motion_key, motion_tokens in motions.items():
            motion_string = tokenizer.decode(motion_tokens)

            user_prompt = user_prompt.replace(motion_key, '<Motion>'+motion_string+'</Motion>')
            response = response.replace(motion_key, '<Motion>'+motion_string+'</Motion>')

        for text_key, text in texts.items():
            user_prompt = user_prompt.replace(text_key, text).replace('..', '.')
            response = response.replace(text_key, text).replace('..', '.')


        prompt_messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # build full input text, target, and attention mask
        if self.context_style == 'gemma':
            bos_id = tokenizer.bos_token_id
            one_input_ids, one_target_ids = [], []
            one_input_ids.append(bos_id)
            one_target_ids.append(-100)

            input_text = system_prompt + '\n\n' + "User: " + user_prompt + '\n\n'
            if self.args.prompt_w_response:
                input_text += ' Response:'

            response_text = (response + '<eos>')

            input_text_ids = tokenizer(input_text, add_special_tokens=False).input_ids
            one_target_ids += [-100] * len(input_text_ids)

            response_text_ids = tokenizer(response_text, add_special_tokens=False).input_ids
            one_target_ids += response_text_ids

            one_input_ids += input_text_ids + response_text_ids
            full_text = '<bos>' + input_text + response_text
            
            assert len(one_input_ids) == len(one_target_ids)
            assert tokenizer.decode(one_input_ids) == full_text
        elif self.context_style == 'qwen':
            one_input_ids, one_target_ids = [], []
            input_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            response_text = (response + tokenizer.eos_token)

            input_text_ids = tokenizer(input_text, add_special_tokens=False).input_ids
            one_target_ids += [-100] * len(input_text_ids)

            response_text_ids = tokenizer(response_text, add_special_tokens=False).input_ids
            one_target_ids += response_text_ids

            one_input_ids += input_text_ids + response_text_ids
            full_text = input_text + response_text
            
            assert len(one_input_ids) == len(one_target_ids)
            assert tokenizer.decode(one_input_ids) == full_text

        return one_input_ids, one_target_ids


    def build_test_context(self, test_task, inputs, tokenizer):
        conversation_template = self.test_conversation_templates[test_task]
        meta_system_prompt = conversation_template['system_prompt']
        meta_user_prompt  = conversation_template['user_prompt']
        
        batch_inputs = []

        # here it should be a list, e.g., [{'<Goal_Caption>': 'caption_1'}, {'<Goal_Caption>': 'caption_2'}]
        batch_size = len(inputs['texts']) if 'texts' in inputs else len(inputs['motions'])
        texts = inputs['texts'] if 'texts' in inputs else [{} for b in range(batch_size)]
        motions = inputs['motions'] if 'motions' in inputs else [{} for b in range(batch_size)]
        # batch_size = len(texts)    
        if self.context_style == 'gemma':
            for b in range(batch_size):
                text_info = texts[b]
                motion_info = motions[b]
                system_prompt = meta_system_prompt
                user_prompt = meta_user_prompt
                for txt_key, txt in text_info.items():
                    system_prompt = system_prompt.replace(txt_key, txt).replace('..', '.')
                    user_prompt = user_prompt.replace(txt_key, txt).replace('..', '.')
                    if '<Goal_Caption>' in user_prompt:
                        import pdb; pdb.set_trace()
                for mo_key, mo in motion_info.items():
                    if type(mo) == str:
                        system_prompt = system_prompt.replace(mo_key, '<Motion>'+mo+'</Motion>')
                        user_prompt = user_prompt.replace(mo_key, '<Motion>'+mo+'</Motion>')
                    else:
                        mo = tokenizer.decode(mo)
                        system_prompt = system_prompt.replace(mo_key, '<Motion>'+mo+'</Motion>')
                        user_prompt = user_prompt.replace(mo_key, '<Motion>'+mo+'</Motion>')
                if test_task not in ['t2m_motionagent']:
                    input_text = system_prompt + '\n\n' + "User: " + user_prompt + '\n\n'
                else:
                    input_text = system_prompt + '\n\n' + user_prompt + '\n\n'
                if self.args.prompt_w_response:
                    if test_task not in ['t2m_motionagent']:
                        input_text += ' Response:'
                    else:
                        input_text += 'Response: '
                if self.gen_forcing and test_task in ['t2m', "instructed_motion_refinement", 't2m_motionagent']:
                    input_text += '<Motion>'
                batch_inputs.append(input_text)
                # import pdb; pdb.set_trace()
            input_ = tokenizer(batch_inputs, return_tensors="pt", padding=True) # 这里会自动加入 <bos>
            input_ids = input_.input_ids
            attention_mask = input_.attention_mask
        
        return input_ids, attention_mask