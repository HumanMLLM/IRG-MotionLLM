import torch
from torch.nn.utils import rnn
import re 
@torch.inference_mode()
def multi_round_inference_engine(eval_model, text_processor, task_type, input_info, device, max_refine_round=5, do_sample_=False, verbose=False, return_inter_motions=False):
    eval_model.eval()
    # import pdb; pdb.set_trace()
    if task_type in ['t2m', 't2m_motionagent', 'unified_mogen_cot_v2']:
        input_ids, attention_mask = text_processor.build_test_context(test_task=task_type, inputs=input_info, tokenizer=eval_model.tokenizer)
        bs = input_ids.shape[0]
        do_sample = do_sample_
        # Generate outputs for the batch
        outputs = eval_model.llm.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_length=2048,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
        # import pdb; pdb.set_trace()
        
        results = eval_model.post_process_final_motion_output(outputs, input_ids, bs, return_dict=True)
        if verbose:
            print('=' * 50)
            print(input_ids)
            print('-' * 50)
            print(attention_mask)
            print('-' * 50)
            tknz = eval_model.tokenizer
            print(tknz.batch_decode(input_ids)[0])
            print('-' * 50)
            print(results[0])
            print('=' * 50)

        if return_inter_motions:
            total_think_motion_tokens = []
            total_think_motion_pos = []
            completion_only_scores = torch.stack(outputs.scores).transpose(0,1)
            for j, score in enumerate(completion_only_scores):
                answer_scores_j, think_scores_j, st_ans_pos_j, ed_ans_pos_j = eval_model.extract_answer_scores(score, eval_model.tokenizer, return_position=True, return_prefix_scores=True)
                # import pdb; pdb.set_trace()
                # answer_ids_j = outputs.sequences[j][input_ids.shape[1]:][st_ans_pos_j: ed_ans_pos_j]
                think_ids_j = outputs.sequences[j][input_ids.shape[1]:][:st_ans_pos_j]
                

                # answer_motion_tokens_j = eval_model.get_motion_from_scores_w_ids(answer_scores_j, answer_ids_j) 

                all_think_motion_pos_j = eval_model.extract_motions_from_scores_w_ids(think_ids_j)
                all_think_motion_ids_j = []
                all_think_motion_tokens_j = []
                
                for g, pos in enumerate(all_think_motion_pos_j):
                    think_motion_ids_g = think_ids_j[pos[0]: pos[1]+1]
                    think_motion_scores_g = think_scores_j[pos[0]: pos[1]+1]
                    think_motion_tokens_g = eval_model.get_motion_from_scores_w_ids(think_motion_scores_g, think_motion_ids_g) 
                    all_think_motion_ids_j.append(think_motion_ids_g)
                    all_think_motion_tokens_j.append(think_motion_tokens_g)

                results[j]['think_motion_tokens'] = all_think_motion_tokens_j


    elif task_type in ['unified_mogen_cot_v3', 'unified_mogen_cot_v3_wo_assess_refine']:
        input_ids, attention_mask = text_processor.build_test_context(test_task=task_type, inputs=input_info, tokenizer=eval_model.tokenizer)
        bs = input_ids.shape[0]
        do_sample = do_sample_
        # Generate outputs for the batch
        outputs = eval_model.llm.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_length=1300,
            do_sample=True,
            return_dict_in_generate=True,
            use_cache=True,
        )
        results = []
        for b in range(bs):
            results.append({
                'input_text': eval_model.tokenizer.decode(input_ids[b], skip_special_tokens=True),
                'best_beam_text': eval_model.tokenizer.decode(outputs.sequences[b][input_ids.shape[1]: ], skip_special_tokens=True),
            })
        total_think_motion_tokens = []
        total_think_motion_pos = []
        # import pdb; pdb.set_trace()

        for j, res in enumerate(results):
            motion_txt_j = res['best_beam_text'].split('<Motion>')[-1].split('</Motion>')[0]

            pattern = r'<Motion_(\d+)>'                                      
            matches = re.findall(pattern, motion_txt_j)                              
            final_motion_tokens = torch.tensor([int(match) for match in matches]).to(device)

            results[j]['motion_tokens'] = final_motion_tokens                                                

    elif task_type == 'direct_generation-instructed_refinement':
        input_ids, attention_mask = text_processor.build_test_context(
            test_task='t2m', inputs=input_info, tokenizer=eval_model.tokenizer
        )
        bs = input_ids.shape[0]
        do_sample = do_sample_

        outputs = eval_model.llm.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_length=1024,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
        results_step1 = eval_model.post_process_final_motion_output(
            outputs, input_ids, bs, return_dict=True
        )
        final_results = results_step1

        if verbose:
            print('=' * 50)
            print('Step 1: Initial Motion Generation')
            print('-' * 50)
            print(eval_model.tokenizer.batch_decode(input_ids)[0])
            print('-' * 50)
            print(results_step1[0])
            print('=' * 50)

        refine_signal = [True for _ in range(bs)]
        for refine_round in range(max_refine_round):
            previous_motion_strings = [
                res['best_beam_text'].split('</Motion>')[0].split('<Motion>')[-1]
                for res in final_results
            ]

            input_info['motions'] = [
                {'<Generated_Motion>': previous_motion_strings[b]} for b in range(bs)
            ]
            input_ids, attention_mask = text_processor.build_test_context(
                test_task='refine_instruction', inputs=input_info, tokenizer=eval_model.tokenizer
            )

            outputs = eval_model.llm.generate(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=1024,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=False,
                use_cache=True,
            )
            refine_instructions = eval_model.tokenizer.batch_decode(
                outputs.sequences[:, input_ids.shape[1]:], skip_special_tokens=True
            )
            refine_signal = [
                refine_signal[b] and (refine_instructions[b] != 'No refinements needed.')
                for b in range(bs)
            ]

            if verbose:
                print('=' * 50)
                print('Step 2: Instructed Refinement')
                print(f'Round {refine_round + 1} / {max_refine_round}')
                print('Motion Refinement Instructing')
                print('-' * 50)
                print(refine_signal)
                print('-' * 50)
                print(eval_model.tokenizer.batch_decode(input_ids)[0])
                print('-' * 50)
                print(refine_instructions)
                print('=' * 50)

            if not any(refine_signal):
                if verbose:
                    print('=' * 50)
                    print('Early stop refinement')
                    print('=' * 50)
                break

            for b in range(bs):
                input_info['texts'][b]['<Refinement_Instruction>'] = refine_instructions[b]

            input_ids, attention_mask = text_processor.build_test_context(
                test_task='instructed_motion_refinement',
                inputs=input_info,
                tokenizer=eval_model.tokenizer,
            )
            outputs = eval_model.llm.generate(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=1024,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
            refined_results = eval_model.post_process_final_motion_output(
                outputs, input_ids, bs, return_dict=True
            )

            if verbose:
                print('=' * 50)
                print('Step 2: Instructed Refinement')
                print(f'Round {refine_round + 1} / {max_refine_round}')
                print('Motion Refinement')
                print('-' * 50)
                print(refine_signal)
                print('-' * 50)
                print(eval_model.tokenizer.batch_decode(input_ids)[0])
                print('-' * 50)
                print(refined_results[0])
                print('=' * 50)

            final_results = [
                refined_results[b] if refine_signal[b] else final_results[b]
                for b in range(bs)
            ]

        results = final_results

    return results