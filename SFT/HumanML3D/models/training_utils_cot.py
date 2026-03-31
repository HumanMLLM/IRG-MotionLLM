import random
import torch
from torch.nn.utils import rnn

def process_cot_batch(data, tokenizer,max_tgt_len, gt_forcing=False, ignore_incorrect=False):
    if gt_forcing:
        return process_cot_batch_gt_forcing(data, tokenizer, max_tgt_len)
    if ignore_incorrect:
        return process_cot_batch_ignore_incorrect(data, tokenizer, max_tgt_len)
    bos = tokenizer.bos_token_id
    mask_id = tokenizer.encode('<mask>', add_special_tokens=False)[0]
    if bos is None:
        # TODO:
        assert False, "No implementation"

    system_prompts = data['system_prompt']
    instructions = data['instruction']
    responses = data['response']
    all_motion_info = data['motion_info']

    batch_size = len(system_prompts)
    batch_input_ids, batch_target_ids = [], []

    for b in range(batch_size):
        one_input_ids, one_target_ids = [], []
        
        one_input_ids.append(bos)
        one_target_ids.append(-100)  # do not perform loss regression on human prompt
        texts = ''
        # prompt = "<System_Prompt_Placeholder>\n\n"
        instruction_meta = '<Instruction_Placeholder> \n\n'
        response_meta = '<Response_Placeholder>'

        prompt = f"{system_prompts[b]}\n\n"
        instruction_post = instruction_meta.replace('<Instruction_Placeholder>', instructions[b])
        response_post = response_meta.replace('<Response_Placeholder>', responses[b])

        for motion_key, motion_info in all_motion_info[b].items():
            motion_tokens = motion_info['motion_tokens']
            motion_string = tokenizer.decode(motion_tokens)

            instruction_post = instruction_post.replace(motion_key, '<Motion>'+motion_string+'</Motion>')
            response_post = response_post.replace(motion_key, '<Motion>'+motion_string+'</Motion>')
        
        input_text = prompt + instruction_post
        response_text = (response_post + '<eos>').replace(' <eos>', '<eos>')
        full_text = (prompt + instruction_post + response_post + '<eos>').replace(' <eos>', '<eos>')

        input_text_ids = tokenizer(input_text, add_special_tokens=False).input_ids
        one_target_ids += [-100] * len(input_text_ids)

        response_text_ids = tokenizer(response_text, add_special_tokens=False).input_ids
        one_target_ids += response_text_ids

        # input_ids += tokenizer(full_text, add_special_tokens=False).input_ids
        one_input_ids += input_text_ids + response_text_ids

        assert len(one_input_ids) == len(one_target_ids)
        # import pdb; pdb.set_trace()
        
        # check if the training data is valid
        # print('checking data')
        # check_results = check_format(response_text)
        # total = len(check_results)
        # passed = sum(1 for _, (is_met, _) in check_results.items() if is_met)
        # assert passed == total, input_text + '\n' + response_text

        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
        
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)

    assert input_ids.size() == target_ids.size()
    # assert input_ids.shape[0] < max_tgt_len
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


def process_cot_batch_gt_forcing(data, tokenizer,max_tgt_len):
    bos = tokenizer.bos_token_id
    mask_id = tokenizer.encode('<mask>', add_special_tokens=False)[0]
    if bos is None:
        # TODO:
        assert False, "No implementation"

    system_prompts = data['system_prompt']
    instructions = data['instruction']
    responses = data['response']
    breakdown_responses = data['breakdown_response']
    all_motion_info = data['motion_info']

    batch_size = len(system_prompts)
    batch_input_ids, batch_target_ids = [], []
    batch_attention_mask = []
    batch_prompt_len = []

    nextline_tag_ids = tokenizer('\n', add_special_tokens=False).input_ids
    
    gt_forcing_position_all = []
    for bs in range(batch_size):
        one_input_ids, one_target_ids = [], []
        gt_forcing_position_b = []
        one_full_text = tokenizer.decode(bos)

        one_input_ids.append(bos)
        one_target_ids.append(-100)  # do not perform loss regression on human prompt

        system_prompt = system_prompts[bs] + '\n\n'
        user_prompt = instructions[bs] + '\n\n'
        motion_info_b = all_motion_info[bs]

        # <prompt>
        full_prompt = system_prompt + user_prompt
        full_prompt_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids
        one_full_text += full_prompt
        one_input_ids += full_prompt_ids
        one_target_ids += [-100] * len(full_prompt_ids)
        prompt_len = len(one_input_ids)
        batch_prompt_len.append(prompt_len)
        # <think>
        breakdown_response = breakdown_responses[bs]
        if len(breakdown_response['think']) > 0:
            think_begin_tag_ids = tokenizer('<think>', add_special_tokens=False).input_ids
            think_end_tag_ids = tokenizer('</think>', add_special_tokens=False).input_ids

            one_full_text += '<think>'
            one_input_ids += think_begin_tag_ids
            one_target_ids += think_begin_tag_ids

        for step in breakdown_response['think']:
            mode, value = step

            if mode == 'analyze':
                analyze_begin_tag_ids = tokenizer('[analyze]', add_special_tokens=False).input_ids
                analyze_end_tag_ids = tokenizer('[/analyze]', add_special_tokens=False).input_ids
                analyze_ids = tokenizer(value, add_special_tokens=False).input_ids
                step_ids = analyze_begin_tag_ids + analyze_ids + analyze_end_tag_ids + nextline_tag_ids
                analyze_text = '[analyze]' + value + '[/analyze]' + '\n'

                one_full_text += analyze_text
                one_input_ids += step_ids
                one_target_ids += step_ids
            if mode == 'assess':
                assess_begin_tag_ids = tokenizer('[assess]', add_special_tokens=False).input_ids
                assess_end_tag_ids = tokenizer('[/assess]', add_special_tokens=False).input_ids
                assess_ids = tokenizer(value, add_special_tokens=False).input_ids
                step_ids = assess_begin_tag_ids + assess_ids + assess_end_tag_ids + nextline_tag_ids
                assess_text = '[assess]' + value + '[/assess]' + '\n'

                one_full_text += assess_text
                one_input_ids += step_ids
                one_target_ids += step_ids
            if mode == 'generate':
                generate_begin_tag_ids = tokenizer('[generate]', add_special_tokens=False).input_ids
                generate_end_tag_ids = tokenizer('[/generate]', add_special_tokens=False).input_ids
                motion_begin_tag_ids = tokenizer('<Motion>', add_special_tokens=False).input_ids
                motion_end_tag_ids = tokenizer('</Motion>', add_special_tokens=False).input_ids

                # 处理generate id
                # 在这个地方，value 必须 写成 "<Goal_Motion>" 或者 "<Goal_Motion><Intermediate_Motion_x>"的形式
                to_gen_list = value.split('><')
                if len(to_gen_list) == 1:
                    # 只有Goal_Motion
                    assert to_gen_list[0] == '<Goal_Motion>'
                    goal_motion_string = tokenizer.decode(motion_info_b[to_gen_list[0]]['motion_tokens'])
                    goal_motion_ids = motion_begin_tag_ids + tokenizer(goal_motion_string, add_special_tokens=False).input_ids + motion_end_tag_ids

                    one_full_text += '[generate]' + '<Motion>' + goal_motion_string + '</Motion>' + '[/generate]' + '\n'
                    step_ids = generate_begin_tag_ids + goal_motion_ids + generate_end_tag_ids + nextline_tag_ids
                    one_input_ids += step_ids
                    one_target_ids += step_ids
                else:
                    # 这时候需要设计 GT-Forcing
                    motion_gt = to_gen_list[0] + '>'
                    motion_exact = '<' + to_gen_list[1]
                    # 我们期望的目标motion
                    goal_motion_string = tokenizer.decode(motion_info_b[motion_gt]['motion_tokens'])
                    goal_motion_ids = motion_begin_tag_ids + tokenizer(goal_motion_string, add_special_tokens=False).input_ids + motion_end_tag_ids
                    # 实际 reasoning trace里的 motion
                    exact_motion_string = tokenizer.decode(motion_info_b[motion_exact]['motion_tokens'])
                    exact_motion_ids = motion_begin_tag_ids + tokenizer(exact_motion_string, add_special_tokens=False).input_ids + motion_end_tag_ids
                    
                    goal_step_ids = generate_begin_tag_ids + goal_motion_ids + generate_end_tag_ids
                    exact_step_ids = generate_begin_tag_ids + exact_motion_ids + generate_end_tag_ids

                    one_full_text += '[generate]' + '<Motion>' + goal_motion_string + '</Motion>' + '[/generate]'
                    one_full_text += '[generate]' + '<Motion>' + exact_motion_string + '</Motion>' + '[/generate]' + '\n'

                    gt_forcing_position_b.append([len(one_input_ids), len(one_input_ids) + len(goal_step_ids)])   # 这个位置表示了 GT-Forcing 的位置，存这个的目的是为了后面构造特殊的attention-mask
                    one_input_ids += goal_step_ids
                    one_input_ids += exact_step_ids
                    one_input_ids += nextline_tag_ids
                    one_target_ids += goal_step_ids
                    one_target_ids += [-100] * len(exact_step_ids)      # 我们不能让模型学习错误的motion
                    one_target_ids += nextline_tag_ids

        if len(breakdown_response['think']) > 0:
            one_full_text += '</think>'
            one_full_text += '\n'
            one_input_ids += think_end_tag_ids + nextline_tag_ids
            one_target_ids += think_end_tag_ids + nextline_tag_ids

        # <answer>
        answer_begin_tag_ids = tokenizer('<answer>', add_special_tokens=False).input_ids
        answer_end_tag_ids = tokenizer('</answer>', add_special_tokens=False).input_ids
        answer = breakdown_response['answer'][0][1]
        assert answer == '<Goal_Motion>'

        goal_motion_string = tokenizer.decode(motion_info_b['<Goal_Motion>']['motion_tokens'])
        goal_motion_ids = motion_begin_tag_ids + tokenizer(goal_motion_string, add_special_tokens=False).input_ids + motion_end_tag_ids

        one_full_text += '<answer>' + '<Motion>' + goal_motion_string + '</Motion>' + '</answer>'
        one_input_ids += answer_begin_tag_ids + goal_motion_ids + answer_end_tag_ids
        one_target_ids += answer_begin_tag_ids + goal_motion_ids + answer_end_tag_ids
        
        eos_tag_ids = tokenizer('<eos>', add_special_tokens=False).input_ids
        one_full_text += '<eos>'
        one_input_ids += eos_tag_ids
        one_target_ids += eos_tag_ids
        
        assert one_full_text == tokenizer.decode(one_input_ids), f"{one_full_text}\n\n\n\n{tokenizer.decode(one_input_ids)}"
        assert len(one_input_ids) == len(one_target_ids)
        
        

        gt_forcing_position_all.append(gt_forcing_position_b) 
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))

        one_attention_mask = torch.ones(len(one_input_ids), len(one_input_ids))
        # 应用因果掩码（上三角为0，防止关注未来 token）
        one_attention_mask = torch.tril(one_attention_mask)  # 下三角矩阵，保留 1，未来位置为 0
        # 针对GT-forcing修改attention mask
        for position in gt_forcing_position_b:
            st, ed = position
            one_attention_mask[ed: , st:ed] = 0
        batch_attention_mask.append(one_attention_mask)

        # if bs == 0:
    
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()

    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]

    # 做Attention Mask
    max_len = input_ids.shape[1]
    attention_mask = torch.zeros(input_ids.shape[0], 1, max_len, max_len)   # B, head, L, L
    for b, attn_b in enumerate(batch_attention_mask):
        attn_b = attn_b[:max_tgt_len, :max_tgt_len]
        attention_mask[b, :, :attn_b.shape[1], :attn_b.shape[1]] = attn_b

    # attention_visualizer(attention_mask[0,0,:,:], input_ids[0], tokenizer=tokenizer, attn_start_idx=batch_prompt_len[0], target_ids=target_ids[0])

    # # # 如果模型需要负无穷来屏蔽，转换掩码
    attention_mask = attention_mask.masked_fill(attention_mask == 0, -1e10)       # 这个地方A不可以赋值为 -inf，否则在前向和反向的时候都会出现nan
    attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)  # 1 -> 0, 0 -> -inf
    attention_mask = attention_mask.to(dtype=torch.bfloat16)
    assert attention_mask.shape[-1] == input_ids.shape[1]
    # attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    # assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask



def process_cot_batch_ignore_incorrect(data, tokenizer,max_tgt_len):
    bos = tokenizer.bos_token_id
    mask_id = tokenizer.encode('<mask>', add_special_tokens=False)[0]
    if bos is None:
        # TODO:
        assert False, "No implementation"

    system_prompts = data['system_prompt']
    instructions = data['instruction']
    responses = data['response']
    breakdown_responses = data['breakdown_response']
    all_motion_info = data['motion_info']

    batch_size = len(system_prompts)
    batch_input_ids, batch_target_ids = [], []
    batch_attention_mask = []
    batch_prompt_len = []

    nextline_tag_ids = tokenizer('\n', add_special_tokens=False).input_ids
    
    gt_forcing_position_all = []
    for bs in range(batch_size):
        one_input_ids, one_target_ids = [], []
        gt_forcing_position_b = []
        one_full_text = tokenizer.decode(bos)

        one_input_ids.append(bos)
        one_target_ids.append(-100)  # do not perform loss regression on human prompt

        system_prompt = system_prompts[bs] + '\n\n'
        user_prompt = instructions[bs] + '\n\n'
        motion_info_b = all_motion_info[bs]

        # <prompt>
        full_prompt = system_prompt + user_prompt
        full_prompt_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids
        one_full_text += full_prompt
        one_input_ids += full_prompt_ids
        one_target_ids += [-100] * len(full_prompt_ids)
        prompt_len = len(one_input_ids)
        batch_prompt_len.append(prompt_len)
        # <think>
        breakdown_response = breakdown_responses[bs]
        if len(breakdown_response['think']) > 0:
            think_begin_tag_ids = tokenizer('<think>', add_special_tokens=False).input_ids
            think_end_tag_ids = tokenizer('</think>', add_special_tokens=False).input_ids

            one_full_text += '<think>'
            one_input_ids += think_begin_tag_ids
            one_target_ids += think_begin_tag_ids

        for step in breakdown_response['think']:
            mode, value = step

            if mode == 'analyze':
                analyze_begin_tag_ids = tokenizer('[analyze]', add_special_tokens=False).input_ids
                analyze_end_tag_ids = tokenizer('[/analyze]', add_special_tokens=False).input_ids
                analyze_ids = tokenizer(value, add_special_tokens=False).input_ids
                step_ids = analyze_begin_tag_ids + analyze_ids + analyze_end_tag_ids + nextline_tag_ids
                analyze_text = '[analyze]' + value + '[/analyze]' + '\n'

                one_full_text += analyze_text
                one_input_ids += step_ids
                one_target_ids += step_ids
            if mode == 'assess':
                assess_begin_tag_ids = tokenizer('[assess]', add_special_tokens=False).input_ids
                assess_end_tag_ids = tokenizer('[/assess]', add_special_tokens=False).input_ids
                assess_ids = tokenizer(value, add_special_tokens=False).input_ids
                step_ids = assess_begin_tag_ids + assess_ids + assess_end_tag_ids + nextline_tag_ids
                assess_text = '[assess]' + value + '[/assess]' + '\n'

                one_full_text += assess_text
                one_input_ids += step_ids
                one_target_ids += step_ids
            if mode == 'generate':
                generate_begin_tag_ids = tokenizer('[generate]', add_special_tokens=False).input_ids
                generate_end_tag_ids = tokenizer('[/generate]', add_special_tokens=False).input_ids
                motion_begin_tag_ids = tokenizer('<Motion>', add_special_tokens=False).input_ids
                motion_end_tag_ids = tokenizer('</Motion>', add_special_tokens=False).input_ids

                # 处理generate id
                # 在这个地方，value 必须 写成 "<Goal_Motion>" 或者 "<Goal_Motion><Intermediate_Motion_x>"的形式
                to_gen_list = value.split('><')
                if len(to_gen_list) == 1:
                    # 只有Goal_Motion
                    assert to_gen_list[0] == '<Goal_Motion>'
                    goal_motion_string = tokenizer.decode(motion_info_b[to_gen_list[0]]['motion_tokens'])
                    goal_motion_ids = motion_begin_tag_ids + tokenizer(goal_motion_string, add_special_tokens=False).input_ids + motion_end_tag_ids

                    one_full_text += '[generate]' + '<Motion>' + goal_motion_string + '</Motion>' + '[/generate]' + '\n'
                    step_ids = generate_begin_tag_ids + goal_motion_ids + generate_end_tag_ids + nextline_tag_ids
                    one_input_ids += step_ids
                    one_target_ids += step_ids
                else:
                    # 这时候需要设计 GT-Forcing
                    motion_gt = to_gen_list[0] + '>'
                    motion_exact = '<' + to_gen_list[1]
                    # 我们期望的目标motion
                    # goal_motion_string = tokenizer.decode(motion_info_b[motion_gt]['motion_tokens'])
                    # goal_motion_ids = motion_begin_tag_ids + tokenizer(goal_motion_string, add_special_tokens=False).input_ids + motion_end_tag_ids
                    # 实际 reasoning trace里的 motion
                    exact_motion_string = tokenizer.decode(motion_info_b[motion_exact]['motion_tokens'])
                    exact_motion_ids = motion_begin_tag_ids + tokenizer(exact_motion_string, add_special_tokens=False).input_ids + motion_end_tag_ids
                    
                    # goal_step_ids = generate_begin_tag_ids + goal_motion_ids + generate_end_tag_ids
                    exact_step_ids = generate_begin_tag_ids + exact_motion_ids + generate_end_tag_ids

                    # one_full_text += '[generate]' + '<Motion>' + goal_motion_string + '</Motion>' + '[/generate]'
                    one_full_text += '[generate]' + '<Motion>' + exact_motion_string + '</Motion>' + '[/generate]' + '\n'

                    # gt_forcing_position_b.append([len(one_input_ids), len(one_input_ids) + len(goal_step_ids)])   # 这个位置表示了 GT-Forcing 的位置，存这个的目的是为了后面构造特殊的attention-mask
                    # one_input_ids += goal_step_ids
                    one_input_ids += exact_step_ids
                    one_input_ids += nextline_tag_ids
                    # one_target_ids += goal_step_ids
                    one_target_ids += [-100] * len(exact_step_ids)      # 我们不能让模型学习错误的motion
                    one_target_ids += [-100] * len(nextline_tag_ids)

        if len(breakdown_response['think']) > 0:
            one_full_text += '</think>'
            one_full_text += '\n'
            one_input_ids += think_end_tag_ids + nextline_tag_ids
            one_target_ids += think_end_tag_ids + nextline_tag_ids

        # <answer>
        answer_begin_tag_ids = tokenizer('<answer>', add_special_tokens=False).input_ids
        answer_end_tag_ids = tokenizer('</answer>', add_special_tokens=False).input_ids
        answer = breakdown_response['answer'][0][1]
        assert answer == '<Goal_Motion>'

        goal_motion_string = tokenizer.decode(motion_info_b['<Goal_Motion>']['motion_tokens'])
        goal_motion_ids = motion_begin_tag_ids + tokenizer(goal_motion_string, add_special_tokens=False).input_ids + motion_end_tag_ids

        one_full_text += '<answer>' + '<Motion>' + goal_motion_string + '</Motion>' + '</answer>'
        one_input_ids += answer_begin_tag_ids + goal_motion_ids + answer_end_tag_ids
        one_target_ids += answer_begin_tag_ids + goal_motion_ids + answer_end_tag_ids
        
        eos_tag_ids = tokenizer('<eos>', add_special_tokens=False).input_ids
        one_full_text += '<eos>'
        one_input_ids += eos_tag_ids
        one_target_ids += eos_tag_ids
        # import pdb; pdb.set_trace()
        assert one_full_text == tokenizer.decode(one_input_ids)
        assert len(one_input_ids) == len(one_target_ids)

        # # check if the training data is valid
        # check_results = check_format(one_full_text.replace(full_prompt, ''))
        # total = len(check_results)
        # passed = sum(1 for _, (is_met, _) in check_results.items() if is_met)
        # assert passed == total

        # gt_forcing_position_all.append(gt_forcing_position_b) 
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))

        # one_attention_mask = torch.ones(len(one_input_ids), len(one_input_ids))
        # 应用因果掩码（上三角为0，防止关注未来 token）
        # one_attention_mask = torch.tril(one_attention_mask)  # 下三角矩阵，保留 1，未来位置为 0
        # 针对GT-forcing修改attention mask
        # for position in gt_forcing_position_b:
        #     st, ed = position
        #     one_attention_mask[ed: , st:ed] = 0
        # batch_attention_mask.append(one_attention_mask)

        # if bs == 0:
    
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()

    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]

    # 做Attention Mask
    # max_len = input_ids.shape[1]
    # attention_mask = torch.zeros(input_ids.shape[0], 1, max_len, max_len)   # B, head, L, L
    # for b, attn_b in enumerate(batch_attention_mask):
    #     attn_b = attn_b[:max_tgt_len, :max_tgt_len]
    #     attention_mask[b, :, :attn_b.shape[1], :attn_b.shape[1]] = attn_b

    # target_visualizer(input_ids=input_ids[0], tokenizer=tokenizer, attn_start_idx=0, target_ids=target_ids[0], record_name='debug_attention_ignore_incorrect')

    # # # 如果模型需要负无穷来屏蔽，转换掩码
    # attention_mask = attention_mask.masked_fill(attention_mask == 0, -1e10)       # 这个地方A不可以赋值为 -inf，否则在前向和反向的时候都会出现nan
    # attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)  # 1 -> 0, 0 -> -inf
    # attention_mask = attention_mask.to(dtype=torch.bfloat16)
    # assert attention_mask.shape[-1] == input_ids.shape[1]
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask


def target_visualizer(input_ids, target_ids, tokenizer, attn_start_idx=0, record_name='debug'):
    all_token_str = []
    all_target_ids = []
    for i, input_id in enumerate(input_ids[attn_start_idx:]):
        token_str = tokenizer.decode([input_id])
        all_token_str.append(token_str)
    for i, target_id in enumerate(target_ids[attn_start_idx:]):
        all_target_ids.append(str(target_id))
    f = open(f'/mnt/data1/yuanming/Code/Motion_Gen/Motion-Agent/{record_name}', 'w')
    f.write(tokenizer.decode(input_ids[attn_start_idx:]) + '\n')
    max_target_id_length = max([len(t_id) for t_id in all_target_ids])
    max_token_length = max([len(token_str) for token_str in all_token_str])
    print("##################################################")
    f.write("######################## Response ##########################\n")

    for i in range(len(all_token_str)):
        token_str = all_token_str[i]
        token_str = token_str + " " * (max_token_length - len(token_str))
        target_id = all_target_ids[i]
        target_id_str = target_id + " " * (max_target_id_length - len(target_id))
        line_txt = token_str + "| " + f"target: {target_id_str} | " + str(i) + " "
        pure_txt = token_str + "| " + f"target: {target_id_str} | " + str(i) + " "
        line_txt += "   " + " |"
        pure_txt += "   " + " |\n" 
        f.write(pure_txt)
        print(line_txt)
    print("##################################################")
    f.write("##################################################\n")

    f.close()

def attention_visualizer(attn, input_ids, target_ids, tokenizer, attn_start_idx=0, record_name='debug_attention_gt_forcing'):
    """
    绘制注意力图
    :param attn: [S, S]
    :param input_ids: [S]
    :return:
    """
    # Print the matrix with words as row labels
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BLACK_SQUARE = "■"
    WHITE_SQUARE = "⬚"
    all_token_str = []
    all_target_ids = []

    attn_values = attn[attn_start_idx:, attn_start_idx:]
    # valid_target_ids = 
    for i, input_id in enumerate(input_ids[attn_start_idx:]):
        token_str = tokenizer.decode([input_id])
        all_token_str.append(token_str)
    for i, target_id in enumerate(target_ids[attn_start_idx:]):
        all_target_ids.append(str(target_id))
    f = open(f'/mnt/data1/yuanming/Code/Motion_Gen/Motion-Agent/{record_name}', 'w')
    f.write(tokenizer.decode(input_ids[attn_start_idx:]) + '\n')
    max_target_id_length = max([len(t_id) for t_id in all_target_ids])
    max_token_length = max([len(token_str) for token_str in all_token_str])
    print("##################################################")
    f.write("######################## Attention for Response ##########################\n")

    for i in range(len(all_token_str)):
        token_str = all_token_str[i]
        token_str = token_str + " " * (max_token_length - len(token_str))
        target_id = all_target_ids[i]
        target_id_str = target_id + " " * (max_target_id_length - len(target_id))

        line_txt = token_str + "| " + f"target: {target_id_str} | " + str(i) + " "
        pure_txt = token_str + "| " + f"target: {target_id_str} | " + str(i) + " "
        for j, attn_value in enumerate(attn_values[i]):
            if attn_value > 0.0:
                if i == j:
                    line_txt += f"{GREEN}{BLACK_SQUARE}{RESET} "
                    pure_txt += BLACK_SQUARE + ' '
                else:
                    line_txt += BLACK_SQUARE + ' '
                    pure_txt += BLACK_SQUARE + ' '
            else:
                line_txt += WHITE_SQUARE + " "
                pure_txt += WHITE_SQUARE + ' '

        line_txt += "   " + " |"
        pure_txt += "   " + " |\n" 
        f.write(pure_txt)
        print(line_txt)
    print("##################################################")
    f.write("##################################################\n")

    f.close()



import re
from typing import Dict, List, Tuple

def check_format(output: str) -> Dict[str, Tuple[bool, str]]:
    """
    检查LLM输出是否符合指定格式要求
    
    参数:
        output: LLM生成的文本
        
    返回:
        包含每个要求检查结果的字典，键为要求描述，值为(是否满足, 详细信息)元组
    """
    results = {}
    # 要求1: 包含think和answer两部分
    think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
    
    has_think = think_match is not None
    has_answer = answer_match is not None
    
    results["包含think和answer部分"] = (
        has_think and has_answer,
        f"think部分: {'存在' if has_think else '缺失'}, answer部分: {'存在' if has_answer else '缺失'}"
    )
    
    if not has_think or not has_answer:
        return results
    
    think_content = think_match.group(1)
    answer_content = answer_match.group(1)
    
    # 要求2: think部分包含analyze, generate和assess三个子部分
    analyze_match = re.search(r'\[analyze\](.*?)\[/analyze\]', think_content, re.DOTALL)
    generate_matches = list(re.finditer(r'\[generate\](.*?)\[/generate\]', think_content, re.DOTALL))
    assess_matches = list(re.finditer(r'\[assess\](.*?)\[/assess\]', think_content, re.DOTALL))
    
    has_analyze = analyze_match is not None
    has_generate = len(generate_matches) > 0
    has_assess = len(assess_matches) > 0
    
    results["think部分包含analyze, generate和assess子部分"] = (
        has_analyze and has_generate and has_assess,
        f"analyze: {'存在' if has_analyze else '缺失'}, generate: {len(generate_matches)}个, assess: {len(assess_matches)}个"
    )
    
    # 要求3: analyze部分出现在最前面
    if has_analyze:
        analyze_start = analyze_match.start()
        content_before_analyze = think_content[:analyze_start].strip()
        analyze_is_first = len(content_before_analyze) == 0
        
        results["analyze部分出现在最前面"] = (
            analyze_is_first,
            f"analyze之前的内容: '{content_before_analyze}'" if content_before_analyze else "analyze确实在最前面"
        )
    else:
        results["analyze部分出现在最前面"] = (False, "analyze部分缺失")
    
    # 要求4: 至少一轮generate和assess过程
    has_at_least_one_pair = len(generate_matches) >= 1 and len(assess_matches) >= 1
    
    results["至少一轮generate和assess过程"] = (
        has_at_least_one_pair,
        f"generate数量: {len(generate_matches)}, assess数量: {len(assess_matches)}"
    )
    
    # 要求5: analyze, generate和assess三个子部分没有任何重叠
    no_overlap = True
    overlap_details = "各部分位置: "
    
    if has_analyze and has_generate and has_assess:
        sections = []
        
        # 收集所有部分的位置
        sections.append(("analyze", analyze_match.start(), analyze_match.end()))
        
        for i, match in enumerate(generate_matches):
            sections.append((f"generate_{i+1}", match.start(), match.end()))
        
        for i, match in enumerate(assess_matches):
            sections.append((f"assess_{i+1}", match.start(), match.end()))
        
        # 按开始位置排序
        sections.sort(key=lambda x: x[1])
        
        # 检查是否有重叠
        for i in range(len(sections) - 1):
            current_end = sections[i][2]
            next_start = sections[i+1][1]
            
            if current_end > next_start:
                no_overlap = False
                overlap_details += f"{sections[i][0]}与{sections[i+1][0]}重叠; "
            else:
                overlap_details += f"{sections[i][0]}->"
        
        if sections:
            overlap_details += sections[-1][0]
    else:
        no_overlap = False
        overlap_details = "部分子部分缺失，无法检查重叠"
    
    results["analyze, generate和assess子部分没有任何重叠"] = (no_overlap, overlap_details)
    
    # 要求6: generate子部分中包含被<Motion></Motion>包裹的motion
    generate_has_motion = []
    for i, match in enumerate(generate_matches):
        gen_content = match.group(1)
        has_motion = re.search(r'<Motion>.*?</Motion>', gen_content, re.DOTALL) is not None
        generate_has_motion.append(has_motion)
    
    all_generate_have_motion = all(generate_has_motion)
    
    results["generate子部分包含<Motion>包裹的motion"] = (
        all_generate_have_motion,
        f"共有{len(generate_matches)}个generate部分，其中{sum(generate_has_motion)}个包含Motion标签"
    )
    
    # 要求7: answer部分中包含被<Motion></Motion>包裹的motion
    answer_has_motion = re.search(r'<Motion>.*?</Motion>', answer_content, re.DOTALL) is not None
    
    results["answer部分包含<Motion>包裹的motion"] = (
        answer_has_motion,
        "answer部分包含Motion标签" if answer_has_motion else "answer部分缺少Motion标签"
    )
    
    # 额外检查: generate和assess数量是否匹配
    generate_assess_match = len(generate_matches) == len(assess_matches)
    results["generate和assess数量匹配"] = (
        generate_assess_match,
        f"generate数量: {len(generate_matches)}, assess数量: {len(assess_matches)}"
    )
    
    # 额外检查: generate和assess的顺序是否正确
    correct_order = True
    order_details = ""
    
    if generate_matches and assess_matches:
        # 找到第一个generate和第一个assess的位置
        first_gen_pos = generate_matches[0].start()
        first_ass_pos = assess_matches[0].start()
        
        # 检查是否至少有一对generate在assess之前
        if first_gen_pos > first_ass_pos:
            correct_order = False
            order_details = "第一个assess出现在第一个generate之前"
        else:
            # 检查所有generate和assess的顺序
            for i in range(min(len(generate_matches), len(assess_matches))):
                gen_pos = generate_matches[i].start()
                ass_pos = assess_matches[i].start()
                
                if gen_pos > ass_pos:
                    correct_order = False
                    order_details = f"第{i+1}个assess出现在第{i+1}个generate之前"
                    break
            
            if correct_order:
                order_details = "generate和assess顺序正确"
    else:
        correct_order = False
        order_details = "generate或assess部分缺失"
    
    results["generate和assess顺序正确"] = (correct_order, order_details)
    
    return results