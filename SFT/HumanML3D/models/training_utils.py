import torch
from torch.nn.utils import rnn

# Training utils
def build_one_instance_t2m(tokenizer, captions, motion):
    input_ids, target_ids = [], []
    # Handle case where tokenizer.bos_token_id is None (e.g., Qwen2.5)
    bos = tokenizer.bos_token_id
    if bos is None:
        input_ids, target_ids = build_with_chat_template_m2t(tokenizer, captions, motion)
        return input_ids, target_ids
    else:    
        input_ids.append(bos)
        target_ids.append(-100)  # do not perform loss regression on human prompt
        texts = ''
        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
        input_text = '### Input:\n' + captions + '\n\nResponse: <Motion>'
        text = prompt + instruction + input_text
        texts += text
        one_input_id = tokenizer(text, add_special_tokens=False).input_ids
        input_ids += one_input_id
        target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt

        text = '</Motion><eos>'
        one_input_id = tokenizer(text, add_special_tokens=False).input_ids
        # print(one_input_id)
        # print(one_input_id)
        # print(motion)
        input_ids += motion.tolist() + one_input_id
        target_ids += motion.tolist() + one_input_id
        return input_ids, target_ids

def build_with_chat_template_m2t(tokenizer, captions, motion):
    # 创建对话格式的消息列表
    messages = [
        {"role": "system", "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."},
        {"role": "user", "content": (
            "### Instruction:\n"
            "Generate a motion matching the following input human motion description\n\n"
            "### Input:\n" + captions + "\n\n"
        )}
    ]
    
    # 使用 apply_chat_template 构建输入序列
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # 添加 assistant 开始标记
        return_tensors=None,  # 返回列表
    )
    
    # 目标 ID 初始化（前面部分设为 -100）
    target_ids = [-100] * len(input_ids)
    
    # 添加运动 token 和结束标记
    motion_start_token_id = tokenizer.encode(
        '<Motion>', 
        add_special_tokens=False
    )
    end_tokens = tokenizer.encode(
        '</Motion>', 
        add_special_tokens=False
    ) + [tokenizer.eos_token_id]
    
    # 扩展输入序列
    input_ids += motion_start_token_id + motion.tolist() + end_tokens
    
    # 扩展目标序列（仅运动 token 和结束标记需要计算损失）
    target_ids += motion_start_token_id + motion.tolist() + end_tokens
    
    return input_ids, target_ids




def build_one_instance_m2t(tokenizer, captions, motion):
    input_ids, target_ids = [], []
    # Handle case where tokenizer.bos_token_id is None (e.g., Qwen2.5)
    # bos = tokenizer.bos_token_id
    bos = tokenizer.bos_token_id
    if not bos is None:
        input_ids.append(bos)
        target_ids.append(-100)  # do not perform loss regression on human prompt
    texts = ''
    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    instruction = "### Instruction:\nGenerate a caption matching the following input human motion token sequence.\n\n"
    input_text = '### Input:\n' + "<Motion>" + tokenizer.decode(motion) + '</Motion>' + '\n\nResponse: '
    text = prompt + instruction + input_text
    texts += text
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt

    text = '<eos>'
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    # print(one_input_id)
    # print(one_input_id)
    # print(motion)
    # input_ids += motion.tolist() + one_input_id
    # target_ids += motion.tolist() + one_input_id
    input_ids += tokenizer(captions, add_special_tokens=False).input_ids + one_input_id
    target_ids += tokenizer(captions, add_special_tokens=False).input_ids + one_input_id
    return input_ids, target_ids

def process_batch(tokenizer, batch_of_captions, max_tgt_len, batch_of_motions, training_task):
    batch_input_ids, batch_target_ids = [], []
    for caption, motion in zip(batch_of_captions, batch_of_motions):
        if training_task == 't2m':
            one_input_ids, one_target_ids = build_one_instance_t2m(tokenizer, caption, motion)
        elif training_task == 'm2t':
            one_input_ids, one_target_ids = build_one_instance_m2t(tokenizer, caption, motion)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()