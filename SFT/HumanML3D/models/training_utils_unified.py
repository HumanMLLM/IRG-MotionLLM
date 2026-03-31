# NOTE: Here are the scripts for unified training
import random
import torch
from torch.nn.utils import rnn

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


def process_unified_batch(data, tokenizer,max_tgt_len):
    bos = tokenizer.bos_token_id
    mask_id = tokenizer.encode('<mask>', add_special_tokens=False)[0]
    if bos is None:
        # TODO:
        assert False, "No implementation"
    tasks = data['task_name']
    sub_tasks = data['sub_task_name']
    templates = data['template']
    batch_size = len(tasks)

    batch_input_ids, batch_target_ids = [], []

    # NOTE: build one instance
    for b in range(batch_size):
        input_template = templates[b]['input_template']
        output_template = templates[b]['output_template']

        sub_task = data['sub_task_name'][b]

        caption_ref = data['caption_ref'][b]
        caption_A = data['caption_A'][b]
        caption_B = data['caption_B'][b]
        caption_O = data['caption_O'][b]

        motion_token_A  = data['motion_tokens_A'][b]
        motion_token_B  = data['motion_tokens_B'][b]
        motion_token_O  = data['motion_tokens_O'][b]
        motion_token_ref  = data['motion_tokens_ref'][b]
        motion_string_A = tokenizer.decode(motion_token_A)
        motion_string_B = tokenizer.decode(motion_token_B)
        motion_string_O = tokenizer.decode(motion_token_O)
        motion_string_ref = tokenizer.decode(motion_token_ref)

        motion_second_ref = data['motion_second_ref'][b]
        motion_second_A = data['motion_second_A'][b]
        motion_second_B = data['motion_second_B'][b]
        motion_second_O = data['motion_second_O'][b]

        motion_len_ref = data['motion_len_ref'][b]
        motion_len_A = data['motion_len_A'][b]
        motion_len_B = data['motion_len_B'][b]
        motion_len_O = data['motion_len_O'][b]

        # Mask motion: 这里我们假设只需要mask motion_ref, mask率为40%
        if len(motion_token_ref) > 0:
            mask_ratio = 0.4
            num_mask_token = int(len(motion_token_ref) * mask_ratio)
            start_mask_idx = random.randint(0, len(motion_token_ref)-num_mask_token-1)
            end_mask_idx = start_mask_idx + num_mask_token
            masked_motion_token_ref = [tk if (t_id < start_mask_idx or t_id >= end_mask_idx) else mask_id for t_id, tk in enumerate(motion_token_ref)]
            masked_motion_string_ref = tokenizer.decode(masked_motion_token_ref)

            # Split Motion: 这里我们暂定 split率为30%
            split_ratio = 0.3
            num_split_token = int(len(motion_token_ref) * split_ratio)
            splited_motion_token_ref = motion_token_ref[:-num_split_token]
            splited_motion_string_ref = tokenizer.decode(splited_motion_token_ref)
        else:
            splited_motion_string_ref = ""
            masked_motion_string_ref = ""


        one_input_ids, one_target_ids = [], []
        
        one_input_ids.append(bos)
        one_target_ids.append(-100)  # do not perform loss regression on human prompt
        texts = ''
        prompt = "Below is an description of a task togeter with the inputs that provides further context. Write a response that appropriately completes the request.\n\n"
        instruction_meta = '<Instruction_Placeholder> \n\n Response:'
        response_meta = '<Response_Placeholder>'
                        # 'Undertsanding/General'
        if sub_task in ['Understanding/General', 'Understanding/Other_info','Generation/General', "Generation/Coarse/Caption-to-Motion", "Understanding/Coarse/Motion-to-Caption","Understanding/Coarse/Motion-to-Framelen","Understanding/Coarse/Motion-to-Secondlen", "Understanding/Coarse/Motion-Secondlen-to-Caption","Understanding/Coarse/Motion-Framelen-to-Caption", "Generation/Coarse/Caption-Secondlen-to-Motion", "Generation/Coarse/Caption-Framelen-to-Motion"]:
            instruction = input_template.replace('<Caption_Placeholder>', caption_ref).replace('<Second_Placeholder>', str(motion_second_ref)).replace('<Frame_Placeholder>', str(motion_len_ref)).replace('<Framelen_Placeholder>', str(motion_len_ref))
            response = output_template.replace('<Caption_Placeholder>', caption_ref).replace('<Second_Placeholder>', str(motion_second_ref)).replace('<Frame_Placeholder>', str(motion_len_ref)).replace('<Framelen_Placeholder>', str(motion_len_ref))
            instruction = instruction.replace('<Motion_Placeholder>', '<Motion>'+motion_string_ref+'</Motion>')
            response = response.replace('<Motion_Placeholder>', '<Motion>'+motion_string_ref+'</Motion>')
        elif sub_task in ['Editing/General']:
            instruction = input_template.replace('<Motion_Placeholder>', '<Motion>'+motion_string_A+'</Motion>')
            response = output_template.replace('<Motion_Placeholder>', '<Motion>'+motion_string_B+'</Motion>')
        elif sub_task in ['Prediction/Masked','Prediction/Splited',"Prediction/Coarse/Motion-masked-Caption-to-Motion","Prediction/Coarse/Motion-masked-to-Motion", "Prediction/Coarse/Motion-former-Caption-to-Motion-latter", "Prediction/Coarse/Motion-former-to-Motion-latter"]:
            instruction = input_template.replace('<Motion_Placeholder_Masked>', '<Motion>'+masked_motion_string_ref+'</Motion>').replace('<Motion_Placeholder_s1>', '<Motion>'+splited_motion_string_ref+'</Motion>').replace('<Caption_Placeholder>', caption_ref)
            response = output_template.replace('<Motion_Placeholder>', '<Motion>'+motion_string_ref+'</Motion>').replace('<Motion_Placeholder_s2>', '<Motion>'+motion_string_ref+'</Motion>')
        elif sub_task in ['Assessment/Fine/General/Direction/Point_wise/Brief', 'Assessment/Fine/General/BodyPart/Point_wise/Brief', 'Assessment/Fine/General/Order/Point_wise/Brief', 'Assessment/Fine/General/State/Point_wise/Brief', 'Assessment/Fine/General/Frequency/Point_wise/Brief']:
            instruction = input_template.replace('<Caption_Placeholder>', caption_A).replace('<Motion_Placeholder>', '<Motion>'+motion_string_ref+'</Motion>')
            response = output_template.split('>', maxsplit=1)[-1].split('</')[0]
        else:
            assert 1==2
        assert not 'Placeholder' in instruction
        assert not 'Placeholder' in response
        
        instruction_post = instruction_meta.replace('<Instruction_Placeholder>', instruction)
        response_post = response_meta.replace('<Response_Placeholder>', response)
        
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