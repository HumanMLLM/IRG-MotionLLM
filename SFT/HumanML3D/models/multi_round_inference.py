import torch
from torch.nn.utils import rnn
@torch.inference_mode()
def multi_round_inference_engine(eval_model, text_processor, task_type, input_info, device, max_refine_round=5, do_sample_=False, verbose=False, return_inter_motions=False):
    eval_model.eval()

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
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
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
        completion_only_scores = torch.stack(outputs.scores).transpose(0,1)
        for j, score in enumerate(completion_only_scores):
            answer_scores_j, think_scores_j, st_ans_pos_j, ed_ans_pos_j = eval_model.extract_answer_scores(score, eval_model.tokenizer, return_position=True, return_prefix_scores=True)
            think_ids_j = outputs.sequences[j][input_ids.shape[1]:]
            all_think_motion_pos_j = eval_model.extract_motions_from_scores_w_ids(think_ids_j)
            try:
                final_motion_pos = all_think_motion_pos_j[-1]
                final_motion_ids = think_ids_j[final_motion_pos[0]: final_motion_pos[1]+1]
                final_motion_scores = think_scores_j[final_motion_pos[0]: final_motion_pos[1]+1]
            except:
                final_motion_ids = think_ids_j[0: -1]
                final_motion_scores = think_scores_j[0: -1]
            final_motion_tokens = eval_model.get_motion_from_scores_w_ids(final_motion_scores, final_motion_ids) 

            if return_inter_motions:
                all_think_motion_ids_j = []
                all_think_motion_tokens_j = []
                
                for g, pos in enumerate(all_think_motion_pos_j):
                    think_motion_ids_g = think_ids_j[pos[0]: pos[1]+1]
                    think_motion_scores_g = think_scores_j[pos[0]: pos[1]+1]
                    think_motion_tokens_g = eval_model.get_motion_from_scores_w_ids(think_motion_scores_g, think_motion_ids_g) 
                    all_think_motion_ids_j.append(think_motion_ids_g)
                    all_think_motion_tokens_j.append(think_motion_tokens_g)
                results[j]['think_motion_tokens'] = all_think_motion_tokens_j
            results[j]['motion_tokens'] = final_motion_tokens
            results[j]['num_gens'] = len(all_think_motion_pos_j)
    elif task_type in ['unified_mogen_cot_v3_w_max_rounds']:
        input_ids, attention_mask = text_processor.build_test_context(test_task='unified_mogen_cot_v3', inputs=input_info, tokenizer=eval_model.tokenizer)
        bs = input_ids.shape[0]
        do_sample = do_sample_
        # Generate outputs for the batch
        outputs = eval_model.llm.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_length=1300,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
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
        completion_only_scores = torch.stack(outputs.scores).transpose(0,1)
        for j, score in enumerate(completion_only_scores):
            answer_scores_j, think_scores_j, st_ans_pos_j, ed_ans_pos_j = eval_model.extract_answer_scores(score, eval_model.tokenizer, return_position=True, return_prefix_scores=True)
            think_ids_j = outputs.sequences[j][input_ids.shape[1]:]
            all_think_motion_pos_j = eval_model.extract_motions_from_scores_w_ids(think_ids_j)
            try:
                if len(all_think_motion_pos_j) > max_refine_round:
                    final_motion_pos = all_think_motion_pos_j[:max_refine_round]
                else:
                    final_motion_pos = all_think_motion_pos_j[-1]
                final_motion_ids = think_ids_j[final_motion_pos[0]: final_motion_pos[1]+1]
                final_motion_scores = think_scores_j[final_motion_pos[0]: final_motion_pos[1]+1]
            except:
                final_motion_ids = think_ids_j[0: -1]
                final_motion_scores = think_scores_j[0: -1]
            final_motion_tokens = eval_model.get_motion_from_scores_w_ids(final_motion_scores, final_motion_ids) 

            if return_inter_motions:
                all_think_motion_ids_j = []
                all_think_motion_tokens_j = []
                
                for g, pos in enumerate(all_think_motion_pos_j):
                    think_motion_ids_g = think_ids_j[pos[0]: pos[1]+1]
                    think_motion_scores_g = think_scores_j[pos[0]: pos[1]+1]
                    think_motion_tokens_g = eval_model.get_motion_from_scores_w_ids(think_motion_scores_g, think_motion_ids_g) 
                    all_think_motion_ids_j.append(think_motion_ids_g)
                    all_think_motion_tokens_j.append(think_motion_tokens_g)
                results[j]['think_motion_tokens'] = all_think_motion_tokens_j
            results[j]['motion_tokens'] = final_motion_tokens
            results[j]['num_gens'] = len(all_think_motion_pos_j)
    elif task_type in ['unified_mogen_cot_v3_random_flip']:
        # import pdb; pdb.set_trace()
        input_ids, attention_mask = text_processor.build_test_context(test_task='unified_mogen_cot_v3', inputs=input_info, tokenizer=eval_model.tokenizer)
        bs = input_ids.shape[0]
        do_sample = do_sample_
        # Generate outputs for the batch
        outputs = eval_model.llm.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_length=1300,
            do_sample=do_sample,
            eos_token_id=[eval_model.tokenizer.eos_token_id, eval_model.tokenizer.encode('</Motion>', add_special_tokens=False)[0]],
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
        )
        out_ids_w_neg_motion = []
        all_neg_motion_ids = []
        # import pdb; pdb.set_trace()
        for b in range(bs):
            out_ids_b = outputs.sequences[b]
            bos_idx = torch.where(out_ids_b == eval_model.tokenizer.bos_token_id)[0]
            # eos_idx = torch.where(out_ids_b == eval_model.tokenizer.eos_token_id)[0]
            eog_idx = torch.where(out_ids_b == eval_model.tokenizer.encode('</Motion>', add_special_tokens=False)[0])[0]
            out_ids_b = out_ids_b[bos_idx[0]: eog_idx[0]+1]
            init_gen_idx = torch.where(out_ids_b == eval_model.tokenizer.encode('<Motion>', add_special_tokens=False)[0])[0]

            # out_texts.append(eval_model.tokenizer.decode(outputs.sequences[b], skip_special_tokens=True))
            # out_texts_till_analysis.append(eval_model.tokenizer.decode(outputs.sequences[b], skip_special_tokens=True).split('<Motion>')[0])
            neg_motion_string = '<Motion>' + eval_model.tokenizer.decode(input_info['motions'][b]['<Neg_Motion>']) + '</Motion>'
            neg_motion_ids = torch.LongTensor(eval_model.tokenizer.encode(neg_motion_string,  add_special_tokens=False)).to(out_ids_b.device)
            
            all_neg_motion_ids.append(neg_motion_ids[1:-1]) # ignore the <Motion> and </Motion>

            out_ids_w_neg_motion_b = torch.cat([out_ids_b[:init_gen_idx[0]], neg_motion_ids])
            out_ids_w_neg_motion.append(out_ids_w_neg_motion_b)

        out_ids_w_neg_motion_flipped = [seq.flip(0) for seq in out_ids_w_neg_motion]
        new_input_ids_w_neg_motion_flipped = rnn.pad_sequence(out_ids_w_neg_motion_flipped, batch_first=True, padding_value=eval_model.tokenizer.pad_token_id)
        new_input_ids_w_neg_motion = new_input_ids_w_neg_motion_flipped.flip(1)
        # import pdb; pdb.set_trace()
        new_attention_mask_w_neg_motion = new_input_ids_w_neg_motion.ne(eval_model.tokenizer.pad_token_id).long()
        outputs = eval_model.llm.generate(
            new_input_ids_w_neg_motion.to(device),
            attention_mask=new_attention_mask_w_neg_motion.to(device),
            max_length=1300,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
        
        results = []
        for b in range(bs):
            results.append({
                'input_text': eval_model.tokenizer.decode(new_input_ids_w_neg_motion[b], skip_special_tokens=True),
                'best_beam_text': eval_model.tokenizer.decode(outputs.sequences[b][new_input_ids_w_neg_motion.shape[1]: ], skip_special_tokens=True),
            })
        total_think_motion_tokens = []
        total_think_motion_pos = []
        completion_only_scores = torch.stack(outputs.scores).transpose(0,1)
        # import pdb; pdb.set_trace()
        for j, score in enumerate(completion_only_scores):
            answer_scores_j, think_scores_j, st_ans_pos_j, ed_ans_pos_j = eval_model.extract_answer_scores(score, eval_model.tokenizer, return_position=True, return_prefix_scores=True)
            think_ids_j = outputs.sequences[j][new_input_ids_w_neg_motion.shape[1]:]
            all_think_motion_pos_j = eval_model.extract_motions_from_scores_w_ids(think_ids_j)
            try:
                final_motion_pos = all_think_motion_pos_j[-1]
                if final_motion_pos == [0, -1]:
                    final_motion_tokens = all_neg_motion_ids[j] - (len(eval_model.tokenizer)-(eval_model.args.nb_code)) 
                    # import pdb; pdb.set_trace()
                else:
                    final_motion_ids = think_ids_j[final_motion_pos[0]: final_motion_pos[1]+1]
                    final_motion_scores = think_scores_j[final_motion_pos[0]: final_motion_pos[1]+1]
                    final_motion_tokens = eval_model.get_motion_from_scores_w_ids(final_motion_scores, final_motion_ids) 
            except:
                final_motion_ids = think_ids_j[0: -1]
                final_motion_scores = think_scores_j[0: -1]
                final_motion_tokens = eval_model.get_motion_from_scores_w_ids(final_motion_scores, final_motion_ids) 

            if return_inter_motions:
                all_think_motion_ids_j = []
                all_think_motion_tokens_j = []
                
                for g, pos in enumerate(all_think_motion_pos_j):
                    think_motion_ids_g = think_ids_j[pos[0]: pos[1]+1]
                    think_motion_scores_g = think_scores_j[pos[0]: pos[1]+1]
                    think_motion_tokens_g = eval_model.get_motion_from_scores_w_ids(think_motion_scores_g, think_motion_ids_g) 
                    all_think_motion_ids_j.append(think_motion_ids_g)
                    all_think_motion_tokens_j.append(think_motion_tokens_g)
                results[j]['think_motion_tokens'] = all_think_motion_tokens_j
            results[j]['motion_tokens'] = final_motion_tokens
            results[j]['num_gens'] = len(all_think_motion_pos_j)
        # import pdb; pdb.set_trace()
    elif task_type in ['pure_random_flip']:
        # # import pdb; pdb.set_trace()
        input_ids, attention_mask = text_processor.build_test_context(test_task='unified_mogen_cot_v3', inputs=input_info, tokenizer=eval_model.tokenizer)
        bs = input_ids.shape[0]
        # # 随机选择一个测试的样本来破坏原本的推理路径 
        out_ids_w_neg_motion = []
        all_neg_motion_ids = []
        # import pdb; pdb.set_trace()
        for b in range(bs):
            neg_motion_string = '<Motion>' + eval_model.tokenizer.decode(input_info['motions'][b]['<Neg_Motion>']) + '</Motion>'
            neg_motion_ids = torch.LongTensor(eval_model.tokenizer.encode(neg_motion_string,  add_special_tokens=False)).to(device)
            
            all_neg_motion_ids.append(neg_motion_ids[1:-1] - (len(eval_model.tokenizer)-(eval_model.args.nb_code)) ) # ignore the <Motion> and </Motion>
        
        results = []
        for b in range(bs):
            results.append({
                'input_text': "",
                'best_beam_text': "",
                'motion_tokens': all_neg_motion_ids[b]
            })
    elif task_type in ['unified_mogen_cot_v3_initial_gen']:
        input_ids, attention_mask = text_processor.build_test_context(test_task='unified_mogen_cot_v3', inputs=input_info, tokenizer=eval_model.tokenizer)
        bs = input_ids.shape[0]
        do_sample = do_sample_
        # Generate outputs for the batch
        outputs = eval_model.llm.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_length=1300,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
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
        completion_only_scores = torch.stack(outputs.scores).transpose(0,1)
        for j, score in enumerate(completion_only_scores):
            answer_scores_j, think_scores_j, st_ans_pos_j, ed_ans_pos_j = eval_model.extract_answer_scores(score, eval_model.tokenizer, return_position=True, return_prefix_scores=True)
            think_ids_j = outputs.sequences[j][input_ids.shape[1]:]
            all_think_motion_pos_j = eval_model.extract_motions_from_scores_w_ids(think_ids_j)
            try:
                final_motion_pos = all_think_motion_pos_j[0]
                final_motion_ids = think_ids_j[final_motion_pos[0]: final_motion_pos[1]+1]
                final_motion_scores = think_scores_j[final_motion_pos[0]: final_motion_pos[1]+1]
            except:
                final_motion_ids = think_ids_j[0: -1]
                final_motion_scores = think_scores_j[0: -1]
            final_motion_tokens = eval_model.get_motion_from_scores_w_ids(final_motion_scores, final_motion_ids) 

            if return_inter_motions:
                all_think_motion_ids_j = []
                all_think_motion_tokens_j = []
                
                for g, pos in enumerate(all_think_motion_pos_j):
                    think_motion_ids_g = think_ids_j[pos[0]: pos[1]+1]
                    think_motion_scores_g = think_scores_j[pos[0]: pos[1]+1]
                    think_motion_tokens_g = eval_model.get_motion_from_scores_w_ids(think_motion_scores_g, think_motion_ids_g) 
                    all_think_motion_ids_j.append(think_motion_ids_g)
                    all_think_motion_tokens_j.append(think_motion_tokens_g)
                results[j]['think_motion_tokens'] = all_think_motion_tokens_j
            results[j]['motion_tokens'] = final_motion_tokens
            results[j]['num_gens'] = len(all_think_motion_pos_j)
        
    


    elif task_type == 'analysis_guided_t2m':
        """
        Prompt -> Analysis -> Motion
        """
        # Step 1: Prompt Analyze
        input_ids, attention_mask = text_processor.build_test_context(test_task='prompt_analysis', inputs=input_info, tokenizer=eval_model.tokenizer)
        bs = input_ids.shape[0]
        do_sample = False
        # Generate outputs for the batch
        outputs = eval_model.llm.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_length=1024,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
        )
        analysis_ids = outputs.sequences[:, input_ids.shape[1]:]
        analysis_texts = eval_model.tokenizer.batch_decode(analysis_ids, skip_special_tokens=True)
        if verbose:
            print('=' * 50)
            print('Step 1: Prompt Analyze')
            print('-' * 50)
            print(eval_model.tokenizer.batch_decode(input_ids)[0])
            print('-' * 50)
            print(analysis_texts[0])
            print('=' * 50)
        # Step 2: Analysis-2-Motion
        input_info_s2 = {'texts': [{'<Goal_Caption>': analysis_texts[b].lower()} for b in range(bs)]}
        input_ids, attention_mask = text_processor.build_test_context(test_task='t2m', inputs=input_info_s2, tokenizer=eval_model.tokenizer)
        bs = input_ids.shape[0]
        do_sample = False
        # Generate outputs for the batch
        outputs = eval_model.llm.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_length=1024,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
        results = eval_model.post_process_final_motion_output(outputs, input_ids, bs, return_dict=True)
        if verbose:
            print('=' * 50)
            print('Step 2: Analysis-guided Motion Generation')
            print('-' * 50)
            tknz = eval_model.tokenizer
            print(tknz.batch_decode(input_ids)[0])
            print('-' * 50)
            print(results[0])
            print('=' * 50)

    elif task_type == 'direct_generation-instructed_refinement':
        """
        Prompt -> Motion -> Refinement Instruction -> Refined-Motion
        """
        # Step 1: Motion Generation
        input_ids, attention_mask = text_processor.build_test_context(test_task='t2m', inputs=input_info, tokenizer=eval_model.tokenizer)
        bs = input_ids.shape[0]
        do_sample = False
        # Generate outputs for the batch
        outputs = eval_model.llm.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_length=1024,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
        results_step1 = eval_model.post_process_final_motion_output(outputs, input_ids, bs, return_dict=True)
        final_results = results_step1
        
        if verbose:
            print('=' * 50)
            print('Step 1: Initial Motion Generation')
            print('-' * 50)
            tknz = eval_model.tokenizer
            print(tknz.batch_decode(input_ids)[0])
            print('-' * 50)
            print(results_step1[0])
            print('=' * 50)

        refine_signal = [True for b in range(bs)]    # 用来标识是否需要refine
        for refine_round in range(max_refine_round):
            previous_motion_tokens = [res['motion_tokens']for res in final_results]
            previous_motion_strings = [res['best_beam_text'].split('</Motion>')[0].split('<Motion>')[-1] for res in final_results]

            # Get Refinement Instruction
            input_info['motions'] = [{'<Generated_Motion>': previous_motion_strings[b]} for b in range(bs)]
            input_ids, attention_mask = text_processor.build_test_context(test_task='refine_instruction', inputs=input_info, tokenizer=eval_model.tokenizer)
            bs = input_ids.shape[0]
            do_sample = False
            # Generate outputs for the batch
            outputs = eval_model.llm.generate(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=1024,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=False,
                use_cache=True,
            )
            # check the instruction 
            refine_instructions = eval_model.tokenizer.batch_decode(outputs.sequences[:, input_ids.shape[1]:], skip_special_tokens=True)
            refine_signal_update = [refine_signal[b] and (not refine_instructions[b]=='No refinements needed.') for b in range(bs)]
            if verbose:
                print('=' * 50)
                print('Step 2: Instructed Refinement')
                print(f'Round {refine_round+1} / {max_refine_round}')
                print('Motion Refinement Instructing')
                print('-' * 50)
                print(refine_signal_update)
                print('-' * 50)
                print(eval_model.tokenizer.batch_decode(input_ids)[0])
                print('-' * 50)
                print(refine_instructions)
                print('=' * 50)
            
            refine_signal = refine_signal_update

            

            if not any(refine_signal):
                if verbose:
                    print('=' * 50)
                    print(refine_signal)
                    print('-' * 50)
                    print('Early stop refinement')
                    print('=' * 50)
                break
            
            # Refine motion
            for b in range(bs):
                input_info['texts'][b]['<Refinement_Instruction>'] = refine_instructions[b]

            input_ids, attention_mask = text_processor.build_test_context(test_task='instructed_motion_refinement', inputs=input_info, tokenizer=eval_model.tokenizer)
            bs = input_ids.shape[0]
            do_sample = False
            # Generate outputs for the batch
            outputs = eval_model.llm.generate(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=1024,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
            refined_results = eval_model.post_process_final_motion_output(outputs, input_ids, bs, return_dict=True)
            if verbose:
                print('=' * 50)
                print('Step 2: Instructed Refinement')
                print(f'Round {refine_round+1} / {max_refine_round}')
                print('Motion Refinement')
                print('-' * 50)
                print(refine_signal)
                print('-' * 50)
                print(eval_model.tokenizer.batch_decode(input_ids)[0])
                print('-' * 50)
                print(refined_results[0])
                print('=' * 50)
            
            final_results = [refined_results[b] if refine_signal[b] else final_results[b] for b in range(bs)]

        results = final_results


    elif task_type == 'analysis_generation-instructed_refinement':
        """
        Prompt -> Analysis -> Motion -> Refinement Instruction -> Refined-Motion
        """
        # Step 1: Prompt Analyze
        input_ids, attention_mask = text_processor.build_test_context(test_task='prompt_analysis', inputs=input_info, tokenizer=eval_model.tokenizer)
        bs = input_ids.shape[0]
        do_sample = False
        # Generate outputs for the batch
        outputs = eval_model.llm.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_length=1024,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
        )
        analysis_ids = outputs.sequences[:, input_ids.shape[1]:]
        analysis_texts = eval_model.tokenizer.batch_decode(analysis_ids, skip_special_tokens=True)
        if verbose:
            print('=' * 50)
            print('Step 1: Prompt Analyze')
            print('-' * 50)
            print(eval_model.tokenizer.batch_decode(input_ids)[0])
            print('-' * 50)
            print(analysis_texts[0])
            print('=' * 50)
        # Step 2: Analysis-2-Motion
        input_info_s2 = {'texts': [{'<Goal_Caption>': analysis_texts[b].lower()} for b in range(bs)]}
        input_ids, attention_mask = text_processor.build_test_context(test_task='t2m', inputs=input_info_s2, tokenizer=eval_model.tokenizer)
        bs = input_ids.shape[0]
        do_sample = False
        # Generate outputs for the batch
        outputs = eval_model.llm.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_length=1024,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
        results = eval_model.post_process_final_motion_output(outputs, input_ids, bs, return_dict=True)
        if verbose:
            print('=' * 50)
            print('Step 2: Analysis-guided Motion Generation')
            print('-' * 50)
            tknz = eval_model.tokenizer
            print(tknz.batch_decode(input_ids)[0])
            print('-' * 50)
            print(results[0])
            print('=' * 50)
        
        final_results = results
        refine_signal = [True for b in range(bs)]    # 用来标识是否需要refine
        for refine_round in range(max_refine_round):
            previous_motion_tokens = [res['motion_tokens']for res in final_results]
            previous_motion_strings = [res['best_beam_text'].split('</Motion>')[0] for res in final_results]

            # Get Refinement Instruction
            input_info['motions'] = [{'<Generated_Motion>': previous_motion_strings[b]} for b in range(bs)]
            input_ids, attention_mask = text_processor.build_test_context(test_task='refine_instruction', inputs=input_info, tokenizer=eval_model.tokenizer)
            bs = input_ids.shape[0]
            do_sample = False
            # Generate outputs for the batch
            outputs = eval_model.llm.generate(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=1024,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=False,
                use_cache=True,
            )
            # check the instruction 
            refine_instructions = eval_model.tokenizer.batch_decode(outputs.sequences[:, input_ids.shape[1]:], skip_special_tokens=True)
            refine_signal_update = [refine_signal[b] and (not refine_instructions[b]=='No refinements needed.') for b in range(bs)]
            if verbose:
                print('=' * 50)
                print('Step 3: Instructed Refinement')
                print(f'Round {refine_round+1} / {max_refine_round}')
                print('Motion Refinement Instructing')
                print('-' * 50)
                print(refine_signal_update)
                print('-' * 50)
                print(eval_model.tokenizer.batch_decode(input_ids)[0])
                print('-' * 50)
                print(refine_instructions)
                print('=' * 50)
            
            refine_signal = refine_signal_update

            

            if not any(refine_signal):
                if verbose:
                    print('=' * 50)
                    print(refine_signal)
                    print('-' * 50)
                    print('Early stop refinement')
                    print('=' * 50)
                break
            
            # Refine motion
            for b in range(bs):
                input_info['texts'][b]['<Refinement_Instruction>'] = refine_instructions[b]

            input_ids, attention_mask = text_processor.build_test_context(test_task='instructed_motion_refinement', inputs=input_info, tokenizer=eval_model.tokenizer)
            bs = input_ids.shape[0]
            do_sample = False
            # Generate outputs for the batch
            outputs = eval_model.llm.generate(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=1024,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
            refined_results = eval_model.post_process_final_motion_output(outputs, input_ids, bs, return_dict=True)
            if verbose:
                print('=' * 50)
                print('Step 3: Instructed Refinement')
                print(f'Round {refine_round+1} / {max_refine_round}')
                print('Motion Refinement')
                print('-' * 50)
                print(refine_signal)
                print('-' * 50)
                print(eval_model.tokenizer.batch_decode(input_ids)[0])
                print('-' * 50)
                print(refined_results[0])
                print('=' * 50)
            
            final_results = [refined_results[b] if refine_signal[b] else final_results[b] for b in range(bs)]

        results = final_results
        
    elif task_type == 'm2t':
        """
        Motion -> Caption
        """
        input_ids, attention_mask = text_processor.build_test_context(test_task=task_type, inputs=input_info, tokenizer=eval_model.tokenizer)
        bs = input_ids.shape[0]
        do_sample = do_sample_

        outputs = eval_model.llm.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_length=256,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
        # import pdb; pdb.set_trace()
        
        results = eval_model.tokenizer.batch_decode(outputs.sequences[:, input_ids.shape[1]:], skip_special_tokens=True)
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
    return results