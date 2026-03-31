import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## device
    parser.add_argument('--device', type=str, default='cuda:0', help='device')

    ## MotionLLM training
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size per GPU')
    parser.add_argument('--epochs-t2m', type=int, default=500, help='number of epochs for t2m')
    parser.add_argument('--epochs-m2t', type=int, default=10, help='number of epochs for m2t')
    parser.add_argument('--epochs_multitask', type=int, default=500, help='number of epochs for multitask')
    parser.add_argument('--epochs-unified', type=int, default=500, help='number of epochs for unified task')

    parser.add_argument('--training-task', type=str, default='t2m', help='training task, t2m or m2t')
    parser.add_argument('--epochs-start-val', type=int, default=70, help='number of epochs to start validation')
    parser.add_argument('--epochs-val-interval', type=int, default=3, help='number of epochs between validation')
    
    ## DeepSpeed specific
    parser.add_argument('--deepspeed', action='store_true', help='use deepspeed for training')
    parser.add_argument('--zero-stage', type=int, default=2, help='ZeRO optimization stage (0, 1, 2, 3)')
    parser.add_argument('--fp16', action='store_true', help='enable fp16 mixed precision training')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training (automatically set by launcher)')
    parser.add_argument('--local-rank', type=int, default=0, help='alternative local rank argument (automatically set by launcher)')

    ## LLM 
    parser.add_argument('--llm-backbone', type=str, default='/mnt/data1/yuanming/pretrained_models/gemma-2-2b-it', help='name of huggingface model backbone')
    parser.add_argument('--llm-tokenizer', type=str, default='', help='name of huggingface model tokenizer')

    parser.add_argument('--lora-r-t2m', type=int, default=64, help='lora_r for t2m')
    parser.add_argument('--lora-alpha-t2m', type=int, default=64, help='lora_alpha for t2m')
    parser.add_argument('--lora-r-m2t', type=int, default=32, help='lora_r for m2t')
    parser.add_argument('--lora-alpha-m2t', type=int, default=32, help='lora_alpha for m2t')
    parser.add_argument('--lora-dropout', type=float, default=0.1, help='lora_dropout')

    ## dataloader  
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='experiments', help='output directory')
    parser.add_argument('--exp-name', type=str, default='test', help='name of the experiment, will create a file inside out-dir')
    
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    parser.add_argument('--first-lora', type=str, default='shared', help='set the order of the LoRA')
    parser.add_argument('--generation-mode', type=str, default='direct', help='mode_of_generation')
    
    # Debug Mode
    parser.add_argument('--debug', action='store_true', help='use DEBUG mode or not')

    # Resume: 默认从resume_llm 中读取应该从哪一轮开始训练，但如果resume_llm中没有，则从resume_epoch中读取
    parser.add_argument('--resume-llm', type=str, help='the path to the llm checkpoint to resume training')
    parser.add_argument('--resume-epoch', type=int, default=0, help='the epoch to resume training')
    

    # Activate Token Embeddings
    parser.add_argument('--activate-token-embeds', action='store_true', help='use DEBUG mode or not')
    parser.add_argument('--activate-new-token-embeds', action='store_true', help='use DEBUG mode or not')

    # Use flash-attention
    parser.add_argument('--w-flash-attention', action='store_true', help='use flash-attn or not')

    # Used base data for unified training
    parser.add_argument('--base-datasets', default=['motion_chain_conv'], nargs='+', help='the base datasets for unified training')
    parser.add_argument('--val-tasks', default=['t2m'], nargs='+', help='tasks to be evaluated during validation')

    # Save frequency for checkpoints
    parser.add_argument('--save-frequency', type=int, default=0, help='frequency (in epochs) to save checkpoints. 0 means disabled.')
    parser.add_argument('--save-step-frequency', type=int, default=0, help='frequency (in epochs) to save checkpoints. 0 means disabled.')

    parser.add_argument('--save_all_params', action='store_true', help='whether save all parameters during training')
    
    parser.add_argument('--full-tuning', action='store_true', help='fully fine-tuning')
    parser.add_argument('--wo-lora', action='store_true', help='')
    parser.add_argument('--wo-plan', action='store_true', help='')
    parser.add_argument('--wo-answer', action='store_true', help='')
    parser.add_argument('--wo_assess_refine', action='store_true', help='')


    parser.add_argument('--wo-scale-lr', action='store_true', help='not to scale learning rate with world size')
    parser.add_argument('--skip-training', action='store_true', help='whether skipping the training phase')
    # parser.add_argument('--skip-forward', action='store_true', help='whether skipping the forward phase')

    parser.add_argument('--gt-forcing', action='store_true', help='strictly teach model to learn gt-motion rather than the wrong one')
    parser.add_argument('--ignore-incorrect', action='store_true', help='strictly ignore the incorrect motion')
    parser.add_argument('--prompt-w-response', action='store_true', help='debug or not')

    parser.add_argument('--data-json-files', type=str, default='', help='strictly ignore the incorrect motion')


    return parser.parse_args()






