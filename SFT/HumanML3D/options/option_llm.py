import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## device
    parser.add_argument('--device', type=str, default='cuda:0', help='device')

    ## LLM 
    parser.add_argument('--llm-backbone', type=str, default='/mnt/data1/yuanming/pretrained_models/gemma-2-2b-it', help='name of huggingface model backbone')
    parser.add_argument('--llm-tokenizer', type=str, default='', help='name of huggingface model backbone')

    parser.add_argument('--llm-ckpt', type=str, default='', help='path to the llm checkpoint')

    parser.add_argument('--lora-r-t2m', type=int, default=64, help='lora_r for t2m')
    parser.add_argument('--lora-alpha-t2m', type=int, default=64, help='lora_alpha for t2m')
    parser.add_argument('--lora-r-m2t', type=int, default=32, help='lora_r for m2t')
    parser.add_argument('--lora-alpha-m2t', type=int, default=32, help='lora_alpha for m2t')
    parser.add_argument('--lora-dropout', type=float, default=0.1, help='lora_dropout')
    parser.add_argument('--eval-task', type=str, default='t2m', help='evaluating task, t2m or m2t or multitask_shared')
    parser.add_argument('--eval-set', type=str, default='test', help='evaluating set: val or test')
    parser.add_argument('--dataset-return-neg-motion', action='store_true', help='return negative motion from dataset')

    # Use flash-attention
    parser.add_argument('--w-flash-attention', action='store_true', help='use flash-attn or not')

    ## dataloader  
    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--eval-batch-size', default=2, type=int, help='batch size')
    parser.add_argument('--eval-tag', default="", type=str, help='a tag for testing')
    

    parser.add_argument('--window-size', type=int, default=64, help='training motion length')

    ## optimization
    parser.add_argument('--total-iter', default=200000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000, 400000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=0.01, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss-vel', type=float, default=0.1, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons-loss', type=str, default='l2', help='reconstruction loss')
    parser.add_argument('--max-refine-round', type=int, default=5, help='max refine round')
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

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    parser.add_argument("--resume-gpt", type=str, default=None, help='resume pth for GPT')
    
    
    ## output directory 
    parser.add_argument('--prompt-file', type=str, default='/mnt/data1/yuanming/Code/Cot-Mogen-V4/src/mogen_r1/src/data/OOD_prompts/sample_idea400.txt', help='path to prompts')
    parser.add_argument('--case-study-name', type=str, default='idea400', help='path to prompts')

    parser.add_argument('--out-dir', type=str, default='experiments', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')
    parser.add_argument('--eval-repeat-times', default=20, type=int, help='repetition of evaluation')
    parser.add_argument('--early-stopping', action='store_true', help='whether using early stopping')

    parser.add_argument('--generation-mode', type=str, default='direct', help='mode_of_generation')
    parser.add_argument('--debug', action='store_true', help='debug or not')
    parser.add_argument('--prompt-w-response', action='store_true', help='debug or not')
    parser.add_argument('--batch-inference', action='store_true', help='debug or not')
    parser.add_argument('--merge-lora', action='store_true', help='debug or not')
    parser.add_argument('--do-sample', action='store_true', help='set do_sample=True or not')




    
    
    return parser.parse_args()