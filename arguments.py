import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    #new_added
    parser.add_argument(
        '--template', type=tuple, default=(6,6)) #(6,6) for negative sentiment, 4,4 for trans&repe
    parser.add_argument(
        '--model_name_or_path', type=str, default='/home/lwd/gpt2-large')
    parser.add_argument(
        '--sent_reward_model', type=str, default='/home/lwd/quark/Sentiment/checkpoint/fudge/disc_tuning_positive_temperature0.01_scope_50_epoch_5_f1_0.88_(2,2).ckpt')
    parser.add_argument("--topic_reward_model", type=str, default='/home/lwd/quark/data/double/checkpoint/disc_tuning_positive_temperature0.01_scope_50_epoch_7_f1_0.87_(2,2).ckpt')

    parser.add_argument(
        '--device', type=str, default='cuda')
    parser.add_argument("--pseudo_token", type=str, default='xxx')
    parser.add_argument("--lstm_dropout", type=float, default=0.0)
    parser.add_argument("--max_prompt_length", type=int, default=8) # 32 for translation, 8 for sentiment, 16 for toy
    parser.add_argument("--ranking_scope", type=int, default=50)
    parser.add_argument("--source_mode", type=str, default='neutral')
    parser.add_argument("--target_mode", type=str, default='positive')

    # dataset
    parser.add_argument(
        '--output-dir', type=str, default='outputs')
    parser.add_argument(
        '--dataset-train', type=str, default='data/toxicity/train.jsonl',
        help='JSONL file containing train prompts. Each row must contain a prompt at `row["prompt"]["text"]`.')
    parser.add_argument(
        '--dataset-val', type=str, default='data/toxicity/val.jsonl',
        help='JSONL file containing dev prompts. Each row must contain a prompt at `row["prompt"]["text"]`.')
    parser.add_argument(
        '--perspective-rate-limit', type=int, default=135, help='number of perspective call per second')

    # reward
    parser.add_argument(
        '--n_extra_tokens', type=int, default=5, help='number of reward categorization')
    parser.add_argument(
        '--horizon', type=float, default=2500, help='horizon value in adaptive controller')
    # KL term
    parser.add_argument(
        '--kl_coef', type=float, default=0.02, help='coefficient for KL term in reward,0.05 for sentiment') # 0.06 for sentiment, 0.1 for formal
    parser.add_argument(
        '--adaptive_kl', action='store_true', default=False, help='whether to use adaptive KL controller')
    parser.add_argument(
        '--target_kl', type=float, default=3, help='target value in adaptive KL controller')
    # entropy term
    parser.add_argument(
        '--entropy_coef', type=float, default=0.08, help='coefficient for entropy term in reward') #0.06 for sentiment,translation
    parser.add_argument(
        '--adaptive_entropy', action='store_true', default=False, help='whether to use adaptive entropy controller')
    parser.add_argument(
        '--target_entropy', type=float, default=40, help='target value in adaptive entropy controller')

    # policy
    parser.add_argument(
        '--init-model', type=str, default='/home/lwd/gpt2-large', help='language model used for policy.')
    parser.add_argument(
        '--ref-model', type=str, default='/home/lwd/gpt2-large', help='language model used for reference policy.')
    parser.add_argument(
        '--response-length', type=int, default=64, help='number of tokens to generate for each prompt.')
    parser.add_argument(
        '--temperature', type=float, default=1.0, help='temperature for sampling policy.')

    # training˚
    parser.add_argument(
        '--total-episodes', type=int, default=3000000, help='total number of episodes')
    parser.add_argument(
        '--lr', type=float, default=1e-5, help='learning rate')  # sentiment translation 1e-5
    parser.add_argument(
        '--num_warmup_steps', type=int, default=500, help='number of warmup steps in lr scheduler')
    parser.add_argument(
        '--clip_grad', action='store_true', default=False, help='whether to clip gradient')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5, help='maximum norm of gradients ')

    # generation
    parser.add_argument(
        '--num-samples', type=int, default=25, help='number of samples to generate for each prompt.')
    parser.add_argument(
        '--top-p', type=float, default=1.0, help='hyperparameter for nucleus sampling')

    # other
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=100, help='step interval to print out logs')
    parser.add_argument(
        '--save-interval', type=int, default=500, help='step interval to save model checkpoints')
    parser.add_argument(
        '--eval-interval', type=int, default=500, help='step interval to do evaluation')
    #这个之前是500 eval_internal,是为了统计step-precision暂时设为250
    parser.add_argument(
        '--sample-interval', type=int, default=1000, help='step interval to sample from current policy')
    parser.add_argument(
        '--cuda-deterministic', action='store_false', default=True,
        help="sets flags for determinism when using CUDA (potentially slow!)")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args
