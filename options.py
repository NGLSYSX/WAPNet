import argparse

parser = argparse.ArgumentParser()

# Basic settings
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200, help='maximum number of training epochs')
parser.add_argument('--batch_size', type=int, default=6, help='batch size per GPU')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--de_type', type=str, default='allweather', help='degradation type used for training')
parser.add_argument('--patch_size', type=int, default=128, help='input patch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of dataloader workers')
parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs used for training')

# Dataset paths
parser.add_argument('--data_file_dir', type=str, default='data_dir/', help='directory for data split files')
parser.add_argument('--allweather_lq', type=str, default='./datasets/allweather/lq/', help='path to low-quality images of the All-Weather dataset')
parser.add_argument('--allweather_gt', type=str, default='./datasets/allweather/gt/', help='path to ground-truth images of the All-Weather dataset')

# Output paths
parser.add_argument('--output_path', type=str, default='output/', help='output save path')
parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='directory for saving checkpoints')
parser.add_argument('--log_dir', type=str, default='logs', help='directory for saving training logs')
parser.add_argument('--results_dir', type=str, default='results', help='directory for saving inference outputs')
parser.add_argument('--experiment_name', type=str, default='wapnet_allweather', help='experiment identifier')

# Loss setting
parser.add_argument('--lambda_cls', type=float, default=0.1, help='weight for auxiliary weather classification loss')

# Optional logger name
parser.add_argument('--wblogger', type=str, default='wapnet', help='wandb project name or logger identifier')

options = parser.parse_args()
