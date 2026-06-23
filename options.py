import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=6,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--allweather_lq', type=str, default='/home/liu/LIUYUHUI/snowdata/allweather/lq/', help='LQ images path')
parser.add_argument('--allweather_gt', type=str, default='/home/liu/LIUYUHUI/snowdata/allweather/gt/', help='GT images path')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="promptir",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 1,help = "Number of GPUs to use for training")

parser.add_argument("--experiment_name", type=str, default="", help="Experiment identifier (used for log/ckpt/results subfolders when non-empty)")
parser.add_argument("--lambda_cls", type=float, default=0.1, help="Weight for auxiliary weather classification loss")
parser.add_argument("--log_dir", type=str, default="logs", help="Root directory for TensorBoard logs")
parser.add_argument("--results_dir", type=str, default="results", help="Root directory for inference outputs")

options = parser.parse_args()
