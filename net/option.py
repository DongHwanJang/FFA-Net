import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
import torchvision.utils as vutils
from random import randint
warnings.filterwarnings('ignore')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=100000)
parser.add_argument('--device',type=str,default='Automatic detection')
parser.add_argument("--resume", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument('--eval_step',type=int,default=5000)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--model_dir',type=str,default='./trained_models/')
parser.add_argument('--trainset',type=str,default='its_train')
parser.add_argument('--testset',type=str,default='its_test')
parser.add_argument('--net',type=str,default='ffa')
parser.add_argument('--gps',type=int,default=3,help='residual_groups')
parser.add_argument('--blocks',type=int,default=20,help='residual_blocks')
parser.add_argument('--bs',type=int,default=16,help='batch size')
parser.add_argument('--crop',action='store_true')
parser.add_argument('--crop_size',type=int,default=240,help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche',action='store_true',help='no lr cos schedule')
parser.add_argument('--perloss',action='store_true',help='perceptual loss')

# parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
# parser.add_argument('--test_batch_size', type=int, default=48, help='testing batch size')
# parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
# parser.add_argument('--lr_step_size', type=float, default=8, help='Learning Rate. Default=0.01')
# parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument("--cuda", type=str2bool, nargs='?',
						const=True, default=True,
						help="Activate nice mode.")
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=5000, help='random seed to use. Default=123')
parser.add_argument("--desc", type=str, default="model")
parser.add_argument("--save", type=str, default="results")
parser.add_argument("--tanh", action="store_true")

parser.add_argument("--classifier_tuning", action="store_true")
parser.add_argument("--load_srcnn", type=str, default=None)
# parser.add_argument("--load_classifier", action="store_true")
parser.add_argument("--load_classifier", type=str2bool, nargs='?',
						const=True, default=True,
						help="Activate nice mode.")
# parser.add_argument("--multi", action="store_true")
parser.add_argument("--multi", type=str2bool, nargs='?',
						const=True, default=True,
						help="Activate nice mode.")
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--backbone", type=str, choices=["r18", "r50"])
parser.add_argument("--debug", action="store_true")
parser.add_argument("--load_pretrain_enhancement", action="store_true")
# parser.add_argument("--use_mse_loss", action="store_true")
parser.add_argument("--use_mse_loss", type=str2bool, nargs='?',
						const=True, default=True,
						help="Activate nice mode.")
parser.add_argument("--use_blurred_mse_loss", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
# parser.add_argument("--no_cudnn", action="store_true")
parser.add_argument("--no_cudnn", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument("--vis", action="store_true")
parser.add_argument("--reduction", type=float, default=1, help="channel reduction", choices=[0.25, 0.5, 1, 2, 4, 8, 16])

# parser.add_argument("--num_basis_func", type=int, default=2, help="Define second kernel size", choices=[2, 4, 6, 8])
# parser.add_argument("--second_kernel_size", type=int, default=1, help="Define second kernel size", choices=[1, 3, 5])
# Use like:
# python arg.py --kernel_sizes 1 3
parser.add_argument("--num_basis_func_factor", type=int, default=1, help="Define second kernel size", choices=[1, 2, 3, 4])

parser.add_argument("--refine_group_num", type=int, default=1, help="Define second kernel size", choices=[1, 2, 4, 8, 16])

####### Module
parser.add_argument("--use_skip", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument("--use_in", type=str2bool, nargs='?',
						const=True, default=True,
						help="Activate nice mode.")
parser.add_argument("--use_adain", type=str2bool, nargs='?',
						const=True, default=True,
						help="Activate nice mode.")
parser.add_argument("--use_spade", type=str2bool, nargs='?',
						const=True, default=True,
						help="Activate nice mode.")

####### IN_AdaIN_SPADE opt
parser.add_argument("--use_in_conv", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument("--use_spade_conv", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument("--use_adain_conv", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument("--use_out_conv", type=str2bool, nargs='?',
						const=True, default=True,
						help="Activate nice mode.")
parser.add_argument("--use_front_conv", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument("--add_spade_adain_front_conv", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.") # not important value

#use_in_conv = False, use_spade_conv=False, use_out_conv=True
parser.add_argument("--lambda_adain_regul_loss", type=float, default=0)
parser.add_argument("--lambda_spade_regul_loss", type=float, default=0)
parser.add_argument("--lambda_adain_var_loss", type=float, default=0)
parser.add_argument("--lambda_spade_var_loss", type=float, default=0)

parser.add_argument("--lambda_flop_loss", type=float, default=0.001)

parser.add_argument("--use_regul_loss", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument("--use_val_max_loss", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument("--use_flop_loss", type=str2bool, nargs='?',
					const=True, default=False,
					help="Activate nice mode.")

parser.add_argument("--use_residual", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument("--use_residual_ks_5", type=str, nargs='?',
						default=None, choices=[None, '33','5'],
						help="Activate nice mode.")

parser.add_argument("--use_delta_input", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument("--use_selector", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument("--use_gumbel_selector", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument("--gumbel_temp", type=float, default=1.0)

parser.add_argument("--front_norm_type", type=str, default="in", help="Define norm type", choices=["in", "bn", "nonorm"])

parser.add_argument("--use_owan_model", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument("--use_urie_model", type=str2bool, nargs='?',
						const=True, default=False,
						help="Activate nice mode.")
parser.add_argument('--eval_batch', action='store_true', help='use cuda?')
parser.add_argument('--ckt', type=str)

parser.add_argument("--use_bias", type=str2bool, nargs='?',
						const=True, default=True,
						help="Use bias=T")
parser.add_argument("--use_adain_output_for_spade", type=str2bool, nargs='?',
						const=True, default=False,
						help="Use adain output for spade input")
parser.add_argument("--use_zero_mean_spade", type=str2bool, nargs='?',
					const=True, default=True,
					help="Use zero mean spade")

parser.add_argument("--use_our_block", type=str2bool, nargs='?',
					const=True, default=False,
					help="Use zero mean spade")

opt=parser.parse_args()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'
model_name=opt.trainset+'_'+opt.net.split('.')[0]+'_'+str(opt.gps)+'_'+str(opt.blocks)+'_'+str(opt.desc)+'_'+str(randint(1,1000))
opt.model_dir=opt.model_dir+model_name+'.pk'
log_dir='logs/'+model_name

print(opt)
print('model_dir:',opt.model_dir)


if not os.path.exists('trained_models'):
	os.mkdir('trained_models')
if not os.path.exists('numpy_files'):
	os.mkdir('numpy_files')
if not os.path.exists('logs'):
	os.mkdir('logs')
if not os.path.exists('samples'):
	os.mkdir('samples')
if not os.path.exists(f"samples/{model_name}"):
	os.mkdir(f'samples/{model_name}')
if not os.path.exists(log_dir):
	os.mkdir(log_dir)
