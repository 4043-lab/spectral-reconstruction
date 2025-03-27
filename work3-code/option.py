import argparse
import template
import core.logger as Logger

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--template', default='my_model',
                    help='You can set various templates in option.py')
parser.add_argument('-c', '--config', type=str, default='config/cassi.json',
                        help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
parser.add_argument('-f', '--isFinetune', type=str, default='False', help='is finetune stage?')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/my_model/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='my_model', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi',
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask

# Training specifications
parser.add_argument('--batch_size', type=int, default=5, help='the number of HSIs per batch')
parser.add_argument('--patch_size', type=int, default=256, help='the patch size of HSIs')
parser.add_argument('--in_channels', type=int, default=28, help='the channel number of HSIs')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=5000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0004)
parser.add_argument("--train_dataset", type=str, default="CAVE512", help='choose a dataset for training,CAVE512 or CAVE1024')
parser.add_argument('--data_path_real', default='../../datasets/TSA_real_data/Measurements/', type=str,help='path of data')
parser.add_argument('--data_path_CAVE', default='../../datasets/cave_1024_28/', type=str,
                        help='path of data')
parser.add_argument('--data_path_KAIST', default='../../datasets/KAIST_CVPR2021/', type=str,
                    help='path of data')
parser.add_argument('--mask_path_real', default='../../datasets/TSA_real_data/', type=str,
                    help='path of real mask')

parser.add_argument('--pretrained_DAUHST_model_path', type=str, default="experiments/dauhst_3stg_icvl/model_epoch_230.pth", help='pretrained DAUHST model directory')

parser.add_argument('--pretrained_DDPM_model_path', type=str, default="experiments/cassi_shadow_dauhst3stg_fullChannel_icvl_sr3_250105_170146/checkpoint/I240000_E240_gen.pth", help='pretrained DDPM model directory')

opt = parser.parse_args()
template.set_template(opt)

diffusion_opt = Logger.parse(opt, isTrainCASSI=True)
# Convert to NoneDict, which return None for missing key.
diffusion_opt = Logger.dict_to_nonedict(diffusion_opt)

# dataset
if opt.train_dataset == "CAVE1024":
    opt.data_path = f"{opt.data_root}/cave_1024_28/"
    opt.test_path = f"{opt.data_root}/TSA_simu_data/Truth/"
elif opt.train_dataset == "CAVE512":
    opt.data_path = f"{opt.data_root}/CAVE_512_28/"
    opt.test_path = f"{opt.data_root}/TSA_simu_data/Truth/"
elif opt.train_dataset == "ICVL":
    opt.data_path = f"{opt.data_root}/ICVL/training_data/"
    opt.test_path = f"{opt.data_root}/ICVL/testing_data/"
opt.mask_path = f"{opt.data_root}/TSA_simu_data/"


for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False