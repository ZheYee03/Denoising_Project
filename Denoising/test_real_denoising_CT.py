import argparse
import os
import sys
from glob import glob
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Some Kaggle environments do not ship with lmdb, but this evaluation path
# only needs the network and metric code.
try:
    import lmdb  # noqa: F401
except ImportError:
    sys.modules['lmdb'] = ModuleType('lmdb')

import utils
from basicsr.models.archs.CLIPDenoising_arch import CLIPDenoising
from basicsr.metrics.CT_psnr_ssim import compute_PSNR, compute_SSIM

DEFAULT_INPUT_DIR = '/kaggle/input/datasets/leongxinying/ldct-dataset/LDCT_npy'
DEFAULT_CHECKPOINT_PATH = (
    '/kaggle/input/datasets/leongxinying/pretrained-syntheticdenoising/'
    'CLIPDenoising_SyntheticDenoising_GaussianSigma15/models/net_g_300000.pth'
)
DEFAULT_CLIP_MODEL_PATH = '/kaggle/working/RN50.pt'

parser = argparse.ArgumentParser(description='Standalone LDCT zero-shot evaluation')
parser.add_argument(
    '--input_dir',
    default=DEFAULT_INPUT_DIR,
    type=str,
    help='Directory containing flat LDCT .npy files.',
)
parser.add_argument(
    '--checkpoint_path',
    default=DEFAULT_CHECKPOINT_PATH,
    type=str,
    help='Path to the synthetic denoising checkpoint.',
)
parser.add_argument(
    '--clip_model_path',
    default=DEFAULT_CLIP_MODEL_PATH,
    type=str,
    help='Path to RN50.pt.',
)

args = parser.parse_args()


def load_checkpoint_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'params' in checkpoint:
            return checkpoint['params']
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
    return checkpoint


def proc(tar_img, prd_img):
    PSNR = utils.calculate_psnr(tar_img, prd_img)
    SSIM = utils.calculate_ssim(tar_img, prd_img)
    return PSNR, SSIM

# network arch
'''
type: CLIPDenoising
inp_channels: 3
out_channels: 3
depth: 5
wf: 64 
num_blocks: [3, 4, 6, 3] 
bias: false
model_path: /kaggle/working/RN50.pt

aug_level: 0.025
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build the native synthetic-denoising RGB model so the checkpoint can load
# exactly, then bridge CT slices by repeating 1-channel input to RGB and
# averaging the 3-channel prediction back to 1 channel for CT metrics.
model_restoration = CLIPDenoising(
    inp_channels=3,
    out_channels=3,
    depth=5,
    wf=64,
    num_blocks=[3, 4, 6, 3],
    bias=False,
    model_path=args.clip_model_path,
    aug_level=0.025,
)
state_dict = load_checkpoint_state_dict(args.checkpoint_path)
load_result = model_restoration.load_state_dict(state_dict, strict=True)
print(f'Using device: {device}')
print(f'Loaded CLIP encoder weights from: {args.clip_model_path}')
print(f'Loaded synthetic checkpoint from: {args.checkpoint_path}')
print(load_result)

model_restoration.to(device)
model_restoration.eval()
##########################

factor = 32

test_patient = 'L506'
target_path = sorted(glob(os.path.join(args.input_dir, '*target*')))
input_path = sorted(glob(os.path.join(args.input_dir, '*input*')))

input_ = [f for f in input_path if test_patient in f]
target_ = [f for f in target_path if test_patient in f]

if len(input_) != len(target_):
    raise ValueError(
        f'Mismatched number of L506 inputs and targets: '
        f'{len(input_)} inputs vs {len(target_)} targets.'
    )
if not input_:
    raise FileNotFoundError(
        f'No L506 input/target pairs found under: {args.input_dir}'
    )


trunc_min = -1024; trunc_max = 3072; shape_ = 512
psnr_list = []; ssim_list = []

for noise, clean in tqdm(zip(input_, target_)):
    
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        img_clean = np.load(clean)[..., np.newaxis]
        img_clean = torch.from_numpy(img_clean).permute(2,0,1)
        img_clean = img_clean.unsqueeze(0).float().to(device)

        img = np.load(noise)[..., np.newaxis]
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).float().to(device)

        # Repeat grayscale CT input to 3 channels to match the RGB synthetic
        # checkpoint architecture.
        input_rgb = input_.repeat(1, 3, 1, 1)

        # Padding in case images are not multiples of 32
        h,w = input_rgb.shape[2], input_rgb.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_rgb = F.pad(input_rgb, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_rgb)

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]
        restored = restored.mean(dim=1, keepdim=True)
        
        psnr, ssim = compute_PSNR(restored, img_clean, data_range=trunc_max-trunc_min, trunc_min=trunc_min, trunc_max=trunc_max,
                                 norm_range_max=3096, norm_range_min=-1024), \
                     compute_SSIM(restored, img_clean, data_range=trunc_max-trunc_min, trunc_min=trunc_min, trunc_max=trunc_max,
                                 norm_range_max=3096, norm_range_min=-1024)

        psnr_list.append(psnr); ssim_list.append(ssim)
        

print('CT dataset: psnr:{:.2f}, ssim:{:.3f}'.format(
                        sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list)))
