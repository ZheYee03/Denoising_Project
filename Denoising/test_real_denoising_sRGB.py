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
# only needs the network definition and metric helpers.
try:
    import lmdb  # noqa: F401
except ImportError:
    sys.modules['lmdb'] = ModuleType('lmdb')

import utils
from basicsr.models.archs.CLIPDenoising_arch import CLIPDenoising

DEFAULT_INPUT_DIR = '/kaggle/input/datasets/leongxinying/srgb-dataset/sRGB-clipdenoise'
DEFAULT_CHECKPOINT_PATH = (
    '/kaggle/input/clipdenoising-pretrained-srgb/'
    'CLIPDenoising_sRGBDenoising_FixedPoissonGaussian/models/net_g_latest.pth'
)
DEFAULT_CLIP_MODEL_PATH = '/kaggle/working/RN50.pt'

parser = argparse.ArgumentParser(description='Real sRGB Denoising')
parser.add_argument(
    '--input_dir',
    default=DEFAULT_INPUT_DIR,
    type=str,
    help='Directory containing CC, PolyU, and SIDD/SIDD_test_PNG.',
)
parser.add_argument(
    '--checkpoint_path',
    default=DEFAULT_CHECKPOINT_PATH,
    type=str,
    help='Path to the sRGB-trained checkpoint.',
)
parser.add_argument(
    '--clip_model_path',
    default=DEFAULT_CLIP_MODEL_PATH,
    type=str,
    help='Path to RN50.pt.',
)

args = parser.parse_args()


def proc(tar_img, prd_img):
    psnr = utils.calculate_psnr(tar_img, prd_img)
    ssim = utils.calculate_ssim(tar_img, prd_img)
    return psnr, ssim


def main():
    input_dir = Path(args.input_dir)
    checkpoint_path = Path(args.checkpoint_path)
    clip_model_path = Path(args.clip_model_path)

    if not input_dir.is_dir():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    if not clip_model_path.is_file():
        raise FileNotFoundError(f'RN50.pt not found: {clip_model_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Input directory: {input_dir}')
    print(f'Checkpoint path: {checkpoint_path}')
    print(f'CLIP model path: {clip_model_path}')

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

    model_restoration = CLIPDenoising(
        inp_channels=3,
        out_channels=3,
        depth=5,
        wf=64,
        num_blocks=[3, 4, 6, 3],
        bias=False,
        model_path=str(clip_model_path),
        aug_level=0.025,
    )
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    load_result = model_restoration.load_state_dict(checkpoint['params'])
    print(load_result)

    model_restoration.to(device)
    model_restoration.eval()

    factor = 32
    testsets = ['CC', 'PolyU', 'SIDD_val']

    for testset in testsets:
        if testset == 'CC':
            noises = sorted(glob(str(input_dir / testset / '*real.png')))
            cleans = sorted(glob(str(input_dir / testset / '*mean.png')))

        elif testset == 'PolyU':
            noises = sorted(glob(str(input_dir / testset / '*real.JPG')))
            cleans = sorted(glob(str(input_dir / testset / '*mean.JPG')))

        elif testset == 'SIDD_val':
            cleans = sorted(glob(str(input_dir / 'SIDD' / 'SIDD_test_PNG' / 'GT' / '*.png')))
            noises = sorted(glob(str(input_dir / 'SIDD' / 'SIDD_test_PNG' / 'noisy' / '*.png')))

        psnr_list = []
        ssim_list = []
        for noise, clean in tqdm(zip(noises, cleans), total=len(noises), desc=testset):
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()

                img_clean = utils.load_img(clean)
                img = np.float32(utils.load_img(noise)) / 255.0

                img = torch.from_numpy(img).permute(2, 0, 1)
                input_ = img.unsqueeze(0).to(device)

                # Padding in case images are not multiples of 32
                h, w = input_.shape[2], input_.shape[3]
                H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                padh = H - h if h % factor != 0 else 0
                padw = W - w if w % factor != 0 else 0
                input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

                restored = model_restoration(input_)

                # Unpad images to original dimensions
                restored = restored[:, :, :h, :w]

                restored = (
                    torch.clamp(restored, 0, 1)
                    .cpu()
                    .detach()
                    .permute(0, 2, 3, 1)
                    .squeeze(0)
                    .numpy()
                )
                restored = (restored * 255.0).round().astype(np.uint8)

                psnr, ssim = proc(img_clean, restored)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

        print(
            'dataset:{}, psnr:{:.2f}, ssim:{:.3f}'.format(
                testset,
                sum(psnr_list) / len(psnr_list),
                sum(ssim_list) / len(ssim_list),
            )
        )


if __name__ == '__main__':
    main()
