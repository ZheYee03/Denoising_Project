import argparse
import os
import sys
from glob import glob
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Some Kaggle environments do not ship with lmdb, but the test path only needs
# the network definition, not LMDB utilities.
try:
    import lmdb  # noqa: F401
except ImportError:
    sys.modules['lmdb'] = ModuleType('lmdb')

import utils
from basicsr.models.archs.CLIPDenoising_arch import CLIPDenoising
from basicsr.models.archs.CLIPEncoder_util import ModifiedResNet

DEFAULT_INPUT_DIR = '/kaggle/input/datasets/leongxinying/cbsd432-dataset'
DATASET_PATHS = {
    'CBSD68': '/kaggle/input/datasets/leongxinying/cbsd432-dataset/cbsd68/cbsd68',
    'McM': '/kaggle/input/datasets/leongxinying/cbsd432-dataset/McM/McM',
    'Kodak24': '/kaggle/input/datasets/leongxinying/cbsd432-dataset/kodak24/kodak24',
    'Urban100': '/kaggle/input/datasets/leongxinying/cbsd432-dataset/urban100/urban100',
}
DEFAULT_CHECKPOINT_PATH = (
    '/kaggle/input/datasets/leongxinying/pretrained-syntheticdenoising/'
    'CLIPDenoising_SyntheticDenoising_GaussianSigma15/models/net_g_300000.pth'
)
DEFAULT_CLIP_MODEL_PATH = '/kaggle/working/RN50.pt'

parser = argparse.ArgumentParser(description='Synthetic Color Denoising')
parser.add_argument(
    '--input_dir',
    default=DEFAULT_INPUT_DIR,
    type=str,
    help='Fallback directory containing validation datasets when a dataset is not in DATASET_PATHS.',
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


def resolve_clip_model_path(cli_path):
    candidate_paths = [Path(cli_path)]

    for candidate in candidate_paths:
        if candidate and candidate.is_file():
            print(f'Using CLIP encoder weights from: {candidate}')
            return str(candidate)

    print(
        'CLIP RN50 weights were not found at '
        f'{cli_path}. Proceeding without loading a local RN50.pt file.'
    )
    return None


_original_load_pretrain_model = ModifiedResNet.load_pretrain_model


def _safe_load_pretrain_model(self, model_path):
    if not model_path:
        return
    if not Path(model_path).is_file():
        print(
            f'CLIP model path not found: {model_path}. '
            'Proceeding without encoder pretrain loading.'
        )
        return
    _original_load_pretrain_model(self, model_path)


ModifiedResNet.load_pretrain_model = _safe_load_pretrain_model


def resolve_dataset_dir(input_dir, dataset_name):
    mapped_dir = DATASET_PATHS.get(dataset_name)
    if mapped_dir is not None:
        mapped_path = Path(mapped_dir)
        if mapped_path.is_dir():
            return mapped_path, collect_image_files(mapped_path)
        return mapped_path, []

    input_path = Path(input_dir)
    candidate_dirs = [
        input_path / dataset_name,
        input_path,
    ]

    for candidate in candidate_dirs:
        if candidate.is_dir():
            files = collect_image_files(candidate)
            if files:
                return candidate, files

    for candidate in candidate_dirs:
        if candidate.is_dir():
            files = collect_image_files(candidate)
            return candidate, files

    return candidate_dirs[0], []


def collect_image_files(folder):
    patterns = ['*.png', '*.tif', '*.jpg', '*.jpeg']
    files = []
    for pattern in patterns:
        files.extend(sorted(glob(str(folder / pattern))))
    return files


def load_checkpoint_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'params' in checkpoint:
            return checkpoint['params']
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
    return checkpoint


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    clip_model_path = resolve_clip_model_path(args.clip_model_path)

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
        model_path=clip_model_path,
        aug_level=0.025,
    )

    checkpoint_path = Path(DEFAULT_CHECKPOINT_PATH)
    if not checkpoint_path.is_file():
        print(f'Checkpoint not found: {checkpoint_path}')
        return

    state_dict = load_checkpoint_state_dict(str(checkpoint_path))
    load_result = model_restoration.load_state_dict(state_dict, strict=True)
    print(f'Checkpoint loaded from: {checkpoint_path}')
    print(load_result)

    model_restoration.to(device)
    model_restoration.eval()

    factor = 32

    datasets = ['CBSD68', 'McM', 'Kodak24', 'Urban100']
    noise_types = ['gauss', 'spatial_gauss', 'poisson']

    for dataset in datasets:
        inp_dir, files = resolve_dataset_dir(args.input_dir, dataset)
        print(f'Dataset name: {dataset}')
        print(f'Dataset path: {inp_dir}')
        print(f'Number of images found: {len(files)}')
        print('First 5 file paths found:')
        for file_path in files[:5]:
            print(file_path)

        if not inp_dir.is_dir():
            print(f'Dataset folder does not exist for {dataset}: {inp_dir}. Skipping.')
            continue

        if not files:
            print(f'No images found for dataset {dataset}. Skipping.')
            continue

        for noise_type in noise_types:
            if noise_type == 'gauss':
                sigmas = [15, 25, 50]
            elif noise_type == 'poisson':
                sigmas = [2, 2.5, 3, 3.5]
            elif noise_type == 'spatial_gauss':
                sigmas = [40, 45, 50, 55]
            else:
                continue

            for sigma_test in sigmas:
                psnr_list = []
                ssim_list = []

                with torch.no_grad():
                    for file_ in tqdm(files, desc=f'{dataset}-{noise_type}-{sigma_test}'):
                        try:
                            img_clean = np.float32(utils.load_img(file_)) / 255.0
                        except Exception as exc:
                            print(f'Unable to read image, skipping: {file_}')
                            print(f'Read error: {exc}')
                            continue

                        np.random.seed(seed=0)  # for reproducibility

                        # gaussian noise
                        if noise_type == 'gauss':
                            img = img_clean + np.random.normal(
                                0, sigma_test / 255.0, img_clean.shape
                            )

                        # poisson noise
                        elif noise_type == 'poisson':
                            img = utils.add_poisson_noise(img_clean, scale=sigma_test)

                        elif noise_type == 'spatial_gauss':
                            noise = np.random.normal(0, sigma_test / 255.0, img_clean.shape)
                            kernel = np.ones((3, 3)) / 9.0
                            for chn in range(img_clean.shape[-1]):
                                noise[..., chn] = convolve(noise[..., chn], kernel)

                            img = img_clean + noise

                        img = torch.from_numpy(img).permute(2, 0, 1).float()
                        input_ = img.unsqueeze(0).to(device)

                        h, w = input_.shape[2], input_.shape[3]
                        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                        padh = H - h if h % factor != 0 else 0
                        padw = W - w if w % factor != 0 else 0
                        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

                        restored = model_restoration(input_)
                        restored = restored[:, :, :h, :w]

                        restored = (
                            torch.clamp(restored, 0, 1)
                            .cpu()
                            .detach()
                            .permute(0, 2, 3, 1)
                            .squeeze(0)
                            .numpy()
                        )
                        psnr, ssim = proc(img_clean * 255.0, restored * 255.0)

                        psnr_list.append(psnr)
                        ssim_list.append(ssim)

                if not psnr_list:
                    print(
                        f'noise_type:{noise_type}, dataset:{dataset}, sigma:{sigma_test}, '
                        'no valid images were processed.'
                    )
                    continue

                print(
                    'noise_type:{}, dataset:{}, sigma:{}, psnr:{:.2f}, ssim:{:.3f}'.format(
                        noise_type,
                        dataset,
                        sigma_test,
                        sum(psnr_list) / len(psnr_list),
                        sum(ssim_list) / len(ssim_list),
                    )
                )


if __name__ == '__main__':
    main()
