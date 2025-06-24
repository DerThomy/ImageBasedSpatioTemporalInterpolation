
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.io as tio
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

# from lpips import LPIPS
from loss_utils import ssim
from flip_loss import LDRFLIPLoss

device = "cuda:0"

C0 = torch.Tensor((-0.002136485053939582, -0.000749655052795221, -0.005386127855323933)).to(device)
C1 = torch.Tensor((0.2516605407371642, 0.6775232436837668, 2.494026599312351)).to(device)
C2 = torch.Tensor((8.353717279216625, -3.577719514958484, 0.3144679030132573)).to(device)
C3 = torch.Tensor((-27.66873308576866, 14.26473078096533, -13.64921318813922)).to(device)
C4 = torch.Tensor((52.17613981234068, -27.94360607168351, 12.94416944238394)).to(device)
C5 = torch.Tensor((-50.76852536473588, 29.04658282127291, 4.23415299384598)).to(device)
C6 = torch.Tensor((18.65570506591883, -11.48977351997711, -5.601961508734096)).to(device)

def colormap_magma(x):
	x = torch.clip(x, 0, 1)
	res = (C0+x*(C1+x*(C2+x*(C3+x*(C4+x*(C5+C6*x))))))
	return torch.clip(res, 0, 1)

class ImagePairDataset(Dataset):
    def __init__(self, gt_dir, render_dir, fnames, transform=None, device='cpu'):
        """
        Args:
            file_list (list of str): List of image file paths.
            transform (callable, optional): Optional transform to be applied on an image.
            device (str, optional): Device to load images onto ('cpu' or 'cuda').
        """
        self.gt_dir = gt_dir
        self.render_dir = render_dir
        self.file_list = fnames
        self.transform = transform
        self.device = torch.device(device)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        def load_img(dir):
            image = tio.read_image(os.path.join(dir, fname), tio.ImageReadMode.RGB).float() / 255.0
            if self.transform:
                image = self.transform(image)
            # image = F.interpolate(image.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False, antialias=True).squeeze(0)
            return image
        
        return (load_img(self.gt_dir), load_img(self.render_dir), fname) #if fname == "0702.png" else (torch.Tensor(), torch.Tensor(), fname)
    
# Example usage:
if __name__ == "__main__":
    magma_cm = plt.get_cmap('magma')
    psnr_fn = lambda x, y: (-10.0 * torch.log(F.mse_loss(x, y)) / np.log(10.0))
    ssim_fn = lambda x, y: ssim(x, y).mean()

    # lpips_net = LPIPS(net="vgg").to(device)
    # lpips_fn = lambda x, y: lpips_net(x, y, normalize=True).mean()
    flip = LDRFLIPLoss()
    flip_fn = lambda x, y: flip(x, y).mean()

    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth_dir", type=str, help="GT images directory")
    parser.add_argument("render_dir", type=str, help="render images directory")
    parser.add_argument("-ff", "--frames_from", type=int, default=0, help="frame index from")
    parser.add_argument("-ft", "--frames_to", type=int, default=-1, help="frame index to")
    args = parser.parse_args()

    if not os.path.exists(args.ground_truth_dir) or not os.path.exists(args.render_dir):
        print(f"ERROR - Directory does not exist! GT={args.ground_truth_dir}, RENDER={args.render_dir}")
        exit(0)

    frames = sorted([f for f in os.listdir(args.ground_truth_dir) if os.path.splitext(f)[1]])
    frames = frames[args.frames_from:(len(frames) + args.frames_to if args.frames_to < 0 else args.frames_to)]

    dataset = ImagePairDataset(args.ground_truth_dir, args.render_dir, frames, transform=None, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    metrics_fns = {"PSNR": psnr_fn, "SSIM": ssim_fn, "FLIP": flip_fn}#, "LPIPS": lpips_fn}

    vmaf_frames = None
    if os.path.exists(os.path.join(args.render_dir, "vmaf.json")):
        vmaf_json = json.load(open(os.path.join(args.render_dir, "vmaf.json")))
        vmaf_frames = [v["metrics"]["vmaf"] for v in vmaf_json["frames"][args.frames_from:(args.frames_from + len(frames))]]
        metrics_fns["VMAF"] = lambda x, y: torch.Tensor([vmaf_frames.pop(0)]) if vmaf_frames is not None and len(vmaf_frames) > 0 else torch.tensor([float('nan')])

    per_image_metrics = {}
    per_metric_lists =  {k: [] for k in metrics_fns.keys()}

    for (gt_img, render_img, fname) in tqdm(dataloader, "Evaluating Images"):
        if (gt_img.shape[1] < 1 and render_img.shape[1] < 1):
            continue

        gt_img = gt_img.to(device)
        render_img = render_img.to(device)
        fname = fname[0]

        if False:
            diff_flip_img = flip(render_img, gt_img)
            diff_magma_flip_img = colormap_magma(diff_flip_img[..., None]).squeeze() * 255
            # print(diff_flip_img.mean())
            os.makedirs(os.path.join(args.render_dir, "flip"), exist_ok=True)
            cv2.imwrite(os.path.join(args.render_dir, "flip", fname), cv2.cvtColor(diff_magma_flip_img.cpu().numpy(), cv2.COLOR_RGB2BGR))
        
        per_image_metrics[fname] = {}
        for metric_key, metric_fn in metrics_fns.items():
            val = metric_fn(render_img, gt_img).item()
            per_image_metrics[fname][metric_key] = val
            per_metric_lists[metric_key].append(val)
        
    avg_metrics = {key : np.mean(np.array(vals)).item() for key, vals in per_metric_lists.items()}
    print(avg_metrics)
    
    with open(os.path.join(args.render_dir, "results.json"), 'w') as fp:
        json.dump(avg_metrics, fp, indent=True)

    with open(os.path.join(args.render_dir, "per_image_results.json"), 'w') as fp:
        json.dump(per_image_metrics, fp, indent=True)