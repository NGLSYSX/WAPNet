import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset

from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)

    def forward(self, x):
        return self.net(x)


def tta_restore(net_promptir, lq):
    """
    Geometry-based test-time augmentation.
    Four inputs are used: original image, horizontal flip, vertical flip, and 90-degree rotation.
    """
    with torch.no_grad():
        y0 = net_promptir(lq)
        y1 = torch.flip(net_promptir(torch.flip(lq, dims=[3])), dims=[3])
        y2 = torch.flip(net_promptir(torch.flip(lq, dims=[2])), dims=[2])
        y3 = torch.rot90(net_promptir(torch.rot90(lq, 1, [2, 3])), -1, [2, 3])

        out = (y0 + y1 + y2 + y3) / 4.0

    return out


class SimpleTestDataset(Dataset):
    def __init__(self, lq_dir, gt_dir):
        super().__init__()

        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        self.to_tensor = ToTensor()

        if not os.path.isdir(lq_dir):
            raise FileNotFoundError(f"LQ directory not found: {lq_dir}")

        self.file_names = sorted(os.listdir(lq_dir))
        self.file_names = [
            x for x in self.file_names
            if x.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]

        if len(self.file_names) == 0:
            raise RuntimeError(f"No valid image files found in: {lq_dir}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        lq_path = os.path.join(self.lq_dir, file_name)
        lq_img = Image.open(lq_path).convert("RGB")

        gt_path = os.path.join(self.gt_dir, file_name) if self.gt_dir else ""
        has_gt = os.path.exists(gt_path)

        if has_gt:
            gt_img = Image.open(gt_path).convert("RGB")
        else:
            gt_img = lq_img.copy()

        # Make image size divisible by 8.
        w, h = lq_img.size
        new_w = w - (w % 8)
        new_h = h - (h % 8)

        if new_w < 8 or new_h < 8:
            new_w = w
            new_h = h

        if new_w != w or new_h != h:
            lq_img = lq_img.crop((0, 0, new_w, new_h))
            gt_img = gt_img.crop((0, 0, new_w, new_h))

        lq_tensor = self.to_tensor(lq_img)
        gt_tensor = self.to_tensor(gt_img)

        return file_name, lq_tensor, gt_tensor, has_gt


def _to_file_name_list(file_name):
    if isinstance(file_name, (list, tuple)):
        return list(file_name)
    return [file_name]


def test_dataset(
    net,
    dataset_name,
    lq_path,
    gt_path,
    output_root,
    use_tta=False,
    num_workers=4,
    batch_size=1,
    save_images=True,
    pin_memory=True,
    persistent_workers=False,
    amp=False,
    channels_last=False,
    empty_cache_every=0,
):
    print(f"\nTesting on {dataset_name}...")
    print(f"LQ Path: {lq_path}")
    print(f"GT Path: {gt_path}")

    dataset = SimpleTestDataset(lq_path, gt_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )

    save_dir = os.path.join(output_root, dataset_name)
    if save_images:
        os.makedirs(save_dir, exist_ok=True)

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    if channels_last:
        net = net.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        for batch_idx, (file_name, lq, gt, has_gt) in enumerate(tqdm(dataloader)):
            lq = lq.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)

            if channels_last:
                lq = lq.contiguous(memory_format=torch.channels_last)

            if amp and torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if use_tta:
                        restored = tta_restore(net.net, lq)
                    else:
                        restored = net(lq)
            else:
                if use_tta:
                    restored = tta_restore(net.net, lq)
                else:
                    restored = net(lq)

            file_names = _to_file_name_list(file_name)

            # Save restored images.
            if save_images:
                restored_to_save = restored
                if restored_to_save.dim() == 3:
                    restored_to_save = restored_to_save.unsqueeze(0)

                for i, fn in enumerate(file_names):
                    save_path = os.path.join(save_dir, fn)
                    save_image_tensor(restored_to_save[i], save_path)

            # Compute PSNR/SSIM for images with GT.
            if isinstance(has_gt, torch.Tensor):
                has_gt_tensor = has_gt.bool()
            else:
                has_gt_tensor = torch.tensor(has_gt, dtype=torch.bool)

            valid_indices = torch.nonzero(has_gt_tensor, as_tuple=False).view(-1)

            if valid_indices.numel() > 0:
                valid_indices_cuda = valid_indices.to(restored.device)
                restored_valid = restored.index_select(0, valid_indices_cuda)
                gt_valid = gt.index_select(0, valid_indices_cuda)

                temp_psnr, temp_ssim, n = compute_psnr_ssim(restored_valid, gt_valid)
                psnr_meter.update(temp_psnr, n)
                ssim_meter.update(temp_ssim, n)

            if valid_indices.numel() < len(file_names):
                missing_indices = set(range(len(file_names))) - set(valid_indices.cpu().tolist())
                for i in missing_indices:
                    print(f"  Warning: No GT found for {file_names[i]}, skipping PSNR/SSIM calculation.")

            del lq, gt, restored

            if empty_cache_every and ((batch_idx + 1) % int(empty_cache_every) == 0):
                torch.cuda.empty_cache()

    if psnr_meter.count > 0:
        print(f"Result on {dataset_name}:")
        print(f"PSNR: {psnr_meter.avg:.4f}")
        print(f"SSIM: {ssim_meter.avg:.4f}")
        return psnr_meter.avg, ssim_meter.avg

    print(f"Result on {dataset_name}: No GT available for PSNR/SSIM calculation.")
    return None, None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="results")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--empty_cache_every", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--raindrop_lq", type=str, default="")
    parser.add_argument("--raindrop_gt", type=str, default="")
    parser.add_argument("--test1_lq", type=str, default="")
    parser.add_argument("--test1_gt", type=str, default="")
    parser.add_argument("--snow100k_s_lq", type=str, default="")
    parser.add_argument("--snow100k_s_gt", type=str, default="")
    parser.add_argument("--snow100k_l_lq", type=str, default="")
    parser.add_argument("--snow100k_l_gt", type=str, default="")
    parser.add_argument("--outdoor_rain_lq", type=str, default="")
    parser.add_argument("--outdoor_rain_gt", type=str, default="")

    args = parser.parse_args()

    experiment_name = args.experiment_name.strip()
    output_root = os.path.join(args.output_root, experiment_name) if experiment_name else args.output_root

    datasets = {}

    if args.raindrop_lq and args.raindrop_gt:
        datasets["RainDrop"] = (args.raindrop_lq, args.raindrop_gt)

    if args.test1_lq and args.test1_gt:
        datasets["Test1"] = (args.test1_lq, args.test1_gt)

    if args.snow100k_s_lq and args.snow100k_s_gt:
        datasets["Snow100K-S"] = (args.snow100k_s_lq, args.snow100k_s_gt)

    if args.snow100k_l_lq and args.snow100k_l_gt:
        datasets["Snow100K-L"] = (args.snow100k_l_lq, args.snow100k_l_gt)

    if args.outdoor_rain_lq and args.outdoor_rain_gt:
        datasets["Outdoor-Rain"] = (args.outdoor_rain_lq, args.outdoor_rain_gt)

    if len(datasets) == 0:
        raise RuntimeError("No valid test dataset paths are provided.")

    print(f"Loading model from {args.ckpt_path}")
    model = PromptIRModel.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.cuda()

    if args.benchmark:
        torch.backends.cudnn.benchmark = True

    results = {}

    for name, (lq, gt) in datasets.items():
        if not os.path.exists(lq):
            print(f"Skipping {name}: LQ path not found: {lq}")
            continue

        psnr, ssim = test_dataset(
            model,
            name,
            lq,
            gt,
            output_root,
            use_tta=args.use_tta,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            save_images=(not args.no_save),
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            amp=args.amp,
            channels_last=args.channels_last,
            empty_cache_every=args.empty_cache_every,
        )

        if psnr is not None and ssim is not None:
            results[name] = {"PSNR": psnr, "SSIM": ssim}
        else:
            results[name] = {"PSNR": "N/A", "SSIM": "N/A"}

    print("\n" + "=" * 20 + " Final Results " + "=" * 20)

    psnr_vals = []
    ssim_vals = []

    for name, res in results.items():
        if isinstance(res["PSNR"], str):
            print(f"{name:<15} | PSNR: {res['PSNR']:>6} | SSIM: {res['SSIM']:>6}")
        else:
            print(f"{name:<15} | PSNR: {res['PSNR']:.4f} | SSIM: {res['SSIM']:.4f}")
            psnr_vals.append(res["PSNR"])
            ssim_vals.append(res["SSIM"])

    if psnr_vals:
        print("-" * 55)
        print(f"{'Average':<15} | PSNR: {float(np.mean(psnr_vals)):.4f} | SSIM: {float(np.mean(ssim_vals)):.4f}")

    print("=" * 55)


if __name__ == "__main__":
    main()
