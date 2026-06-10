import argparse
import subprocess
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn as nn 
from PIL import Image
from torchvision.transforms import ToTensor

# 导入必要的模块
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR
import torch.nn.functional as F
import lightning.pytorch as pl
import torch.optim as optim

# 定义模型
class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
    
    def forward(self, x):
        return self.net(x)

##########################################################################
## TTA 工具函数
def dark_channel(x, patch_size=15):
    min_rgb = x.min(dim=1, keepdim=True)[0]
    dark = -F.max_pool2d(-min_rgb,
                          kernel_size=patch_size,
                          stride=1,
                          padding=patch_size // 2)
    return dark


def tv_loss(x):
    """Total Variation loss，抑制高频噪声"""
    diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w


def tta_restore(net_promptir, lq):
    """
    Ensemble TTA：4种增强取平均，无需优化，稳定有效。
    """
    with torch.no_grad():
        #原图
        y0 = net_promptir(lq)
        #平翻转
        y1 = torch.flip(net_promptir(torch.flip(lq, dims=[3])), dims=[3])
        #直翻转
        y2 = torch.flip(net_promptir(torch.flip(lq, dims=[2])), dims=[2])
        # 90度旋转
        y3 = torch.rot90(net_promptir(torch.rot90(lq, 1, [2, 3])), -1, [2, 3])

        out = (y0 + y1 + y2 + y3) / 4.0

    return out

# 定义简单的测试数据集类
class SimpleTestDataset(Dataset):
    def __init__(self, lq_dir, gt_dir):
        super().__init__()
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir

        # 获取所有图片文件名
        self.file_names = sorted(os.listdir(lq_dir))
        # 过滤非图片文件
        self.file_names = [x for x in self.file_names if x.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.toTensor = ToTensor()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        lq_path = os.path.join(self.lq_dir, file_name)
        
        # 读取图片
        lq_img = Image.open(lq_path).convert('RGB')
        
        # 检查是否存在对应的GT文件
        gt_path = os.path.join(self.gt_dir, file_name)
        if os.path.exists(gt_path):
            gt_img = Image.open(gt_path).convert('RGB')
        else:
            # 如果GT文件不存在，创建一个占位符（与LQ相同大小）
            gt_img = lq_img.copy()

        # 确保尺寸是8的倍数（PromptIR通常需要）
        # 确保尺寸是8的倍数（PromptIR通常需要）
        # 同时确保最小尺寸 > 7（SSIM window size requirement）
        w, h = lq_img.size
        new_w = w - (w % 8)
        new_h = h - (h % 8)

        # 如果裁剪后尺寸过小，放弃裁剪（虽然可能导致模型报错，但至少不会让 SSIM 报错）
        if new_w < 8 or new_h < 8:
            new_w = w
            new_h = h

        if new_w != w or new_h != h:
            lq_img = lq_img.crop((0, 0, new_w, new_h))
            gt_img = gt_img.crop((0, 0, new_w, new_h))

        lq_tensor = self.toTensor(lq_img)
        gt_tensor = self.toTensor(gt_img)

        return file_name, lq_tensor, gt_tensor

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

    # 创建数据集和加载器
    dataset = SimpleTestDataset(lq_path, gt_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )

    if save_images:
        save_dir = os.path.join(output_root, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

    # 统计指标
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    if channels_last:
        net = net.to(memory_format=torch.channels_last)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16) if (amp and torch.cuda.is_available()) else None
    )

    with torch.inference_mode():
        for batch_idx, (file_name, lq, gt) in enumerate(tqdm(dataloader)):
            lq = lq.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            if channels_last:
                lq = lq.contiguous(memory_format=torch.channels_last)

            if autocast_ctx is not None:
                with autocast_ctx:
                    if use_tta:
                        restored = tta_restore(net.net, lq)
                    else:
                        restored = net(lq)
            else:
                if use_tta:
                    restored = tta_restore(net.net, lq)
                else:
                    restored = net(lq)

        # 检查GT是否存在（如果GT是占位符则跳过指标计算）
        gt_exists = not torch.equal(gt, lq)  # 如果GT和LQ不同，则认为GT存在
        if gt_exists:
            # 计算指标
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, gt)
            psnr_meter.update(temp_psnr, N)
            ssim_meter.update(temp_ssim, N)
        else:
            print(f"  Warning: No GT found for {file_name[0]}, skipping PSNR/SSIM calculation")

            if save_images:
                if isinstance(file_name, (list, tuple)):
                    file_names = list(file_name)
                else:
                    file_names = [file_name]

                if restored.dim() == 3:
                    restored_to_save = restored.unsqueeze(0)
                else:
                    restored_to_save = restored

                for i, fn in enumerate(file_names):
                    save_image_tensor(restored_to_save[i], os.path.join(save_dir, fn))

            del lq, gt, restored
            if 'temp_psnr' in locals():
                del temp_psnr, temp_ssim
            if empty_cache_every and ((batch_idx + 1) % int(empty_cache_every) == 0):
                torch.cuda.empty_cache()

    if psnr_meter.count > 0:
        print(f"Result on {dataset_name}:")
        print(f"PSNR: {psnr_meter.avg:.4f}")
        print(f"SSIM: {ssim_meter.avg:.4f}")
        return psnr_meter.avg, ssim_meter.avg
    else:
        print(f"Result on {dataset_name}: No GT available for PSNR/SSIM calculation")
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

    print(f"Loading model from {args.ckpt_path}")
    model = PromptIRModel.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.cuda()
    if args.benchmark:
        torch.backends.cudnn.benchmark = True

    # 遍历测试
    results = {}
    for name, (lq, gt) in datasets.items():
        if not os.path.exists(lq):
            print(f"Skipping {name}: LQ Path not found!")
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
            print(f"Dataset {name}: PSNR/SSIM not calculated (no GT)")

    # 打印汇总
    print("\n" + "="*20 + " Final Results " + "="*20)
    psnr_vals = []
    ssim_vals = []
    for name, res in results.items():
        if isinstance(res['PSNR'], str):
            print(f"{name:<15} | PSNR: {res['PSNR']:>6} | SSIM: {res['SSIM']:>6}")
        else:
            print(f"{name:<15} | PSNR: {res['PSNR']:.4f} | SSIM: {res['SSIM']:.4f}")
            psnr_vals.append(res["PSNR"])
            ssim_vals.append(res["SSIM"])
    if psnr_vals:
        print("-"*55)
        print(f"{'Average':<15} | PSNR: {float(np.mean(psnr_vals)):.4f} | SSIM: {float(np.mean(ssim_vals)):.4f}")
    print("="*55)

if __name__ == '__main__':
    main()
