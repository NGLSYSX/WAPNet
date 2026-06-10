# -*- coding: utf-8 -*-
"""
用法：
  # 完整流程（推理 + IQA + Params/GFlops）
  python measure_metrics.py \
      --ckpt /home/liu/LIUYUHUI/PromptIR-main2/train_ckpt/your.ckpt \
      --lq_dir /path/to/snow-real/lq \
      --save_dir ./results_snow_real \
      --use_tta

  #已有推理结果，只算 IQA
  python measure_metrics.py \
      --ckpt /path/to/your.ckpt --skip_infer --save_dir ./results_snow_real

  #只算 Params/GFlops
  python measure_metrics.py --ckpt /path/to/your.ckpt --skip_iqa

  #调试（只跑 10张 ）
  python measure_metrics.py \
      --ckpt /path/to/your.ckpt --skip_infer \
      --save_dir ./results_snow_real --limit 10
"""
import os, sys, time, glob, argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

try:
    from net.model import PromptIR
except ImportError:
    print("[Error] 无法导入 net.model.PromptIR，请在项目根目录运行。"); sys.exit(1)

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

THOP_AVAILABLE = False
try:
    import thop; THOP_AVAILABLE = True
except ImportError:
    print("[Info] thop 未安装（pip install thop）")

PTFLOPS_AVAILABLE = False
try:
    from ptflops import get_model_complexity_info; PTFLOPS_AVAILABLE = True
except ImportError:
    print("[Info] ptflops 未安装（pip install ptflops）")

PYIQA_AVAILABLE = False
try:
    import pyiqa; PYIQA_AVAILABLE = True
except ImportError:
    print("[Info] pyiqa 未安装（pip install pyiqa）")

NIQE_BACKEND = None
try:
    from basicsr.metrics.niqe import calculate_niqe as _basicsr_niqe
    NIQE_BACKEND = "basicsr"; print("[Info] NIQE 使用 basicsr")
except ImportError:
    print("[Info] NIQE 不可用 (未安装 basicsr)")


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
    def forward(self, x, tta_z=None):
        return self.net(x, tta_z=tta_z)


def tta_restore(net, lq):
    with torch.no_grad():
        y0 = net(lq)
        y1 = torch.flip(net(torch.flip(lq, dims=[3])), dims=[3])
        y2 = torch.flip(net(torch.flip(lq, dims=[2])), dims=[2])
        y3 = torch.rot90(net(torch.rot90(lq, 1, [2,3])), -1, [2,3])
        return (y0+y1+y2+y3)/4.0


class LQOnlyDataset(Dataset):
    def __init__(self, lq_dir):
        exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
        self.files = sorted([f for f in os.listdir(lq_dir) if f.lower().endswith(exts)])
        self.lq_dir = lq_dir
        self.toT = ToTensor()
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.lq_dir, name)).convert("RGB")
        w, h = img.size
        nw, nh = max(8,w-(w%8)), max(8,h-(h%8))
        if nw!=w or nh!=h: img = img.crop((0,0,nw,nh))
        return name, self.toT(img)


def run_inference(model, lq_dir, save_dir, use_tta=False):
    os.makedirs(save_dir, exist_ok=True)
    ds = LQOnlyDataset(lq_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    print(f"\n[推理]共 {len(ds)}张，use_tta={use_tta}")
    for (name,), lq in tqdm(loader, desc="推理"):
        lq = lq.cuda()
        out = tta_restore(model.net, lq) if use_tta else model(lq)
        arr = (out.squeeze(0).cpu().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, name), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        del lq, out; torch.cuda.empty_cache()
    print(f"[推理] 结果已保存：{save_dir}")


def measure_complexity(model, img_size=256):
    params = sum(p.numel() for p in model.net.parameters())
    print(f"\n[复杂度] Params: {params/1e6:.4f} M")
    net_cpu = model.net.cpu().eval()
    sz = (3, img_size, img_size); flops = None

    if PTFLOPS_AVAILABLE and flops is None:
        try:
            print(f"[复杂度]尝 ptflops（{img_size}x{img_size}）...")
            with torch.no_grad():
                macs, _ = get_model_complexity_info(
                    net_cpu, sz, as_strings=False,
                    print_per_layer_stat=False, verbose=False)
            flops = macs*2
            print(f"[复杂度] ptflops 成功：{flops/1e9:.4f} G")
        except Exception as e:
            print(f"[复杂度] ptflops失败: {e}")

    if THOP_AVAILABLE and flops is None:
        try:
            print(f"[复杂度]尝 thop（{img_size}x{img_size}）...")
            dummy = torch.randn(1, *sz)
            with torch.no_grad():
                macs, _ = thop.profile(net_cpu, inputs=(dummy,), verbose=False)
            flops = macs*2
            print(f"[复杂度] thop 成功：{flops/1e9:.4f} G")
        except Exception as e:
            print(f"[复杂度] thop失败: {e}")

    if flops is None:
        print("[复杂度] GFlops失败，请安装 ptflops 或 thop")
    if torch.cuda.is_available(): model.net.cuda()
    return params, flops


# ---------------------------------------------------------
# 本地实现 NIQE 算法，彻底摆脱所有第三方库依赖报错！
# ---------------------------------------------------------
import math
import scipy.ndimage
import scipy.special
import scipy.io
import urllib.request
import os
import cv2
import numpy as np

def estimate_aggd_param(block):
    """Estimate AGGD parameters."""
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(scipy.special.gamma(gam_reciprocal * 2)) / (
        scipy.special.gamma(gam_reciprocal) * scipy.special.gamma(gam_reciprocal * 3))

    left_block = block[block < 0]
    right_block = block[block > 0]
    left_std = np.sqrt(np.mean(left_block**2)) if left_block.size > 0 else 0.0
    right_std = np.sqrt(np.mean(right_block**2)) if right_block.size > 0 else 0.0

    if left_std == 0:
        left_std = 1e-8
    if right_std == 0:
        right_std = 1e-8
    gammahat = left_std / right_std
    denom = np.mean(block**2)
    if denom == 0:
        denom = 1e-8
    rhat = (np.mean(np.abs(block)))**2 / denom
    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1)**2)

    array_position = np.argmin((r_gam - rhatnorm)**2)
    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(scipy.special.gamma(1 / alpha) / scipy.special.gamma(3 / alpha))
    beta_r = right_std * np.sqrt(scipy.special.gamma(1 / alpha) / scipy.special.gamma(3 / alpha))
    return alpha, beta_l, beta_r

def compute_feature(block):
    """Compute NIQE feature vector with length 18."""
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    for shifts in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        shifted_block = np.roll(block, shifts, axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (scipy.special.gamma(2 / alpha) / scipy.special.gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat

def calculate_niqe(img, mu_prisparam, cov_prisparam):
    """
    真正的 NIQE 算法核心 (与 MATLAB 官方版对齐)
    """
    img = img.astype(np.float64).round()
    block_size_h, block_size_w = 96, 96

    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    if num_block_h == 0 or num_block_w == 0:
        raise ValueError(f"图像尺寸过小，无法按 {block_size_h}x{block_size_w} 分块: {img.shape}")

    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]
    gaussian_window = cv2.getGaussianKernel(7, 7 / 6)
    gaussian_window = np.outer(gaussian_window, gaussian_window.transpose())

    distparam = []
    for scale in (1, 2):
        mu = scipy.ndimage.convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(np.abs(scipy.ndimage.convolve(np.square(img), gaussian_window, mode='nearest') - np.square(mu)))
        img_normalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                block = img_normalized[
                    idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                    idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale
                ]
                feat.append(compute_feature(block))
        distparam.append(np.array(feat))

        if scale == 1:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
            num_block_h = math.floor(img.shape[0] / (block_size_h // 2))
            num_block_w = math.floor(img.shape[1] / (block_size_w // 2))

    features = np.concatenate(distparam, axis=1)
    if features.shape[1] != mu_prisparam.shape[0]:
        raise ValueError(f"特征维度不匹配: features={features.shape[1]}, mu={mu_prisparam.shape[0]}")

    mu_distparam = np.nanmean(features, axis=0)
    features_no_nan = features[~np.isnan(features).any(axis=1)]
    if features_no_nan.shape[0] < 2:
        raise ValueError("有效特征块数量不足，无法计算协方差")
    cov_distparam = np.cov(features_no_nan, rowvar=False)
    
    # 计算与预训练模型的距离 (分数)
    invcov_param = np.linalg.pinv((cov_prisparam + cov_distparam) / 2)
    diff = mu_prisparam - mu_distparam
    score = np.sqrt(np.dot(np.dot(diff, invcov_param), diff.T))
    
    return np.squeeze(score)

def get_niqe_model():
    local_mat = os.path.join(os.getcwd(), 'niqe_modelparameters.mat')
    if not os.path.exists(local_mat):
        raise FileNotFoundError(f"找不到模型文件: {local_mat}")
    
    # 读取 .mat 权重
    data = scipy.io.loadmat(local_mat)
    mu_prisparam = np.ravel(data['mu_prisparam'])
    cov_prisparam = data['cov_prisparam']
    return mu_prisparam, cov_prisparam

def bgr_to_y_channel_255(img_bgr):
    img = img_bgr.astype(np.float32) / 255.0
    y = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    return y.astype(np.float64)

# ---------------------------------------------------------

def measure_iqa(restored_dir, device="cuda", limit=None):
    exts = ("*.png","*.jpg",".jpeg",".bmp")
    files = sorted(set(
        f for ext in exts
        for f in glob.glob(os.path.join(restored_dir, ext)) +
                 glob.glob(os.path.join(restored_dir, ext.upper()))
    ))
    if not files: print(f"[IQA] {restored_dir} 中无图像"); return {}
    if limit: files = files[:limit]; print(f"[IQA]调：只试：只处理前 {limit}张")
    print(f"\n[IQA]共 {len(files)}张，开始计算...")

    musiq_m = maniqa_m = niqe_pyiqa_m = None
    if PYIQA_AVAILABLE:
        print("[IQA] 初始化 pyiqa（首次运行会下载权重）...")
        try: musiq_m  = pyiqa.create_metric("musiq",  device=device); print("[IQA] MUSIQ就绪")
        except Exception as e: print(f"[IQA] MUSIQ失败: {e}")
        try: maniqa_m = pyiqa.create_metric("maniqa", device=device); print("[IQA] MANIQA就绪")
        except Exception as e: print(f"[IQA] MANIQA失败: {e}")
        if NIQE_BACKEND == "pyiqa":
            try: niqe_pyiqa_m = pyiqa.create_metric("niqe", device=device); print("[IQA] NIQE就绪")
            except Exception as e: print(f"[IQA] NIQE失败: {e}")

    niqe_s=[]; musiq_s=[]; maniqa_s=[]
    t0 = time.time()
    for i, p in enumerate(files):
        if (i+1)%200==0:
            el=time.time()-t0; avg=el/(i+1)
            print(f"  [{i+1}/{len(files)}] {el/60:.1f}min，剩余约 {(len(files)-i-1)*avg/60:.1f}min")

        if NIQE_BACKEND=="skvideo":
            bgr = cv2.imread(p)
            if bgr is not None:
                try:
                    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    img_norm = img_rgb.astype(np.float32) / 255.0
                    img_norm = np.expand_dims(img_norm, axis=0)
                    niqe_s.append(float(sk_niqe(img_norm).mean()))
                except: pass
        elif NIQE_BACKEND=="pyiqa" and niqe_pyiqa_m:
            try: niqe_s.append(niqe_pyiqa_m(p).item())
            except: pass
        if musiq_m:
            try: musiq_s.append(musiq_m(p).item())
            except: pass
        if maniqa_m:
            try: maniqa_s.append(maniqa_m(p).item())
            except: pass

    res={}
    if niqe_s:   res["NIQE"]   = float(np.mean(niqe_s))
    if musiq_s:  res["MUSIQ"]  = float(np.mean(musiq_s))
    if maniqa_s: res["MANIQA"] = float(np.mean(maniqa_s))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       type=str, default=None)
    ap.add_argument("--lq_dir",     type=str, default=None)
    ap.add_argument("--save_dir",   type=str, default="./results_snow_real")
    ap.add_argument("--use_tta",    action="store_true")
    ap.add_argument("--skip_infer", action="store_true")
    ap.add_argument("--skip_iqa",   action="store_true")
    ap.add_argument("--only_niqe",  action="store_true", help="只计算NIQE指标")
    ap.add_argument("--img_size",   type=int, default=256)
    ap.add_argument("--limit",      type=int, default=None)
    ap.add_argument("--no_gpu",     action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.no_gpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    model = params_val = flops_val = None
    if args.ckpt and not args.only_niqe:
        if not os.path.exists(args.ckpt): print(f"[Error] ckpt 不存在: {args.ckpt}"); sys.exit(1)
        print(f"\n[模型] 加载 {args.ckpt}")
        model = PromptIRModel.load_from_checkpoint(args.ckpt)
        model.eval()
        if device=="cuda": model.cuda()
        params_val, flops_val = measure_complexity(model, img_size=args.img_size)
    elif not args.skip_infer and not args.only_niqe:
        print("[Error] 未指定 --ckpt"); sys.exit(1)

    if not args.skip_infer and not args.only_niqe:
        if not args.lq_dir: print("[Error]需要 --lq_dir"); sys.exit(1)
        run_inference(model, args.lq_dir, args.save_dir, use_tta=args.use_tta)
    elif not args.only_niqe:
        print(f"\n[跳过推理] 使用 {args.save_dir} 中已有图像")

    iqa = {}
    if not args.skip_iqa:
        if not os.path.isdir(args.save_dir): print(f"[Error] save_dir 不存在: {args.save_dir}"); sys.exit(1)
        
        if args.only_niqe:
            # 只计算NIQE
            print(f"\n[只计算NIQE] 使用 {args.save_dir} 中图像")
            
            try:
                print("[NIQE] 准备使用纯净版本地 NIQE 算法计算真实分数...")
                
                # 直接读取本地权重，不再依赖任何第三方库
                mu_prisparam, cov_prisparam = get_niqe_model()
                print("[NIQE] 本地权重加载成功。初始化评测模型...")
                
                import glob
                import cv2
                from tqdm import tqdm
                import numpy as np
                
                exts = ("*.png","*.jpg",".jpeg",".bmp")
                files = sorted(set(
                    f for ext in exts
                    for f in glob.glob(os.path.join(args.save_dir, ext)) +
                             glob.glob(os.path.join(args.save_dir, ext.upper()))
                ))
                
                if not files: 
                    print(f"[NIQE] {args.save_dir} 中无图像"); 
                    sys.exit(0)
                
                if args.limit: 
                    files = files[:args.limit]
                    print(f"[NIQE] 调试：只处理前 {args.limit} 张")
                
                print(f"\n[NIQE] 共 {len(files)} 张，开始计算...")
                
                niqe_scores = []
                for p in tqdm(files, desc="计算NIQE"):
                    try:
                        img_bgr = cv2.imread(p)
                        if img_bgr is not None:
                            img_y = bgr_to_y_channel_255(img_bgr)
                            score = calculate_niqe(img_y, mu_prisparam, cov_prisparam)
                            niqe_scores.append(float(score))
                    except Exception as e:
                        print(f"  [警告] 处理 {p} 时出错: {e}")
                        continue

                if niqe_scores:
                    avg_niqe = float(np.mean(niqe_scores))
                    print(f"\n[NIQE] 完成！真实平均分数: {avg_niqe:.4f}")
                    iqa["NIQE"] = avg_niqe
                else:
                    print("[NIQE] 未能计算任何有效分数")
            except Exception as e:
                print(f"[Error] NIQE 计算过程中出错: {e}")
                print("请确保 niqe_modelparameters.mat 文件完整且与 measure.py 同级。")
        else:
            iqa = measure_iqa(args.save_dir, device=device, limit=args.limit)

    print("\n" + "="*50)
    print("           最终指标汇总")
    print("="*50)
    if params_val is not None and not args.only_niqe: print(f"  Params       : {params_val/1e6:.4f} M")
    if flops_val  is not None and not args.only_niqe: print(f"  GFlops       : {flops_val/1e9:.4f} G  ({args.img_size}x{args.img_size})")
    elif args.only_niqe: print(f"  Params/GFlops  : 跳过（--only_niqe 模式）")
    else:                      print(f"  GFlops       :失败（pip install ptflops 或 thop）")
    if "NIQE"   in iqa: print(f"  NIQE    ↓    : {iqa['NIQE']:.4f}")
    if "MUSIQ"  in iqa: print(f"  MUSIQ   ↑    : {iqa['MUSIQ']:.4f}")
    if "MANIQA" in iqa: print(f"  MANIQA  ↑    : {iqa['MANIQA']:.4f}")
    if args.only_niqe and "NIQE" not in iqa: print(f"  NIQE    ↓    : 失败")
    print("="*50)

if __name__ == "__main__":
    main()
