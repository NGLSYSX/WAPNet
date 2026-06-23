# 🚀 WAPNet

**Degradation-Conditioned Pattern Representation and Low-Rank Prompt Modulation for Unified Multi-Weather Image Restoration**

WAPNet is a degradation-conditioned unified multi-weather image restoration framework. It combines continuous weather condition encoding with low-rank prompt modulation to handle adverse-weather degradation scenarios, including raindrops, snowfall, rain-fog, low-light, and composite degradations.

---

## 🛠️ Environment

All experiments were conducted on a Linux workstation with a single NVIDIA RTX 4080 GPU. The model was implemented based on Python 3.8 and PyTorch 2.3.0.

The main training settings used in the paper are as follows:

```text
Operating system: Linux
GPU: NVIDIA RTX 4080
Python version: 3.8
PyTorch version: 2.3.0
Input patch size: 128 × 128
Batch size: 6
Optimizer: AdamW
AdamW betas: β1 = 0.9, β2 = 0.999
Initial learning rate: 2e-4
Learning rate schedule: 15-epoch linear warm-up followed by cosine annealing
Total training epochs: 200
Mixed precision training: enabled
```

Create the environment using:

```bash
conda env create -f env.yml
conda activate wapnet
```

Alternatively, install the required packages with:

```bash
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```text
WAPNet/
├── net/                   # Network architecture and model modules
├── utils/                 # Utility functions
├── train.py               # Training script
├── test_allweather.py     # Testing script for the All-Weather benchmark
├── measure.py             # Model complexity and no-reference IQA measurement script
├── options.py             # Argument and configuration settings
├── env.yml                # Conda environment file
├── requirements.txt       # Python dependencies
└── README.md              # Reproducibility instructions
```

---

## 📦 Dataset Preparation

### All-Weather Dataset

The All-Weather dataset for all-in-one adverse weather image restoration can be downloaded from:

```text
https://drive.google.com/file/d/1tfeBnjZX1wIhIFPl6HOzzOKOyo0GdGHl/view?usp=sharing
```

The corresponding data split files and text files can be downloaded from:

```text
https://drive.google.com/file/d/1UsazX-P3sPcDGw3kxkyFWqUyNfhYN_AM/view?usp=sharing
```

For training, the paired low-quality and ground-truth images should be organized as:

```text
datasets/
└── allweather/
    ├── lq/
    └── gt/
```

where `lq/` contains degraded images and `gt/` contains the corresponding clean images. The degraded and clean images should have matched file names.

For testing, each subset should also be organized into paired `lq/` and `gt/` folders. A recommended structure is:

```text
datasets/
└── test/
    ├── RainDrop/
    │   ├── lq/
    │   └── gt/
    ├── Snow100K-S/
    │   ├── lq/
    │   └── gt/
    ├── Snow100K-L/
    │   ├── lq/
    │   └── gt/
    └── Test1/              # Outdoor-Rain test subset
        ├── lq/
        └── gt/
```

In this repository, `Test1` corresponds to the Outdoor-Rain test subset reported in the paper.

Please replace the dataset paths in the following commands according to your local directory.

### CDD-11 Dataset

The CDD-11 dataset can be downloaded from:

```text
https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvcyFBczNyQ0RST25yYkxncXBlekc0c2FvLXU5ZGREaHc%5FZT1BMFJFSHg&id=CBB69E4E3408EBCD%2138238&cid=CBB69E4E3408EBCD
```

The dataset should be organized according to the paired degraded/clean image structure required by the training or testing scripts.

---

## 🏋️ Training

### Training WAPNet on the All-Weather Dataset

Run the following command to train WAPNet on the All-Weather dataset:

```bash
mkdir -p checkpoints logs

python train.py \
  --ckpt_dir checkpoints \
  --log_dir logs \
  --de_type allweather \
  --allweather_lq ./datasets/allweather/lq/ \
  --allweather_gt ./datasets/allweather/gt/ \
  --batch_size 6 \
  --epochs 200 \
  --patch_size 128 \
  --lr 0.0002 \
  --num_gpus 1 \
  --num_workers 8
```

After training, checkpoints will be saved in the directory specified by `--ckpt_dir`.

---

## 🧪 Testing

### Testing on the All-Weather Benchmark

The script `test_allweather.py` supports testing on multiple All-Weather subsets in a single run, including RainDrop, Snow100K-S, Snow100K-L, and Test1. In this repository, `Test1` corresponds to the Outdoor-Rain test subset reported in the paper.

Run the following command:

```bash
mkdir -p results

python test_allweather.py \
  --ckpt_path checkpoints/<checkpoint_name>.ckpt \
  --output_root results \
  --experiment_name wapnet_allweather \
  --raindrop_lq ./datasets/test/RainDrop/lq/ \
  --raindrop_gt ./datasets/test/RainDrop/gt/ \
  --snow100k_s_lq ./datasets/test/Snow100K-S/lq/ \
  --snow100k_s_gt ./datasets/test/Snow100K-S/gt/ \
  --snow100k_l_lq ./datasets/test/Snow100K-L/lq/ \
  --snow100k_l_gt ./datasets/test/Snow100K-L/gt/ \
  --test1_lq ./datasets/test/Test1/lq/ \
  --test1_gt ./datasets/test/Test1/gt/ \
  --batch_size 1 \
  --num_workers 6 \
  --amp \
  --no_save
```

Please replace `checkpoints/<checkpoint_name>.ckpt` with the checkpoint generated during training.

The option `--no_save` disables saving restored images. To save restored images, remove `--no_save`. The restored images will be saved under:

```text
results/wapnet_allweather/
```

### Testing with Test-Time Augmentation

To enable the geometry-based test-time augmentation strategy used in the paper, add `--use_tta`:

```bash
python test_allweather.py \
  --ckpt_path checkpoints/<checkpoint_name>.ckpt \
  --output_root results \
  --experiment_name wapnet_allweather_tta \
  --raindrop_lq ./datasets/test/RainDrop/lq/ \
  --raindrop_gt ./datasets/test/RainDrop/gt/ \
  --snow100k_s_lq ./datasets/test/Snow100K-S/lq/ \
  --snow100k_s_gt ./datasets/test/Snow100K-S/gt/ \
  --snow100k_l_lq ./datasets/test/Snow100K-L/lq/ \
  --snow100k_l_gt ./datasets/test/Snow100K-L/gt/ \
  --test1_lq ./datasets/test/Test1/lq/ \
  --test1_gt ./datasets/test/Test1/gt/ \
  --batch_size 1 \
  --num_workers 6 \
  --amp \
  --use_tta \
  --no_save
```

The script prints the PSNR and SSIM of each subset and the average result over all tested subsets.

---

## 📊 Output Example

After testing, the terminal will print results in the following format:

```text
==================== Final Results ====================
RainDrop        | PSNR: xx.xxxx | SSIM: x.xxxx
Snow100K-S      | PSNR: xx.xxxx | SSIM: x.xxxx
Snow100K-L      | PSNR: xx.xxxx | SSIM: x.xxxx
Test1           | PSNR: xx.xxxx | SSIM: x.xxxx
-------------------------------------------------------
Average         | PSNR: xx.xxxx | SSIM: x.xxxx
=======================================================
```

---

## 📏 Model Complexity Measurement

The script `measure.py` is used to measure the number of parameters and computational complexity of WAPNet.

Run:

```bash
python measure.py \
  --ckpt checkpoints/<checkpoint_name>.ckpt \
  --skip_iqa \
  --img_size 256
```

Please replace `checkpoints/<checkpoint_name>.ckpt` with the checkpoint generated during training.

The script reports the number of parameters and computational complexity of the model. If optional packages such as `thop`, `ptflops`, or `pyiqa` are installed, additional complexity or image-quality measurements can also be computed.

---

## 📌 Notes

1. The degraded and clean images should have matched file names.
2. Images whose sizes are not divisible by 8 are automatically cropped by the testing script.
3. For fair comparison, the same checkpoint and the same test-set paths should be used when reproducing the reported results.
4. Test-time augmentation improves output stability but increases inference time.
