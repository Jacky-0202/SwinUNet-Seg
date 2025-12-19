# Swin-UNet: From Natural Scenes to Medical Imaging

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Medical Segmentation](https://img.shields.io/badge/Medical-Segmentation-blue?style=for-the-badge)

This repository implements **Swin-UNet**, a pure Transformer-based U-Net architecture. While the primary demonstration focuses on **Synapse Multi-Organ Segmentation** (CT scans), the codebase is engineered to be **format-agnostic**, supporting both medical formats (`.npz`, `.h5`) and standard image formats (`.jpg`, `.png`).

---
### 🔬 Domain Analysis: Why Synapse?

The ADE20K Experiment:
Initial experiments were conducted on the ADE20K dataset (Scene Parsing, 150 classes) to evaluate Swin-UNet on complex natural images. The pipeline successfully handled .jpg inputs and multi-class masking.

Strategic Pivot to Medical Imaging:
While the model performed adequately on natural scenes, our analysis revealed that Swin-UNet's inductive bias—specifically its ability to model global context—is far more effective for anatomical structures which have relatively fixed positions and shapes compared to the high variance of natural scenes.

Decision: 
The project focus was shifted to Synapse to maximize the architecture's potential, achieving higher precision with optimized resource utilization.

Note: To switch back to general image training, simply update config.py to point to your .jpg directory and adjust NUM_CLASSES.

![Prediction Result1](figures/predict.png)
![Prediction Result2](figures/training_curves.png)

---
## 📂 Project Structure

```text
├── data/                       # Dataset directory
│   ├── synapse/                # Medical Data (Current Focus)
│   └── ade20k/                 # General Scene Data (Supported)
├── models/
│   └── swinunet.py             # Swin-UNet architecture
├── utils/
│   ├── dataset.py              # Dual-mode dataset loader (Synapse & Standard RGB)
│   ├── logger.py               # Custom CSV Logger
│   ├── loss.py                 # Hybrid Loss (CrossEntropy + Dice)
│   └── metrics.py              # Evaluation metrics
├── config.py                   # Global configuration
├── train.py                    # Unified training script
├── predict_synapse.py          # 3D Inference & Visualization
└── README.md
```

---
### 🧠 Dataset Details

The project uses the **Synapse Multi-Organ CT Dataset**.

- **Classes (9 total)**: 0. Background
    
    1. Aorta
    2. Gallbladder
    3. Kidney (Left)
    4. Kidney (Right)
    5. Liver
    6. Pancreas
    7. Spleen
    8. Stomach

---
### 🛠️ Installation

1. **Clone the repository**
```
git clone https://github.com/Jacky-0202/SwinUNet-Seg.git
cd swin-unet-synapse
```
    
2. **Install dependencies** 
	It is recommended to use a virtual environment (Conda or venv) to avoid conflicts.
    
```bash
pip install -r requirements.txt
```

---
### 🚀 Usage

##### 1. Training

To start training the Swin-UNet model from scratch (or pretrained weights).
The training logs will be saved to results/.

```bash
python train.py
```

- **Configuration**: You can adjust `BATCH_SIZE`, `LR`, `EPOCHS`, and paths in `config.py`.
- **Logging**: Check `results/training_log.csv` for real-time metrics.

##### 2. Inference (Testing)

To evaluate the model on 3D test volumes (`.h5` files) and generate visualization.

```bash
python predict_synapse.py
```

- This script performs slice-by-slice inference, reconstructs the 3D volume, and calculates the **3D Dice Score** for each organ.
- Visualization results (Ground Truth vs. Prediction) will be saved in `test_results_volume/`.

---
### 🤝 Reference

If you find this project useful, please refer to the original Swin-UNet paper:

- **Swin-UNet: Unet-like Pure Transformer for Medical Image Segmentation**
    
    - _H. Cao, Y. Wang, J. Chen, D. Jiang, X. Zhang, Q. Tian, M. Wang_
    - [arXiv:2105.05537](https://arxiv.org/abs/2105.05537)
