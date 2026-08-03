# AI Image Enhancer (Semiconductor Image Restoration)

An end-to-end deep learning solution for restoring degraded semiconductor and high-resolution grayscale images using a custom **NAFNet** (Non-linear Activation Free Network) architecture.

---

## 📁 Repository Contents

- **`standalone.py`**: Self-contained Python script for running model inference/evaluation on test images.
- **`best_model.pth`**: Trained model weights checkpoint (`~32 MB`).
- **`training_script.ipynb`**: Complete Google Colab / Jupyter notebook for model training and dataset preparation.
- **`requirements.txt`**: List of required Python packages (`torch`, `opencv-python`, `scikit-image`, `numpy`, etc.).

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Image Restoration & Evaluation
Run the standalone evaluation script to restore degraded images:

```bash
python standalone.py -i <path_to_input_dir> -o <path_to_output_dir> -w best_model.pth
```

**Parameters:**
- `-i`, `--input_dir` *(required)*: Directory containing degraded input files (`.png`, `.jpg`, `.jpeg`, or `.npy`).
- `-o`, `--output_dir` *(required)*: Directory where restored images and `evaluation_summary.json` will be saved.
- `-w`, `--weights` *(optional)*: Path to model weights file (defaults to `best_model.pth`).

---

## 🏋️ Training
To train or fine-tune the model from scratch:
1. Open `training_script.ipynb` in [Google Colab](https://colab.research.google.com/) or a local Jupyter environment.
2. Select a GPU runtime.
3. Run all cells sequentially to train the model and save checkpoints.
