# AI Image Enhancer (Semiconductor Image Restoration)

An end-to-end deep learning pipeline for restoring degraded microscopic and high-resolution zoomed images. Powered by a custom **NAFNet** (Non-linear Activation Free Network) architecture, it performs $2\times$ super-resolution and enhancement, upscaling low-resolution inputs ($128 \times 128$) to high-quality outputs ($256 \times 256$).

---

## Repository Contents

- **`standalone.py`**: Self-contained Python script for running model inference/evaluation on test images.
- **`best_model.pth`**: Trained model weights checkpoint (`~32 MB`).
- **`notebook.ipynb`**: Complete Google Colab / Jupyter notebook for model training and dataset preparation.
- **`requirements.txt`**: List of required Python packages (`torch`, `opencv-python`, `scikit-image`, `numpy`, etc.).
- **`Restored Test Outputs/`**: Directory containing sample outputs generated from model evaluation.
---

## Quick Start

### 1. Environment Setup & Dependencies

```bash
# Create and activate virtual environment
pip install -r requirements.txt
# activate On Windows: .\venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```

### 2. Run Image Restoration
Run the standalone evaluation script to restore degraded images:

```bash
Step 1 :- Run the standalone.py
Step 2 :- Enter Input File or Directory Path: <Path of input folder>
          Enter Output Directory Path: <Path of output folder>
```

## Training
To train or fine-tune the model from scratch:
1. Open `training_script.ipynb` in [Google Colab](https://colab.research.google.com/) or a local Jupyter environment.
2. Select a GPU runtime.
3. Run all cells sequentially to train the model and save checkpoints.
