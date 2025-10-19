# Chimney-height-regression

## 🚀 Polyline Height Probabilistic Regression

This project predicts polyline height (`chi_height_m`) using a probabilistic regression model.

It uses a multi-modal input (5-channel image + 1D length feature) and a custom architecture to output a probabilistic distribution (`mu`, `log_var`) for the height.

---

## 🔬 Model & Features

### Architecture
* `convnext_tiny` backbone.

### Input Fusion
* **FiLM (Feature-wise Linear Modulation):** Injects the 1D length feature into the image backbone.
* **Gating Mechanism:** Combines image features and the 1D feature for the final head.

### Input (Multi-modal)
* **5-Channel Image:** `(B, 5, 224, 224)`
    * `RGB (3)` + `Polyline Mask (1)` + `Padding Mask (1)`
* **1D Feature:** `(B, 1)`
    * Normalized polyline length.

### Output (Probabilistic)
* `mu` (mean) and `log_var` (log variance) of the predicted height.
* **Loss Function:** `GaussianNLLLoss` (Gaussian Negative Log Likelihood).

---

## 🛠️ Setup & Data

* **Environment:** Google Colab.
* **Dependencies:** `torch`, `timm`, `opencv-python-headless`.
* **Data Source:** Requires mounting Google Drive (`/content/drive`).
* **Input Files:** `.tar` archives for images and labels (VIA JSON format).
    * `TS_KS.tar` (Train Images), `TL_LINE.tar` (Train Labels)
    * `VS_KS.tar` (Validation Images), `VL_LINE.tar` (Validation Labels)
* **Data Preprocessing:**
    * **Image:** Letterbox resizing to `(224, 224)`.
    * **Normalization:** Z-Normalization for 1D length (input) and target height (label), using stats from the training set.

---

## 🚀 Usage & Result

### Training
* Run all cells in the `Misson2_markdown+주석.ipynb` notebook.
* The script computes normalization stats, loads data, and runs training.

### Model Saving
* The best model is saved to `SAVE_PATH` based on the lowest validation RMSE.
* The checkpoint includes both model `state_dict` and normalization `stats`.

### Result
* **Best Validation RMSE: 4.091**
