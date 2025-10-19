# Chimney-height-regression

## 🚀 Polyline Height Probabilistic Regression

This project predicts polyline height (`chi_height_m`) using a probabilistic regression model.

It uses a multi-modal input (5-channel image + 1D length feature) and a custom architecture to output a probabilistic distribution (`mu`, `log_var`) for the height.

---

## 🔬 Model & Features

### Architecture
* `convnext_base` backbone.

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
* **Local Storage:** Requires Google Drive (`/content/drive`) for loading data.

### Data Source
* **Provider:** [AI Hub (AI허브)](https://www.aihub.or.kr)
* **Dataset:** [대기오염 배출원 공간 분포 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71805)
* **Files Used (as `.zip`):**
    * `TS_KS.zip` (Training Set Images)
    * `TL_KS_LINE.zip` (Training Labels - VIA JSON)
    * `VS_KS.zip` (Validation Set Images)
    * `VL_KS_LINE.zip` (Validation Labels - VIA JSON)

### Data Preprocessing
* **Image:** Letterbox resizing to `(224, 224)`.
* **Normalization:** Z-Normalization for 1D length (input) and target height (label), using stats from the training set.

---

## 🚀 Usage & Result

### Training
* The script computes normalization stats, loads data, and runs training.

### Model Saving
* The best model is saved to `SAVE_PATH` based on the lowest validation RMSE.
* The checkpoint includes both model `state_dict` and normalization `stats`.

### Result
* **Best Validation RMSE: 4.091**
