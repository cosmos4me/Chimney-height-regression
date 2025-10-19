# Chimney-height-regression

# 🚀 Mission 2: 폴리라인 높이 예측 확률적 회귀 모델

이 프로젝트는 이미지와 이미지 위에 정의된 폴리라인(polyline) 정보를 입력받아, 해당 폴리라인의 실제 높이(`chi_height_m`)를 예측하는 확률적 회귀(Probabilistic Regression) 모델입니다.

모델은 **5채널 이미지**와 **정규화된 폴리라인 길이**라는 두 가지 종류의 입력을 동시에 사용하며, **ConvNeXt** 백본, **FiLM (Feature-wise Linear Modulation)**, **Gating 메커니즘**을 결합한 독특한 아키텍처를 특징으로 합니다.

## 📖 목차

- [주요 특징 (Key Features)](#-주요-특징-key-features)
- [모델 아키텍처 (Model Architecture)](#-모델-아키텍처-model-architecture)
- [설치 (Installation)](#-설치-installation)
- [데이터 준비 (Data Setup)](#-데이터-준비-data-setup)
- [사용 방법 (Usage)](#-사용-방법-usage)
- [학습 결과 (Results)](#-학습-결과-results)

---

## ✨ 주요 특징 (Key Features)

* **다중 입력 (Multi-modal Input):**
    1.  **5-채널 이미지:** `RGB (3채널)` + `폴리라인 마스크 (1채널)` + `패딩 마스크 (1채널)`.
    2.  **1D 특징 벡터:** 정규화된 폴리라인의 길이 (`poly_length_norm`).
* **고급 아키텍처:**
    * **FiLM:** 1D 길이 특징을 사용하여 ConvNeXt 백본에서 추출된 2D 이미지 특징을 동적으로 조절합니다.
    * **Gating:** 이미지 특징과 길이 특징을 모두 사용하여 1D 길이 특징의 중요도를 조절하는 게이트를 학습합니다.
* **확률적 예측 (Probabilistic Prediction):**
    * 모델은 단일 값을 예측하는 대신, 예측의 평균(`mu`)과 분산(`log_var`)을 출력합니다.
    * **Gaussian NLL (음의 로그 우도)** 손실 함수를 사용하여 모델의 불확실성(uncertainty)까지 학습합니다.
* **전처리:**
    * **Letterbox:** 이미지의 종횡비를 유지하면서 (224, 224) 크기로 리사이즈하고, 남는 영역은 패딩 처리합니다.
    * **Z-Normalization:** 입력 길이 특징과 타깃 높이 값을 학습 데이터셋의 평균/표준편차로 정규화하여 학습 안정성을 높입니다.

---

## 🔬 모델 아키텍처 (Model Architecture)

이 모델(`ProbFiLMGate`)은 두 개의 입력 스트림을 영리하게 결합합니다.

1.  **이미지 스트림 (Image Stream):**
    * 5채널 이미지(B, 5, 224, 224)가 `timm`의 `convnext_tiny` 백본을 통과하여 이미지 특징 벡터 `f`를 추출합니다.
2.  **특징 스트림 (Feature Stream):**
    * 1D 길이 특징(B, 1)이 별도의 MLP(`len_to_film`)를 통과하여 FiLM 파라미터인 `gamma`와 `beta`를 생성합니다.
3.  **결합 (Fusion):**
    * **FiLM 적용:** `f_film = f * (1 + gamma) + beta` 연산을 통해 이미지 특징이 길이 특징에 의해 조절됩니다.
    * **Gating 적용:** 1D 길이 특징은 `f`와 `poly_z`를 모두 입력받는 `gate_mlp`를 통과하여 `len_g` (게이트가 적용된 길이 특징)가 됩니다.
4.  **헤드 (Head):**
    * 최종적으로 `f_film`과 `len_g`가 결합되어(concatenate) 확률적 회귀 헤드(`head`)를 통과합니다.
    * 출력은 Z-정규화된 공간의 `mu_z` (평균)와 `log_var` (로그 분산)입니다.

---

## 🛠️ 설치 (Installation)

이 프로젝트는 Google Colab 환경에서 실행하는 것을 기준으로 합니다.

1.  **리포지토리 클론 (필요시):**
    ```bash
    # git clone [your-repo-url]
    # cd [your-repo-name]
    ```

2.  **Google Drive 마운트:**
    * 노트북 코드에 포함된 `drive.mount('/content/drive')`를 실행하여 Google Drive에 접근해야 합니다.

3.  **필요한 라이브러리 설치:**
    (Colab 기본 라이브러리 외 추가 설치 필요)
    ```bash
    pip install -q timm opencv-python-headless
    ```
    * **주요 의존성:** `torch`, `torchvision`, `timm`, `cv2`, `numpy`, `tqdm`.

---

## 📂 데이터 준비 (Data Setup)

모델 학습을 위해 특정 구조의 데이터가 Google Drive에 준비되어 있어야 합니다.

1.  **소스 데이터 (Google Drive):**
    * 다음 `.tar` 파일들이 Google Drive 경로(예: `/content/drive/MyDrive/`)에 존재해야 합니다:
        * `TS_KS.tar` (Training Set Images)
        * `VS_KS.tar` (Validation Set Images)
        * `TL_LINE.tar` (Training Labels - VIA JSON)
        * `VL_LINE.tar` (Validation Labels - VIA JSON)

2.  **압축 해제:**
    * 노트북의 "2. 데이터 압축 해제 유틸 및 실행" 섹션을 실행합니다.
    * 이 유틸은 `.tar` 파일을 로컬로 복사 후, 내부의 `*.zip.part0` 파일들을 찾아 지정된 로컬 경로에 압축을 해제합니다.
    * **최종 데이터 경로 (로컬):**
        * `/content/training_set`
        * `/content/validation_set`
        * `/content/tline_label`
        * `/content/vline_label`

3.  **레이블 형식:**
    * 레이블은 VIA `*.json` 형식이며, `region_attributes`의 `chi_height_m` 필드를 타깃 높이(Y)로 사용합니다.

---

## 🚀 사용 방법 (Usage)

`Misson2_markdown+주석.ipynb` 노트북 파일 자체가 전체 학습 파이프라인입니다.

1.  **하이퍼파라미터 설정:**
    * 노트북 상단의 `HYPERPARAMETERS` 셀에서 주요 설정값( `BATCH_SIZE`, `EPOCHS`, `LR` 등)을 조정할 수 있습니다.

2.  **학습 실행:**
    * 노트북의 모든 셀을 위에서부터 순서대로 실행합니다.
    * **Dataset 생성:** `ImageMaskHeightDataset`이 `compute_stats=True`로 인스턴스화되어 학습셋의 평균/표준편차(`stats`)를 계산합니다.
    * **검증셋 로드:** 검증셋은 학습셋에서 계산된 `stats`를 주입받아 동일한 정규화를 적용합니다 (Data Leakage 방지).
    * **학습 루프:** `gaussian_nll` 손실을 최소화하도록 모델을 학습합니다.

3.  **모델 저장:**
    * 학습 중 검증셋의 **RMSE(Root Mean Squared Error)**가 가장 낮은 시점의 모델이 `SAVE_PATH` (예: `/content/drive/MyDrive/M2_model.pth`)에 저장됩니다.
    * 체크포인트에는 모델 가중치(`state_dict`)뿐만 아니라 정규화에 사용된 `stats` 딕셔너리도 함께 저장됩니다.

---

## 📈 학습 결과 (Results)

노트북 실행 로그 기준, 모델은 다음과 같은 성능을 달성했습니다:

* **최고 성능:** Validation **RMSE 4.830** (Epoch 14)
