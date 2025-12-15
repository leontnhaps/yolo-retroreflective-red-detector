# YOLO 재귀반사 필름(적색) 탐지기

<div align="center">

![YOLOv11](https://img.shields.io/badge/YOLO-v11-blue?style=for-the-badge&logo=ultralytics)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-AGPL%20v3-red?style=for-the-badge)

**Ultralytics YOLO 기반 적색 재귀반사 필름 탐지 모델**  
RGB 원본 영상과 LED 차분(Diff) 영상 모델을 제공합니다.

</div>

---

## 성능 요약

| 모델 타입 | 베스트 모델 | mAP@0.5 | Precision | Recall | 특징 |
|:---:|:---:|:---:|:---:|:---:|:---|
| **Original (RGB)** | **YOLOv11m** | **0.935** | 0.98 | 0.81 | 일반 주간/실내 환경 |
| **Diff (Robust)** | **YOLOv11m_Diff** | *TBD* | *TBD* | *TBD* | **강한 주변광**, 야간, 원거리 |

---

## 모델 상세

### 1. Diff 모델 (차분 영상)
> **학습 데이터**: `(LED ON - LED OFF)` 차분 이미지  
> **특징**: 배경 노이즈 제거, 주변광 간섭 최소화

<details open>
<summary><b>▼ Diff 모델 그래프 확인</b></summary>

| **YOLOv11m_Diff (Main)** | **YOLOv11s_Diff (Light)** |
|:---:|:---:|
| **TBD**<br>*(그래프 추가 예정)* | **TBD**<br>*(그래프 추가 예정)* |

#### 상세 지표
| 모델 | mAP@0.5 | F1-Score | Precision | Recall |
|:---|:---:|:---:|:---:|:---:|
| **YOLOv11m_Diff** | - | - | - | - |
| **YOLOv11s_Diff** | - | - | - | - |

</details>

<br>

### 2. Original 모델 (RGB 영상)
> **학습 데이터**: 표준 RGB 이미지  
> **특징**: 통제된 조명 환경에서 높은 정확도

<details>
<summary><b>▼ Original 모델 그래프 확인</b></summary>

#### YOLOv11m (Main)
* **mAP@0.5**: 0.935
* **Best F1**: 0.91

| Precision-Recall | F1-Confidence |
|:---:|:---:|
| <img src="original_model/YOLOv11m/Yolo11mPrecision_Recall.png" width="100%"/> | <img src="original_model/YOLOv11m/Yolo11mF1_Confidence.png" width="100%"/> |
| **Precision-Confidence** | **Recall-Confidence** |
| <img src="original_model/YOLOv11m/Yolo11mPrecison_Confidence.png" width="100%"/> | <img src="original_model/YOLOv11m/Yolo11mRecall_Confidence.png" width="100%"/> |

<div align="center">
    <img src="original_model/YOLOv11m/Yolo11mConfusionnorm.png" width="60%"/>
    <p>Confusion Matrix (Normalized)</p>
</div>

<br>

#### YOLOv11s (Light)
* **mAP@0.5**: 0.931
* **Best F1**: 0.91

| Precision-Recall | F1-Confidence |
|:---:|:---:|
| <img src="original_model/YOLOv11s/Yolo11sPrecision_Recall.png" width="100%"/> | <img src="original_model/YOLOv11s/Yolo11sF1_Confidence.png" width="100%"/> |

<br>

#### YOLOv8m (Baseline)
* **mAP@0.5**: 0.808
* **Best F1**: 0.76

| Precision-Recall | F1-Confidence |
|:---:|:---:|
| <img src="original_model/YOLOv8m/Yolo8mPrecision_Recall.png" width="100%"/> | <img src="original_model/YOLOv8m/Yolo8mF1_Confidence.png" width="100%"/> |

</details>

---

## 테스트 예시
<div align="center">
<img src="test.png" width="60%"/>
</div>
