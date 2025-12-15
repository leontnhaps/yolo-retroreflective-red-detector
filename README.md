# 🎯 YOLO 재귀반사 필름(적색) 탐지기

<div align="center">

![YOLOv11](https://img.shields.io/badge/YOLO-v11-blue?style=for-the-badge&logo=ultralytics)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-AGPL%20v3-red?style=for-the-badge)

**Ultralytics YOLO를 활용한 고정밀 적색 재귀반사 필름 탐지 모델입니다.**  
다양한 조명 환경에서도 강인한 탐지를 위해 **RGB 원본(Original) 영상**과 **LED 차분(Diff) 영상** 모델을 모두 지원합니다.

</div>

---

## 📊 성능 요약 (Performance Overview)

| 모델 타입 (Model Type) | 베스트 모델 | mAP@0.5 | 정밀도 (Precision) | 재현율 (Recall) | 추천 용도 |
|:---:|:---:|:---:|:---:|:---:|:---|
| **Original (RGB)** | **YOLOv11m** | **0.935** | 0.98 | 0.81 | 일반 환경, 주간/실내 |
| **Diff (Robust)** | **YOLOv11m_Diff** | *TBD* | *TBD* | *TBD* | **강한 주변광**, 야간, 장거리 |

---

## 📂 모델 갤러리

### ✨ Diff 모델 (차분 영상)
> **강인한 성능 (Robustness Choice)** 🛡️  
> `(LED ON - LED OFF)` 차분 영상을 학습했습니다. 배경 노이즈와 주변광 간섭을 제거하는 데 매우 효과적입니다.

<details open>
<summary><b>🔻 클릭하여 Diff 모델 그래프 보기</b></summary>

| **YOLOv11m_Diff (Main)** | **YOLOv11s_Diff (Light)** |
|:---:|:---:|
| **TBD**<br>*(그래프 추가 예정)* | **TBD**<br>*(그래프 추가 예정)* |

#### 성능표 (Diff)
| 모델 | mAP@0.5 | F1-Score | Precision | Recall |
|:---|:---:|:---:|:---:|:---:|
| **YOLOv11m_Diff** | - | - | - | - |
| **YOLOv11s_Diff** | - | - | - | - |

</details>

<br>

### 🧱 Original 모델 (RGB 영상)
> **일반적인 선택 (Standard Choice)** 📷  
> 표준 RGB 영상을 학습했습니다. 통제된 환경에서 높은 정확도를 보여줍니다.

<details>
<summary><b>🔻 클릭하여 Original 모델 그래프 보기</b></summary>

#### YOLOv11m (Main)
> **mAP@0.5**: 0.935 | **Best F1**: 0.91

| Precision-Recall | F1-Confidence |
|:---:|:---:|
| <img src="original_model/YOLOv11m/Yolo11mPrecision_Recall.png" width="100%"/> | <img src="original_model/YOLOv11m/Yolo11mF1_Confidence.png" width="100%"/> |
| **Precision-Confidence** | **Recall-Confidence** |
| <img src="original_model/YOLOv11m/Yolo11mPrecison_Confidence.png" width="100%"/> | <img src="original_model/YOLOv11m/Yolo11mRecall_Confidence.png" width="100%"/> |

<div align="center">
    <img src="original_model/YOLOv11m/Yolo11mConfusionnorm.png" width="60%"/>
    <p>Normalized Confusion Matrix</p>
</div>

<br>

#### YOLOv11s (Light)
> **mAP@0.5**: 0.931 | **Best F1**: 0.91

| Precision-Recall | F1-Confidence |
|:---:|:---:|
| <img src="original_model/YOLOv11s/Yolo11sPrecision_Recall.png" width="100%"/> | <img src="original_model/YOLOv11s/Yolo11sF1_Confidence.png" width="100%"/> |

<br>

#### YOLOv8m (Baseline)
> **mAP@0.5**: 0.808 | **Best F1**: 0.76

| Precision-Recall | F1-Confidence |
|:---:|:---:|
| <img src="original_model/YOLOv8m/Yolo8mPrecision_Recall.png" width="100%"/> | <img src="original_model/YOLOv8m/Yolo8mF1_Confidence.png" width="100%"/> |

</details>

---

## 🧪 테스트 예시
<div align="center">
<img src="test.png" width="60%"/>
</div>
