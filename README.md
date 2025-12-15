# 🎯 YOLO Retroreflective Film (RED) Detector

<div align="center">

![YOLOv11](https://img.shields.io/badge/YOLO-v11-blue?style=for-the-badge&logo=ultralytics)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**High-precision detector for Red Retroreflective Film using Ultralytics YOLO.**  
Supports both **RGB Original** images and **LED Difference** images for robust detection in various lighting conditions.

</div>

---

## 📊 Performance Overview

| Model Type | Best Model | mAP@0.5 | Precision | Recall | Recommended For |
|:---:|:---:|:---:|:---:|:---:|:---|
| **Original (RGB)** | **YOLOv11m** | **0.935** | 0.98 | 0.81 | General Purpose, Day/Indoor |
| **Diff (Robust)** | **YOLOv11m_Diff** | *TBD* | *TBD* | *TBD* | **Strong Ambient Light**, Night, Long Range |

---

## 📂 Model Gallery

### ✨ Diff Model (Difference Image)
> **Robustness Choice** 🛡️  
> Trained on `(LED ON - LED OFF)` difference images. Extremely effective at removing background noise and ambient light interference.

<details open>
<summary><b>🔻 Click to expand Diff Model Graphs</b></summary>

| **YOLOv11m_Diff (Main)** | **YOLOv11s_Diff (Light)** |
|:---:|:---:|
| **TBD**<br>*(Graphs will be added)* | **TBD**<br>*(Graphs will be added)* |

#### Performance Table (Diff)
| Model | mAP@0.5 | F1-Score | Precision | Recall |
|:---|:---:|:---:|:---:|:---:|
| **YOLOv11m_Diff** | - | - | - | - |
| **YOLOv11s_Diff** | - | - | - | - |

</details>

<br>

### 🧱 Original Model (RGB Image)
> **Standard Choice** 📷  
> Trained on standard RGB images. High accuracy in controlled environments.

<details>
<summary><b>🔻 Click to expand Original Model Graphs</b></summary>

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

## 🧪 Test Sample
<div align="center">
<img src="test.png" width="60%"/>
</div>
