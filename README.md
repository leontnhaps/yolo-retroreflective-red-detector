# YOLO Retroreflective Film (RED) Detector
A single-class detector(RetroreflectiveRED) trained with Ultralytics YOLO for red retroereflective flim.
We provide YOLOv11m/YOLOv11s weights for production use; YOLOv8m is a baseline (test) model for comparison.

- Small/Long-range targets : try imgsz = 1024~
  
## Performance Summary

| Model            | mAP@0.5           | BestF1(conf)      | Precision | Recall   |
|:-----------------|:------------------|:------------------|:----------|:---------|
| Yolov8m          | 0.808              | 0.76(0.018)      | 0.94      | 0.57    |
| Yolov11s         | 0.931              | 0.91(0.060)      | 0.96      | 0.82    |
| Yolov11m         | 0.935              | 0.91(0.072)      | 0.98      | 0.81    |

## Per-Model Graphs
YOLOv11m (Main)
<details><summary>Show graphs</summary> <p align="center"> <!-- Precision–Recall --> <img src="YOLOv11m/Yolo11mPrecision_Recall.png" width="45%"/> <!-- F1–Confidence --> <img src="YOLOv11m/Yolo11mF1_Confidence.png" width="45%"/> </p> <p align="center"> <!-- Precision–Confidence (오타 포함 실제 파일명: Precison) --> <img src="YOLOv11m/Yolo11mPrecison_Confidence.png" width="45%"/> <!-- Recall–Confidence --> <img src="YOLOv11m/Yolo11mRecall_Confidence.png" width="45%"/> </p> <p align="center"> <!-- Confusion Matrix (raw / normalized) --> <img src="YOLOv11m/Yolo11mConfusion.png" width="45%"/> <img src="YOLOv11m/Yolo11mConfusionnorm.png" width="45%"/> </p> <p align="center"> <!-- Train/Val results & data dist --> <img src="YOLOv11m/Yolo11mResult.png" width="45%"/> <img src="YOLOv11m/Yolo11mdata.jpg" width="45%"/> </p> </details>
YOLOv11s (Light/Recall)
<details><summary>Show graphs</summary> <p align="center"> <img src="YOLOv11s/Yolo11sPrecision_Recall.png" width="45%"/> <img src="YOLOv11s/Yolo11sF1_Confidence.png" width="45%"/> </p> <p align="center"> <img src="YOLOv11s/Yolo11sPrecison_Confidence.png" width="45%"/> <img src="YOLOv11s/Yolo11sRecall_Confidence.png" width="45%"/> </p> <p align="center"> <img src="YOLOv11s/Yolo11sConfusion.png" width="45%"/> <img src="YOLOv11s/Yolo11sConfusionnorm.png" width="45%"/> </p> <p align="center"> <img src="YOLOv11s/Yolo11sResult.png" width="45%"/> <img src="YOLOv11s/Yolo11sdata.jpg" width="45%"/> </p> </details>
YOLOv8m (Baseline/Test)
<details><summary>Show graphs</summary> <p align="center"> <img src="YOLOv8m/Yolo8mPrecision_Recall.png" width="45%"/> <img src="YOLOv8m/Yolo8mF1_Confidence.png" width="45%"/> </p> <p align="center"> <img src="YOLOv8m/Yolo8mPrecison_Confidence.png" width="45%"/> <img src="YOLOv8m/Yolo8mRecall_Confidence.png" width="45%"/> </p> <p align="center"> <img src="YOLOv8m/Yolo8mConfusion.png" width="45%"/> <img src="YOLOv8m/Yolo8mConfusionnorm.png" width="45%"/> </p> <p align="center"> <img src="YOLOv8m/Yolo8mResult.png" width="45%"/> <img src="YOLOv8m/Yolo8mdata.jpg" width="45%"/> </p> </details>
