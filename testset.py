import os
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd
import torch

# ---------------------------------------------------------
# 1. 경로 설정 (여기는 그대로 두시면 됩니다)
# ---------------------------------------------------------
models = {
    'Teacher (11m)': r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\yolo-retroreflective-red-detector\diff_model\YOLOv11m_Diff\weights\best.pt",
    'Student KD (11n)': r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\yolo-retroreflective-red-detector\diff_model\YOLOv11n_Diff_KD\weights\best.pt",
    'Student Vanilla (11n)': r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\yolo-retroreflective-red-detector\diff_model\YOLOv11n_Diff\weights\best.pt"
}

# 데이터셋 yaml 경로
data_yaml_path = r"C:\Users\gmlwn\Downloads\reflective_diff.v2-testdata.yolov11\data.yaml"

# 타일링 설정 (2x3 = 6분할)
TILE_ROWS = 2
TILE_COLS = 3
TILE_OVERLAP = 0.15
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# ---------------------------------------------------------
# 2. 타일링 + 원본 통합 추론 함수
# ---------------------------------------------------------
def non_max_suppression(boxes, scores, iou_threshold):
    if len(boxes) == 0: return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) > 0: return indices.flatten()
    return []

def predict_custom(model, img, device='cuda'):
    H, W = img.shape[:2]
    
    # 1. 타일 좌표 계산
    tile_h = int(H / TILE_ROWS)
    tile_w = int(W / TILE_COLS)
    ov_h = int(tile_h * TILE_OVERLAP)
    ov_w = int(tile_w * TILE_OVERLAP)
    
    tiles = []
    base_h = H // TILE_ROWS
    base_w = W // TILE_COLS
    
    for r in range(TILE_ROWS):
        for c in range(TILE_COLS):
            y1 = max(0, r * base_h - ov_h)
            x1 = max(0, c * base_w - ov_w)
            y2 = min(H, (r + 1) * base_h + ov_h) if r < TILE_ROWS - 1 else H
            x2 = min(W, (c + 1) * base_w + ov_w) if c < TILE_COLS - 1 else W
            tiles.append((x1, y1, x2, y2))
            
    # 2. 추론 (배치 처리)
    batch_imgs = [img] 
    batch_infos = [('full', 0, 0)]
    
    for tx1, ty1, tx2, ty2 in tiles:
        batch_imgs.append(img[ty1:ty2, tx1:tx2])
        batch_infos.append(('tile', tx1, ty1))
        
    try:
        results_list = model.predict(batch_imgs, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=device, verbose=False)
    except Exception as e:
        # print(f"\n[Error] 추론 중 오류: {e}") 
        return []
    
    all_boxes = []
    all_scores = []
    
    for i, result in enumerate(results_list):
        dtype, off_x, off_y = batch_infos[i]
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                c = float(box.conf.cpu().numpy())
                
                gx1, gy1 = x1 + off_x, y1 + off_y
                gx2, gy2 = x2 + off_x, y2 + off_y
                
                all_boxes.append([int(gx1), int(gy1), int(gx2-gx1), int(gy2-gy1)])
                all_scores.append(c)
                
    indices = non_max_suppression(all_boxes, all_scores, iou_threshold=IOU_THRESHOLD)
    
    final_boxes = [] 
    for idx in indices:
        x, y, w, h = all_boxes[idx]
        final_boxes.append([x, y, x+w, y+h])
        
    return final_boxes

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter / (area1 + area2 - inter + 1e-6)

def evaluate_model(model_name, model_path, img_paths, label_paths):
    print(f"🚀 [{model_name}] 타일링 평가 시작...")
    
    if not os.path.exists(model_path):
        print(f"⚠️ 모델 파일 없음: {model_path}")
        return None
        
    try:
        model = YOLO(model_path)
    except:
        print(f"⚠️ 모델 로드 실패: {model_path}")
        return None

    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for img_path, lbl_path in tqdm(zip(img_paths, label_paths), total=len(img_paths)):
        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        
        pred_boxes = predict_custom(model, img, device=device)
        
        gt_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        _, cx, cy, w, h = parts[:5]
                        x1 = int((cx - w/2) * w_img)
                        y1 = int((cy - h/2) * h_img)
                        x2 = int((cx + w/2) * w_img)
                        y2 = int((cy + h/2) * h_img)
                        gt_boxes.append([x1, y1, x2, y2])

        matched_gt = [False] * len(gt_boxes)
        
        for p_box in pred_boxes:
            best_iou = 0
            best_idx = -1
            for i, g_box in enumerate(gt_boxes):
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou >= 0.5 and not matched_gt[best_idx]:
                tp_total += 1
                matched_gt[best_idx] = True
            else:
                fp_total += 1 
        
        fn_total += matched_gt.count(False)

    precision = tp_total / (tp_total + fp_total + 1e-6)
    recall = tp_total / (tp_total + fn_total + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {
        'Model': model_name, 
        'Precision': round(precision, 4), 
        'Recall': round(recall, 4), 
        'F1-Score': round(f1, 4)
    }

# ---------------------------------------------------------
# 4. 메인 실행 (경로 보정 로직 추가됨)
# ---------------------------------------------------------
if __name__ == '__main__':
    if not os.path.exists(data_yaml_path):
        print("🚨 data.yaml 파일이 없습니다.")
        exit()
        
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    # test 경로 가져오기 (없으면 val)
    test_dir_rel = data_cfg.get('test', data_cfg.get('val'))
    
    # 1차 시도: YAML 대로 경로 생성
    if os.path.isabs(test_dir_rel):
        test_dir = test_dir_rel
    else:
        test_dir = os.path.abspath(os.path.join(os.path.dirname(data_yaml_path), test_dir_rel))
    
    # 2차 시도: 경로가 없으면 'data.yaml' 폴더 내부의 'test/images' 탐색 (자동 보정)
    if not os.path.exists(test_dir):
        # 상위(..)로 나갔던 걸 취소하고 현재 폴더 안에서 찾기
        alternative_dir = os.path.join(os.path.dirname(data_yaml_path), 'test', 'images')
        
        if os.path.exists(alternative_dir):
            print(f"⚠️ YAML 경로({test_dir})가 잘못되어 자동으로 수정했습니다 -> {alternative_dir}")
            test_dir = alternative_dir
        else:
            # images 폴더 없이 그냥 test 폴더인 경우
            alternative_dir_2 = os.path.join(os.path.dirname(data_yaml_path), 'test')
            if os.path.exists(alternative_dir_2):
                test_dir = alternative_dir_2
            else:
                print(f"🚨 테스트 이미지 폴더를 찾을 수 없습니다:\n1. {test_dir}\n2. {alternative_dir}")
                print("💡 해결법: data.yaml 파일 옆에 'test' 폴더가 있는지, 그 안에 이미지가 있는지 확인해주세요.")
                exit()
        
    print(f"📂 테스트 데이터 폴더 확정: {test_dir}")
        
    # 이미지 파일 찾기
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    img_paths = []
    for root, dirs, files in os.walk(test_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in img_exts:
                img_paths.append(os.path.join(root, f))
    
    if len(img_paths) == 0:
        print("🚨 폴더 안에 이미지가 한 장도 없습니다! 경로를 다시 확인해주세요.")
        exit()

    # 라벨 파일 경로 생성
    label_paths = []
    for ip in img_paths:
        # 1. 'images' 폴더명을 'labels'로 변경
        lp = ip.replace('images', 'labels')
        # 2. 확장자 변경 (.jpg -> .txt)
        lp = os.path.splitext(lp)[0] + '.txt'
        label_paths.append(lp)
    
    print(f"🔥 총 {len(img_paths)}장 이미지에 대해 타일링 검증 시작!\n")
    
    results = []
    for name, path in models.items():
        res = evaluate_model(name, path, img_paths, label_paths)
        if res: 
            results.append(res)
            print(f"   -> Result: {res}\n")
        
    # 결과 출력
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("🏆 [Tiling + Full NMS] 최종 실전 성능 비교 🏆")
    print("="*60)
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("결과가 없습니다.")
    print("="*60)