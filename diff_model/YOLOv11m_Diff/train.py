from ultralytics import YOLO
import os

def train_yolo():
    # 스크립트 파일의 절대 경로를 기준으로 data.yaml 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(script_dir, 'data.yaml')
    
    # ---------------------------------------------------------
    # 1. 모델 선택 (원하는 모델의 주석을 해제하여 선택하세요)
    # ---------------------------------------------------------
    model_name = 'yolo11m.pt'  # Medium 모델 (성능 균형)
    
    # 처음 실행 시 자동으로 가중치 파일을 다운로드합니다.
    model = YOLO(model_name)

    # ---------------------------------------------------------
    # 2. 학습 실행 (Arguments 설정)
    # ---------------------------------------------------------
    # data: 수정하신 data.yaml 파일의 경로를 입력하세요.
    # epochs: 학습 반복 횟수 (보통 50~100부터 시작해 보시는 것을 추천합니다)
    # imgsz: 이미지 크기 (기본 640, 데이터셋이 작거나 크면 조절)
    # batch: 메모리 용량에 맞춰 조절 (16, 32 등. -1로 하면 자동 설정)
    # device: GPU 사용 시 0, CPU 사용 시 'cpu'
    # ---------------------------------------------------------
    results = model.train(
        data=data_yaml_path,   # data.yaml 파일의 절대 경로
        epochs=300,         
        imgsz=1280,          
        batch=4,           
        device=0,           # GPU 번호 (없으면 'cpu')
        name='reflective_film_model_11m', # 저장될 폴더 이름 (Medium 모델에 맞춰 11m으로 수정)
        exist_ok=True       # 덮어쓰기 허용 여부
    )

    # ---------------------------------------------------------
    # 3. 학습 결과 확인 (Validation)
    # ---------------------------------------------------------
    # 학습이 끝난 후 validation 셋에 대한 성능 지표를 출력합니다.
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

if __name__ == '__main__':
    # Windows 환경에서 멀티프로세싱 오류 방지를 위해 if __name__ == '__main__': 구문이 필요합니다.
    train_yolo()
