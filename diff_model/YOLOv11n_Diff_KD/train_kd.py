from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import os
import torch
import torch.nn.functional as F
import gc

# ---------------------------------------------------------
# 0. 메모리 정리 (OOM 방지용)
# ---------------------------------------------------------
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 GPU 메모리 정리 완료")

# ---------------------------------------------------------
# 1. 지식 증류 트레이너 클래스 (수정된 버전)
# ---------------------------------------------------------
class KnowledgeDistillationTrainer(DetectionTrainer):
    def __init__(self, teacher_model_path, dist_loss_weight=0.25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model_path = teacher_model_path
        self.dist_loss_weight = dist_loss_weight
        self.teacher = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        
        # 선생님 모델 로드
        print(f"👨‍🏫 선생님 모델 로딩 중: {self.teacher_model_path}")
        self.teacher = YOLO(self.teacher_model_path).model
        self.teacher.eval() 
        for param in self.teacher.parameters():
            param.requires_grad = False 
        
        return model

    def criterion(self, preds, batch):
        loss, loss_items = super().criterion(preds, batch)
        
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])

        dist_loss = 0
        for stud_out, teach_out in zip(preds, teacher_preds):
            if stud_out.shape == teach_out.shape:
                dist_loss += F.mse_loss(stud_out, teach_out)

        total_loss = loss + (self.dist_loss_weight * dist_loss)
        loss_items[0] += dist_loss.item() 
        
        return total_loss, loss_items

def train_yolo_distill():
    # GPU 메모리 청소
    clear_gpu_memory()

    # 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(script_dir, 'data.yaml')
    
    # ---------------------------------------------------------
    # 2. 경로 설정 (사용자 환경에 맞춤)
    # ---------------------------------------------------------
    student_model_name = 'yolo11n.pt'
    # 선생님 모델 경로 (확인 필수!)
    teacher_model_path = r'C:\Users\user\Desktop\이희준\KD_new\best.pt' 

    print(f"🎓 학생: {student_model_name}, 👨‍🏫 선생님: {teacher_model_path}")

    # ---------------------------------------------------------
    # 3. 학습 설정 (Resume 포함)
    # ---------------------------------------------------------
    args = dict(
        model=student_model_name,
        data=data_yaml_path,
        epochs=300,
        imgsz=1280,
        batch=2,
        device=0,
        name='distilled_model_11n',
        exist_ok=True,
        
        # [핵심] 끊긴 곳에서 이어하기 설정
        # 만약 처음부터 다시 하고 싶다면 False로 바꾸세요.
        resume=True 
    )
    
    trainer = KnowledgeDistillationTrainer(
        teacher_model_path=teacher_model_path,
        overrides=args
    )
    
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n🚨 여전히 메모리가 부족합니다! batch 사이즈를 4로 줄여서 다시 시도해보세요.")
        else:
            raise e

if __name__ == '__main__':
    train_yolo_distill()