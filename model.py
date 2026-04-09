import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import io
import hashlib

# 1. 기존 파이프라인: 음식 분류 모델 로드
weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model.eval()
preprocess = weights.transforms()

CALORIE_DB = {
    "pizza": 266, "hamburger": 295, "hotdog": 290,
    "ice cream": 207, "guacamole": 157, "strawberry": 32,
    "banana": 89, "apple": 52
}

# 2. 신규 파이프라인: 성별 식별 모델 로직 (MLOps CI/CD 데모용)
# 운영 환경 시 이 부분에 FaceNet 등 실제 안면 인식 및 성별 분류 모델 가중치를 로드하여 대체하게 됩니다.
# 현재는 데모 진행을 위해 이미지 바이트를 해싱하여 의사 난수 기반으로 결과를 도출하는 Mock Model 형태입니다.
def predict_gender_demo(image_bytes: bytes):
    hash_val = int(hashlib.md5(image_bytes).hexdigest(), 16)
    
    # 해시 짝/홀수로 성별 판정
    gender = "Female" if hash_val % 2 == 0 else "Male"
    # 해시 값 기반 50~99% 사이의 임의 신뢰도 생성
    confidence = 50.0 + (hash_val % 5000) / 100.0  
    
    return {
        "gender": gender,
        "gender_confidence": round(confidence, 2)
    }

def predict_image(image_bytes: bytes):
    try:
        # [공통] 업로드된 이미지 분석 준비
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # --- (기존 파이프라인) Task 1: 음식 분류 ---
        batch = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            prediction = model(batch).squeeze(0).softmax(0)
            
        class_id = prediction.argmax().item()
        food_confidence = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]
        
        calories = 250
        for food_key, cal_value in CALORIE_DB.items():
            if food_key in category_name.lower():
                calories = cal_value
                break
                
        # --- (신규 파이프라인) Task 2: 성별 분류 ---
        gender_result = predict_gender_demo(image_bytes)

        # 모델 앙상블 및 최종 API Response 취합 반환
        return {
            "food_class": category_name,
            "food_confidence": round(food_confidence * 100, 2),
            "estimated_calories_per_100g": calories,
            "predicted_gender": gender_result["gender"],
            "gender_confidence": gender_result["gender_confidence"]
        }
    except Exception as e:
        return {"error": str(e)}
