import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import io

# 모델 가중치 로드 (가벼운 MobileNet V3 Small 모델 사용)
weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model.eval()

preprocess = weights.transforms()

# 데모용 간이 칼로리 데이터베이스 (kcal / 100g)
# 실제 서비스에서는 세분화된 모델(Food-101 등)을 파인튜닝해야 합니다.
CALORIE_DB = {
    "pizza": 266,
    "hamburger": 295,
    "hotdog": 290,
    "ice cream": 207,
    "guacamole": 157,
    "strawberry": 32,
    "banana": 89,
    "apple": 52
}

def predict_image(image_bytes: bytes):
    try:
        # 이미지 전처리
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        batch = preprocess(image).unsqueeze(0)
        
        # 모델 추론
        with torch.no_grad():
            prediction = model(batch).squeeze(0).softmax(0)
            
        class_id = prediction.argmax().item()
        confidence = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]
        
        # 데모 로직: 클래스명에 특정 단어가 포함되어 있으면 매핑, 없으면 임의의 250 kcal 반환
        calories = 250 # Default fallback
        for food_key, cal_value in CALORIE_DB.items():
            if food_key in category_name.lower():
                calories = cal_value
                break

        return {
            "food_class": category_name,
            "confidence": round(confidence * 100, 2),
            "estimated_calories_per_100g": calories
        }
    except Exception as e:
        return {"error": str(e)}
