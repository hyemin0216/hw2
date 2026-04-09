from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from model import predict_image

app = FastAPI(
    title="Food Classifier API",
    description="MLOps Demo: 음식 이미지 분류 및 칼로리 예측 서버",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "Food Classifier API 가 정상적으로 실행 중입니다."
    }

@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    # 파일 확장자/MIME 타입 검사
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    # 메모리에 이미지 파일 읽기
    contents = await file.read()
    
    # ML 모델 추론 시작
    result = predict_image(contents)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return JSONResponse(content=result)
