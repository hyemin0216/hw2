from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from model import predict_image

app = FastAPI(
    title="Food & Gender Classifier API",
    description="MLOps Demo: 음식 분류 및 성별 식별(신규 기능) 분석 서버",
    version="2.0.0"
)

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "message": "(v2.0 업데이트) Food & Gender Classifier API 배포가 완료되었습니다!"
    }

@app.post("/predict")
async def predict_food_and_gender(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    contents = await file.read()
    result = predict_image(contents) # 음식 및 성별 분류 로직 앙상블 실행
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return JSONResponse(content=result)
