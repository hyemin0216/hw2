# 1. 가볍고 안정적인 기반 이미지 적용 (Debian Slim 기반)
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 환경 변수 설정 (패키지 최적화, 파이썬 로그 버퍼링 방지)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. 필수 시스템 패키지 업데이트 (PIL 등 라이브러리를 위한 기본 패키지)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. 의존성 파일 복사
COPY requirements.txt .

# [매우 중요] 6. PyTorch CPU 전용 버전 설치 및 패키지 캐시 무효화 (초경량 최적화)
# 일반 pip install을 하면 CUDA가 포함되어 이미지가 수 GB 단위로 폭증합니다.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 7. 프로젝트 소스코드 복사
COPY . .

# 8. 컨테이너 개방 포트 알림 역할
EXPOSE 8000

# 9. 서버 실행 명령어 (host를 0.0.0.0으로 잡아 외부 접속 허용)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
