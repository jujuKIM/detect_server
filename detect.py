from flask import Flask, request, jsonify
from flask_cors import CORS  # 추가
import torch
from PIL import Image
import base64
import io

app = Flask(__name__)
# CORS 설정
cors = CORS(app)  # 추가

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    # 클라이언트로부터 전송된 이미지 데이터 받기
    image_data = request.json['image']
    
    # # base64 형식의 이미지 데이터를 PIL Image 객체로 변환
    # image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # # 이미지 전처리 및 예측 수행
    # results = model(image)

    # # 결과 추출 및 필요한 정보 추출 (예: 클래스, 경계 상자 등)
    # predictions = results.pandas().xyxy[0].to_dict(orient='records')
    
    
    # Base64 디코딩 및 이미지 열기 시도
    try:
        image_bytes = base64.b64decode(image_data)
        # print(image_bytes)
        image = Image.open(io.BytesIO(image_bytes))
        # 이미지 전처리 및 예측 수행
        results = model(image)
        # 결과 추출 및 필요한 정보 추출 (예: 클래스, 경계 상자 등)
        predictions = results.pandas().xyxy[0].to_dict(orient='records')

    except Exception as e:
        print("Error occurred while decoding or opening the image: {e}")


    return jsonify(predictions=predictions)
if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)
