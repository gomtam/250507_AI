# YOLOv5 객체 인식 PyQt5 애플리케이션

PyQt5와 YOLOv5를 활용한 실시간 객체 인식 애플리케이션입니다. 카메라 또는 화면 캡처를 통해 객체를 감지할 수 있습니다.

![애플리케이션 스크린샷](screenshot.png)

## 주요 기능

- 카메라 및 화면 캡처 모드 지원
- 실시간 객체 감지 (COCO 80개 클래스)
- FPS(초당 프레임) 표시
- 모드 전환 및 종료 기능
- 신뢰도 0.5 이상인 객체만 표시
- 성능 최적화

## 필요 환경

- Python 3.8 이상
- PyQt5
- PyTorch
- OpenCV
- pyautogui
- YOLOv5

## 설치 방법

1. 이 저장소를 클론합니다:
```bash
git clone https://github.com/[사용자명]/yolov5-object-detection-app.git
cd yolov5-object-detection-app
```

2. YOLOv5 저장소를 클론합니다 (서브모듈로 추가하는 것이 좋습니다):
```bash
git clone https://github.com/ultralytics/yolov5.git
```

3. 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

4. YOLOv5 모델 파일을 다운로드합니다:
```bash
cd yolov5
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
cd ..
```

## 사용 방법

1. 애플리케이션 실행:
```bash
python yolov5_qt_app.py
```

2. UI 컨트롤:
   - **모드 변경 버튼**: 카메라와 화면 캡처 모드 간 전환
   - **종료 버튼**: 프로그램 종료

## 코드 구조

- `yolov5_qt_app.py`: 메인 애플리케이션 코드
- `yolov5_2.ui`: Qt Designer로 만든 UI 파일
- `requirements.txt`: 필요한 패키지 목록
- `.gitignore`: Git에서 제외할 파일 목록

## 성능 최적화

성능 최적화를 위한 여러 기법이 적용되어 있습니다:
- 이미지 처리 크기 조정 (416px)
- FP16(반정밀도) 최적화 (CUDA GPU 지원시)
- OpenCL 비활성화
- 화면 텍스트 최소화

## 라이선스

이 프로젝트는 MIT 라이선스로 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 참고

- YOLOv5 원본 저장소: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- 이 프로젝트는 교육 및 학습 목적으로 만들어졌습니다. 