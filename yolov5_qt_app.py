#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import torch
import numpy as np
import time
import pyautogui
from pathlib import Path
from collections import deque

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QImage, QPixmap, QResizeEvent

# YOLOv5 경로 설정
MODEL_PATH = 'yolov5'
sys.path.append(MODEL_PATH)

# YOLOv5 모듈 임포트
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class DetectionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    camera_error_signal = pyqtSignal(str)
    
    def __init__(self, weights='yolov5/yolov5s.pt', device=''):
        super().__init__()
        self.weights = weights  # YOLOv5s 모델 사용
        self.device = device
        self.source = 'webcam'  # 'webcam' 또는 'screen'
        self.conf_thres = 0.5  # 신뢰도 임계값 0.25에서 0.5로 수정
        self.iou_thres = 0.45
        self.model = None
        self.cap = None
        self.is_running = True
        self.camera_open_failed = False
        self.model_name = 'YOLOv5s'  # 모델 이름 추가
        
        # 성능 최적화 관련 변수
        self.img_size = 416  # 처리 이미지 크기 (640 -> 416)
        self.process_every_n_frames = 1  # 매 프레임마다 처리 (스킵 없음)
        self.frame_count = 0
        self.fps_values = deque(maxlen=10)  # FPS 값을 평균내기 위한 큐
        self.average_fps = 0
        self.last_detection_results = None  # 마지막 탐지 결과 저장
        
        # 가속화 옵션
        if torch.cuda.is_available():
            self.half = True  # FP16 (반정밀도) 사용
        else:
            self.half = False
        
        self.load_model()
    
    def load_model(self):
        """YOLOv5 모델 로드"""
        try:
            device = select_device(self.device)
            self.model = attempt_load(self.weights, device=device)
            
            # 모델 최적화
            if self.half:
                self.model.half()  # FP16로 변환
            
            # 워밍업
            dummy_input = torch.zeros((1, 3, self.img_size, self.img_size), device=device)
            if self.half:
                dummy_input = dummy_input.half()
            self.model(dummy_input)
            
            self.device = device
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            print(f"{self.model_name} 모델 로딩 완료! ({'Half precision' if self.half else 'Full precision'})")
        except Exception as e:
            print(f"모델 로딩 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def change_source(self, source):
        """입력 소스 변경 (webcam 또는 screen)"""
        self.source = source
        if self.source == 'webcam':
            if self.cap is None or not self.cap.isOpened():
                self.open_camera()
        
        # 소스 변경 시 마지막 탐지 결과 초기화
        self.last_detection_results = None
    
    def open_camera(self):
        """카메라 열기"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        self.cap = cv2.VideoCapture(0)
        
        # 카메라 해상도 설정 - 성능 최적화
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            self.camera_error_signal.emit("카메라를 열 수 없습니다.")
            self.camera_open_failed = True
            print("카메라를 열 수 없습니다.")
        else:
            self.camera_open_failed = False
            print("카메라 연결 성공!")
    
    def capture_screen(self):
        """화면 캡처"""
        try:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 화면 캡처 크기 조정 (더 작게)
            scale_percent = 40  # 원본 크기의 40% (기존 50%에서 감소)
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            
            return frame
        except Exception as e:
            print(f"화면 캡처 오류: {e}")
            return None
    
    def detect_objects(self, img):
        """이미지에서 객체 탐지"""
        if self.model is None:
            return [], img
        
        # 텐서 값을 Python 스칼라로 변환
        stride = int(self.model.stride.max().item())
        img_size = check_img_size(self.img_size, s=stride)
        
        # 이미지 전처리
        img0 = img.copy()
        img = letterbox(img0, img_size, stride=stride, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(self.device)
        if self.half:
            img = img.half()  # FP16 변환
        else:
            img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        # 추론
        with torch.no_grad():  # 메모리 절약을 위해 gradient 계산 비활성화
            pred = self.model(img, augment=False)[0]
        
        # NMS 적용
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
        
        # 결과 처리
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                
        return pred[0] if pred else [], img0
    
    def process_frame(self, img):
        """프레임 처리 및 시각화"""
        try:
            t0 = time.time()
            self.frame_count += 1
            
            # 모든 프레임에서 객체 탐지 수행
            det, img0 = self.detect_objects(img)
            self.last_detection_results = (det, img0)
            
            # 결과 시각화 - 직접 어노테이션
            annotator = Annotator(img0, line_width=2, example=str(self.names))
            
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
            img0 = annotator.result()
            
            # FPS 계산 및 평균화 (0으로 나누기 오류 수정)
            elapsed_time = time.time() - t0
            if elapsed_time > 0:  # 0으로 나누기 방지
                fps = 1 / elapsed_time
                self.fps_values.append(fps)
                if self.fps_values:  # 큐가 비어있지 않은지 확인
                    self.average_fps = sum(self.fps_values) / len(self.fps_values)
            else:
                # elapsed_time이 0이거나 매우 작은 경우, 이전 FPS 값 재사용
                if self.fps_values:
                    fps = self.fps_values[-1]  # 가장 최근 FPS 값 사용
                else:
                    fps = 30.0  # 기본값 설정
                self.fps_values.append(fps)
                self.average_fps = fps
            
            # 화면 텍스트 표시 - 텍스트 감소로 성능 향상
            text_color = (0, 255, 0)
            font_scale = 0.5  # 텍스트 크기 감소
            
            # FPS 및 모델 표시
            cv2.putText(img0, f'FPS: {self.average_fps:.1f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
            source_text = "카메라" if self.source == 'webcam' else "화면캡처"
            cv2.putText(img0, f'{source_text} | {self.model_name}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
            
            return img0
        except Exception as e:
            print(f"프레임 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return img
    
    def stop(self):
        """스레드 종료"""
        self.is_running = False
        self.wait()
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
    
    def run(self):
        """스레드 실행"""
        if self.source == 'webcam':
            self.open_camera()
            
        while self.is_running:
            try:
                if self.source == 'webcam':
                    if not self.camera_open_failed and self.cap is not None and self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if ret:
                            processed_frame = self.process_frame(frame)
                            self.change_pixmap_signal.emit(processed_frame)
                        else:
                            # 카메라에서 프레임을 읽지 못함 - 에러 메시지를 표시하는 빈 이미지
                            height, width = 480, 640
                            error_img = np.zeros((height, width, 3), dtype=np.uint8)
                            cv2.putText(error_img, "카메라에서 프레임을 읽을 수 없습니다.", (50, height//2), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            self.change_pixmap_signal.emit(error_img)
                    else:
                        # 카메라 열기 실패 - 에러 메시지를 표시하는 빈 이미지
                        height, width = 480, 640
                        error_img = np.zeros((height, width, 3), dtype=np.uint8)
                        cv2.putText(error_img, "카메라를 찾을 수 없습니다.", (50, height//2-30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(error_img, "모드 변경 버튼을 눌러 화면 캡처를 사용하세요.", (50, height//2+30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.change_pixmap_signal.emit(error_img)
                else:
                    # 화면 캡처
                    screen = self.capture_screen()
                    if screen is not None:
                        processed_frame = self.process_frame(screen)
                        self.change_pixmap_signal.emit(processed_frame)
                
                # FPS 제한 - 너무 빠르게 처리되지 않도록
                time.sleep(0.01)  # 약 100 FPS 제한
            except Exception as e:
                print(f"스레드 실행 오류: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)  # 오류 발생 시 잠시 대기

class YoloV5App(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        
        # UI 파일 로드
        uic.loadUi('yolov5_2.ui', self)
        
        # UI 요소 연결
        self.btn_exit.clicked.connect(self.close)
        self.btn_mode.clicked.connect(self.toggle_mode)
        
        # 윈도우 크기 및 제목 설정
        self.setWindowTitle("YOLOv5 객체 인식 앱")
        self.resize(800, 600)
        
        # 기본 설정
        self.source = 'webcam'  # 기본 모드는 카메라
        
        # 탐지 스레드 설정
        self.thread = DetectionThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.camera_error_signal.connect(self.show_camera_error)
        
        # 화면 크기 조정
        self.cam_screen.setScaledContents(True)
        
        # 스레드 시작
        self.thread.start()
        
        # 성능 관련 메시지 표시
        print("성능 최적화가 적용된 YOLOv5 애플리케이션이 시작되었습니다.")
        print(f"- 이미지 처리 크기: {self.thread.img_size}px")
        print(f"- 모든 프레임 처리: 프레임 스킵 없음")
        print(f"- FP16 최적화: {'사용함' if self.thread.half else '사용안함'}")
        
    def resizeEvent(self, event: QResizeEvent):
        """창 크기가 변경될 때 UI 조정"""
        # 창 크기에 맞게 cam_screen 크기 조정
        new_width = self.width() - 40
        new_height = self.height() - 100
        self.cam_screen.setGeometry(20, 20, new_width, new_height)
        
        # 버튼 위치 조정
        self.btn_mode.setGeometry(20, self.height() - 60, 100, 30)
        self.btn_exit.setGeometry(self.width() - 120, self.height() - 60, 100, 30)
        
        super().resizeEvent(event)
        
    def toggle_mode(self):
        """카메라와 화면 캡처 모드 간 전환"""
        if self.source == 'webcam':
            self.source = 'screen'
            self.btn_mode.setText('카메라 모드')
        else:
            self.source = 'webcam'
            self.btn_mode.setText('화면 캡처 모드')
        
        self.thread.change_source(self.source)
    
    def show_camera_error(self, message):
        """카메라 오류 메시지 표시"""
        QtWidgets.QMessageBox.warning(self, "카메라 오류", message)
    
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """OpenCV 이미지를 QLabel에 업데이트"""
        try:
            # QImage로 변환
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # QLabel에 표시
            self.cam_screen.setPixmap(QPixmap.fromImage(qt_image))
        except Exception as e:
            print(f"이미지 업데이트 오류: {e}")
    
    def closeEvent(self, event):
        """앱 종료 시 스레드 정리"""
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    # 메인 스레드에서 OpenCL 비활성화 - 일부 시스템에서 성능 향상
    cv2.ocl.setUseOpenCL(False)
    
    app = QtWidgets.QApplication(sys.argv)
    window = YoloV5App()
    window.show()
    sys.exit(app.exec_()) 