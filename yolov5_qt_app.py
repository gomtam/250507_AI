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
try:
    from models.experimental import attempt_load
    from utils.general import check_img_size, non_max_suppression, scale_boxes
    from utils.plots import Annotator, colors
    from utils.torch_utils import select_device
    from utils.augmentations import letterbox
except ModuleNotFoundError:
    # PyInstaller 실행 환경에서 경로 재설정
    import os
    yolo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov5")
    sys.path.insert(0, yolo_path)
    
    # 직접 경로 참조로 임포트
    from yolov5.models.experimental import attempt_load
    from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
    from yolov5.utils.plots import Annotator, colors
    from yolov5.utils.torch_utils import select_device
    from yolov5.utils.augmentations import letterbox

class DetectionThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.running = True
        self.cap = None
        self.process_every_n_frames = 2  # 2프레임마다 처리
        self.frame_count = 0
        self.last_fps = 0
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.last_frame = None  # 마지막 처리된 프레임 저장

    def run(self):
        try:
            # 카메라 초기화
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.error_occurred.emit("카메라를 열 수 없습니다.")
                return

            # 카메라 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.error_occurred.emit("프레임을 읽을 수 없습니다.")
                    break

                # FPS 계산
                self.fps_counter += 1
                if time.time() - self.fps_start_time >= 1.0:
                    self.last_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = time.time()

                # 프레임 스킵 처리
                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames == 0:
                    # 프레임 처리
                    processed_frame = self.process_frame(frame)
                    self.last_frame = processed_frame
                elif self.last_frame is not None:
                    # 스킵된 프레임에는 이전 결과 재사용
                    processed_frame = self.last_frame.copy()
                else:
                    processed_frame = frame

                self.frame_ready.emit(processed_frame)

                # CPU 사용량 제한
                time.sleep(0.01)

        except Exception as e:
            self.error_occurred.emit(f"프레임 처리 중 오류 발생: {str(e)}")
        finally:
            if self.cap is not None:
                self.cap.release()

    def process_frame(self, frame):
        try:
            # 이미지 크기 조정 (더 작게)
            img = cv2.resize(frame, (320, 320))  # 416x416에서 320x320으로 축소
            
            # YOLOv5 추론
            with torch.no_grad():  # 메모리 사용량 최적화
                results = self.model(img)
            
            # 결과 시각화
            annotated_frame = results.render()[0]
            
            # 원본 크기로 복원
            annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))
            
            # FPS 표시 (텍스트 크기 축소)
            cv2.putText(annotated_frame, f'FPS: {self.last_fps}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            
            return annotated_frame
            
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {str(e)}")
            return frame

    def stop(self):
        self.running = False
        self.wait()

class YOLOv5App(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi('yolov5_2.ui', self)
        
        # 창 크기 설정
        self.resize(1280, 720)  # 1280x720 해상도로 설정
        
        # UI 요소 초기화
        self.image_label = self.findChild(QtWidgets.QLabel, 'cam_screen')
        if self.image_label is None:
            QtWidgets.QMessageBox.critical(self, "오류", "UI 요소를 찾을 수 없습니다. (cam_screen)")
            sys.exit(1)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # 종료 버튼 연결
        self.exit_button = self.findChild(QtWidgets.QPushButton, 'btn_exit')
        if self.exit_button is None:
            QtWidgets.QMessageBox.critical(self, "오류", "UI 요소를 찾을 수 없습니다. (btn_exit)")
            sys.exit(1)
        self.exit_button.clicked.connect(self.close)
        
        # YOLOv5 모델 로드
        self.model = None
        self.load_model()
        
        # 스레드 초기화
        self.detection_thread = None
        
        # 창 크기 조정 이벤트 연결
        self.resizeEvent = self.on_resize
        
        # 초기 이미지 표시
        self.show_initial_image()
        
        # 카메라 시작
        self.start_camera()

    def load_model(self):
        try:
            # 더 가벼운 모델 사용 (yolov5n.pt)
            self.model = torch.hub.load('yolov5', 'custom', path='yolov5/yolov5n.pt', source='local')
            self.model.conf = 0.5  # 신뢰도 임계값을 0.5로 상향 조정
            self.model.iou = 0.45   # IOU 임계값
            if torch.cuda.is_available():
                self.model.cuda()
                self.model.half()  # FP16 사용
                torch.cuda.empty_cache()  # GPU 메모리 정리
            self.model.eval()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "오류", f"모델 로드 중 오류 발생: {str(e)}")
            sys.exit(1)

    def start_camera(self):
        if self.detection_thread is None or not self.detection_thread.isRunning():
            self.detection_thread = DetectionThread(self.model)
            self.detection_thread.frame_ready.connect(self.update_frame)
            self.detection_thread.error_occurred.connect(self.handle_error)
            self.detection_thread.start()

    def update_frame(self, frame):
        try:
            # OpenCV BGR -> RGB 변환
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # QImage 생성
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # QLabel 크기에 맞게 이미지 크기 조정
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # 이미지 표시
            self.image_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"프레임 업데이트 중 오류 발생: {str(e)}")

    def handle_error(self, error_msg):
        QtWidgets.QMessageBox.critical(self, "오류", error_msg)
        self.close()

    def show_initial_image(self):
        # 초기 이미지 생성 (검은색 배경)
        initial_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.update_frame(initial_image)

    def on_resize(self, event):
        if hasattr(self, 'image_label') and self.image_label.pixmap():
            scaled_pixmap = self.image_label.pixmap().scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        super().resizeEvent(event)

    def closeEvent(self, event):
        if self.detection_thread is not None:
            self.detection_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = YOLOv5App()
    window.show()
    sys.exit(app.exec_()) 