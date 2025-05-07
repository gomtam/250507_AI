#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
import time
from pathlib import Path
import sys
import argparse
import pyautogui  # 화면 캡처를 위한 라이브러리

# YOLOv5 모델 경로 설정
MODEL_PATH = 'yolov5'

# YOLOv5 모듈 추가
sys.path.append(MODEL_PATH)

# YOLOv5의 utils 모듈 임포트
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

def load_model(weights='yolov5s.pt', device=''):
    """YOLOv5 모델 로드"""
    device = select_device(device)
    model = attempt_load(weights, device=device)
    return model, device

def detect_objects(model, img, device='', conf_thres=0.25, iou_thres=0.45):
    """이미지에서 객체 탐지"""
    # 텐서 값을 Python 스칼라로 변환
    stride = int(model.stride.max().item())
    img_size = check_img_size(640, s=stride)
    
    # 이미지 전처리
    img0 = img.copy()
    img = letterbox(img0, img_size, stride=stride, auto=False)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    
    # 추론
    pred = model(img, augment=False)[0]
    
    # NMS 적용
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
    
    # 결과 처리
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            
    return pred[0] if pred else [], img0

def capture_screen():
    """화면 캡처"""
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환 (OpenCV 형식)
    return frame

def run_detection(source='webcam', weights='yolov5s.pt', device='', conf_thres=0.25, iou_thres=0.45):
    """카메라 또는 화면 캡처에서 객체 탐지 실행"""
    # 모델 로드
    model, device = load_model(weights, device)
    names = model.module.names if hasattr(model, 'module') else model.names
    
    # 입력 소스 설정
    if source == 'webcam':
        print("카메라를 사용하여 객체 탐지를 시작합니다...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return
    else:
        print("화면 캡처를 사용하여 객체 탐지를 시작합니다...")
    
    # 루프
    while True:
        if source == 'webcam':
            success, img = cap.read()
            if not success:
                print("프레임을 읽을 수 없습니다.")
                break
        else:
            try:
                # 화면 캡처
                img = capture_screen()
                success = True
            except Exception as e:
                print(f"화면 캡처 오류: {e}")
                break
            
        # 탐지 수행
        try:
            t0 = time.time()
            det, img0 = detect_objects(model, img, device, conf_thres, iou_thres)
            
            # 결과 시각화
            annotator = Annotator(img0, line_width=2, example=str(names))
            
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
            # FPS 계산
            fps = 1 / (time.time() - t0)
            img0 = annotator.result()
            
            # FPS 표시
            cv2.putText(img0, f'FPS: {fps:.1f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 소스 표시
            source_text = "카메라" if source == 'webcam' else "화면 캡처"
            cv2.putText(img0, f'소스: {source_text}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 이미지 크기 조정 (화면 캡처는 전체 화면이라 너무 클 수 있음)
            if source != 'webcam':
                scale_percent = 70  # 원본 크기의 70%
                width = int(img0.shape[1] * scale_percent / 100)
                height = int(img0.shape[0] * scale_percent / 100)
                dim = (width, height)
                img0 = cv2.resize(img0, dim, interpolation=cv2.INTER_AREA)
            
            # 이미지 표시
            cv2.imshow('YOLOv5 객체 탐지', img0)
        except Exception as e:
            print(f"객체 탐지 오류: {e}")
            import traceback
            traceback.print_exc()
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # q 키: 종료
            break
            
    # 정리
    if source == 'webcam' and 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

def parse_opt():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='webcam', choices=['webcam', 'screen'], help='입력 소스 (webcam 또는 screen)')
    parser.add_argument('--weights', type=str, default='yolov5/yolov5s.pt', help='모델 가중치 경로')
    parser.add_argument('--device', type=str, default='', help='cuda 장치, 예: 0 또는 0,1,2,3 또는 cpu')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='신뢰도 임계값')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU 임계값')
    return parser.parse_args()

def select_source():
    """사용자에게 입력 소스를 선택하도록 요청"""
    print("\n객체 인식에 사용할 입력 소스를 선택하세요:")
    print("1. 카메라")
    print("2. 화면 캡처")
    
    while True:
        choice = input("선택 (1 또는 2): ")
        if choice == '1':
            return 'webcam'
        elif choice == '2':
            return 'screen'
        else:
            print("잘못된 선택입니다. 1 또는 2를 입력하세요.")

if __name__ == '__main__':
    # 명령행 인수 파싱
    opt = parse_opt()
    
    # 명령행 인수로 소스가 지정되지 않았다면 사용자에게 선택하도록 요청
    if len(sys.argv) == 1:
        opt.source = select_source()
    
    print(f"\nYOLOv5 객체 탐지 프로그램이 시작되었습니다. (소스: {opt.source})")
    print("- 종료하려면 'q' 키를 누르세요.")
    
    # 객체 탐지 실행
    run_detection(
        source=opt.source,
        weights=opt.weights,
        device=opt.device,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres
    ) 