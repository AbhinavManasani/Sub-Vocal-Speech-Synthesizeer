from sub_vocal.preprocess.detect import LipDetector
import cv2, numpy as np

d = LipDetector()
print('backend:', d._backend)

cap = cv2.VideoCapture('data/demo_video.mp4')
ret, frame = cap.read()
cap.release()

import cv2
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print('frame mean:', gray.mean(), 'std:', gray.std())
