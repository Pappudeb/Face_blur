import cv2
import mediapipe as mp
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

detector = FaceDetector(minDetectionCon=0.80)

while True:
    ret, img = cap.read()
    img, bboxs = detector.findFaces(img, draw=False)


    if  bboxs:
        for i, bbox in enumerate(bboxs):
            x, y, h, w = bbox['bbox']
            imgCrop = img[y:y+h, x:x+w]
            imgBlur = cv2.blur(imgCrop,(37,37))
            img[y:y + h, x:x + w] = imgBlur
            #cv2.imshow(f'Image Cropped{i}', imgCrop)

    cv2.imshow('image',img)
    cv2.waitKey(1)
