from deepface import DeepFace
import pandas as pd
import cv2
import os
import glob
#
# def face_recognition(image):
#     face = DeepFace.detectFace(image)
#     results = DeepFace.find(img_path=face, db_path='data/data/db', enforce_detection=False)
#     return results
#
#
# cap = cv2.VideoCapture('data/data/02_65_6504_311618672082.mp4')
# while True:
#     ret, frame = cap.read()
#     r = face_recognition(frame)
#     print(r)
#     if cv2.waitKey(0) and ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

dir = 'data/data/db'
image_list = []
for filename in glob.glob('data/data/db/*.jpg'):
    image_list.append(filename)
    print(filename)