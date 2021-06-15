"""
Mouth detection when its open
is working
"""
# import cv2
# import numpy as np
#
# mouth_cascade = cv2.CascadeClassifier('data/haarcascade_mcs_mouth.xml')
#
# if mouth_cascade.empty():
#   raise IOError('Unable to load the mouth cascade classifier xml file')
#
# cap = cv2.VideoCapture('data/data/02_65_6504_3116186720821.mp4')
# ds_factor = 0.5
#
# while True:
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
#     for (x,y,w,h) in mouth_rects:
#         y = int(y - 0.15*h)
#         cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
#
#         break
#
#     cv2.imshow('Mouth Detector', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import dlib

path = "models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(path)
detector = dlib.get_frontal_face_detector()


def get_landmarks(im):
    rects = detector(im, 1)  # image and no.of rectangles to be drawn
    if len(rects) > 1:
        print("Toomanyfaces")
        return np.matrix([0])
    if len(rects) == 0:
        print("Toofewfaces")
        return np.matrix([0])
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def place_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.3, color=(0, 255, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def upper_lip(landmarks):
    top_lip = []
    for i in range(50, 53):
        top_lip.append(landmarks[i])
    for j in range(61, 64):
        top_lip.append(landmarks[j])
    top_lip_point = (np.squeeze(np.asarray(top_lip)))
    top_mean = np.mean(top_lip_point, axis=0)
    return int(top_mean[1])


def low_lip(landmarks):
    lower_lip = []
    for i in range(65, 68):
        lower_lip.append(landmarks[i])
    for j in range(56, 59):
        lower_lip.append(landmarks[j])
    lower_lip_point = (np.squeeze(np.asarray(lower_lip)))
    lower_mean = np.mean(lower_lip_point, axis=0)

    return int(lower_mean[1])


def decision(image):
    landmarks = get_landmarks(image)
    if (landmarks.all() == [0]):
        return -10  # Dummy value to prevent error
    top_lip = upper_lip(landmarks)
    lower_lip = low_lip(landmarks)
    distance = abs(top_lip - lower_lip)
    return distance


cap = cv2.VideoCapture('data/data/02_65_6504_3116186720821.mp4')
yawns = 0
while (True):
    ret, frame = cap.read()
    if (ret == True):

        landmarks = get_landmarks(frame)
        if (landmarks.all() != [0]):
            l1 = []
            for k in range(48, 60):
                l1.append(landmarks[k])
            l2 = np.asarray(l1)
            lips = cv2.convexHull(l2)
            cv2.drawContours(frame, [lips], -1, (0, 255, 0), 1)

        distance = decision(frame)
        if (distance > 21):  # Use distance according to your convenience
            yawns = yawns + 1

        cv2.putText(frame, "Yawn Count: " + str(yawns), (50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                    color=(0, 255, 255))
        cv2.imshow("Subject Yawn Count", frame)
        if cv2.waitKey(1) == 13:
            break
    else:
        continue
cap.release()
cv2.destroyAllWindows()
