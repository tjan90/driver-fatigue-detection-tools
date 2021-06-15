"""
This file will contain all the code divided into functions
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import tensorflow
import cvlib as cvx
from pygame import mixer
from cvlib.object_detection import draw_bbox
import dlib
from PIL import Image
from scipy.misc import imresize
from skimage.transform import resize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model
from imutils import face_utils
import face_recognition
from scipy.spatial import distance as dist
from drawFace import draw
import reference_world as world
from deepface import DeepFace
import pandas as pd
# import face_landmarks_mp
import objectDistance
import Compare_faces
import itertools

"""
Functions 
"""
class functionalityDrowsiness:
    mixer.init()
    sound = mixer.Sound('alarm.wav')
    aspect_ratio_counter = 0
    frames_closed_eyes = 0
    mouth_status_prev=None

    def __init__(self):
        # self.frame = frame
        self.face = cv2.CascadeClassifier(
            '/Users/tanveerjan/PycharmProjects/Drowsiness_detection/haar_cascade_files/haarcascade_frontalface_alt.xml')
        self.leye = cv2.CascadeClassifier(
            '/Users/tanveerjan/PycharmProjects/Drowsiness_detection/haar_cascade_files/haarcascade_lefteye_2splits.xml')
        self.reye = cv2.CascadeClassifier(
            '/Users/tanveerjan/PycharmProjects/Drowsiness_detection/haar_cascade_files/haarcascade_righteye_2splits.xml')
        self.mouth_cascade = cv2.CascadeClassifier('data/haarcascade_mcs_mouth.xml')
        self.lbl = ['Close', 'Open']
        self.model = load_model('models/cnncat2.h5')
        self.path = os.getcwd()
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.count = 0
        self.score = 0
        self.thicc = 2
        self.rpred = [99]
        self.lpred = [99]
        self.IMG_SIZE = (34, 26)
        self.ds_factor = 0.5

    def drowsinessDetection(self, frame):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = self.leye.detectMultiScale(gray)
        right_eye = self.reye.detectMultiScale(gray)
        mouth_rects = self.mouth_cascade.detectMultiScale(gray, 1.7, 11)
        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        if len(mouth_rects) > 0 and len(right_eye) > 0 and len(left_eye) > 0:
            status = 'Sleepy'

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        for (x, y, w, h) in mouth_rects:
            y = int(y - 0.15 * h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
            self.count = self.count + 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = self.model.predict_classes(r_eye)
            if (rpred[0] == 1):
                lbl = 'Open'
            if (rpred[0] == 0):
                lbl = 'Closed'
            break

        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y + h, x:x + w]
            self.count = self.count + 1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred = self.model.predict_classes(l_eye)
            if (lpred[0] == 1):
                lbl = 'Open'
            if (lpred[0] == 0):
                lbl = 'Closed'
            break
        # print(f"lpred: {lpred}.... rpred: {rpred}")
        if(rpred[0] == 0 and lpred[0] == 0):
            self.score = self.score + 1
            cv2.putText(frame, "Closed", (10, height - 20), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # if(rpred[0]==1 or lpred[0]==1):
        else:
            self.score = self.score - 1
            cv2.putText(frame, "Open", (10, height - 20), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if (self.score < 0):
            self.score = 0
        cv2.putText(frame, 'Score:' + str(self.score), (100, height - 20), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if (self.score > 30):
            # person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(self.path, 'image.jpg'), frame)
            try:
                self.sound.play()

            except:  # isplaying = False
                pass
            if (self.thicc < 16):
                self.thicc = self.thicc + 2
            else:
                self.thicc = self.thicc - 2
                if (self.thicc < 2):
                    self.thicc = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), self.thicc)
            print('returning frame')
        return frame

    def crop_eye(self, img, eye_points):
        x1, y1 = np.amin(eye_points, axis=0)
        x2, y2 = np.amax(eye_points, axis=0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        w = (x2 - x1) * 1.2
        h = w * self.IMG_SIZE[1] / self.IMG_SIZE[0]

        margin_x, margin_y = w / 2, h / 2

        min_x, min_y = int(cx - margin_x), int(cy - margin_y)
        max_x, max_y = int(cx + margin_x), int(cy + margin_y)

        eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

        eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

        return eye_img, eye_rect

    def eyes_detection(self, frame, model, shape_predictor):

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor)
        model = load_model(model)

        img_ori = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
        img = img_ori.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            shapes = predictor(gray, face)
            shapes = face_utils.shape_to_np(shapes)

            eye_img_l, eye_rect_l = self.crop_eye(gray, eye_points=shapes[36:42])
            eye_img_r, eye_rect_r = self.crop_eye(gray, eye_points=shapes[42:48])

            eye_img_l = cv2.resize(eye_img_l, dsize=self.IMG_SIZE)
            eye_img_r = cv2.resize(eye_img_r, dsize=self.IMG_SIZE)
            eye_img_r = cv2.flip(eye_img_r, flipCode=1)

            # cv2.imshow('l', eye_img_l)
            # cv2.imshow('r', eye_img_r)

            eye_input_l = eye_img_l.copy().reshape((1, self.IMG_SIZE[1], self.IMG_SIZE[0], 1)).astype(np.float32) / 255.
            eye_input_r = eye_img_r.copy().reshape((1, self.IMG_SIZE[1], self.IMG_SIZE[0], 1)).astype(np.float32) / 255.

            pred_l = model.predict(eye_input_l)
            pred_r = model.predict(eye_input_r)

            # visualize
            state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
            state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

            state_l = state_l % pred_l
            state_r = state_r % pred_r

            cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255),
                          thickness=1)
            cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255),
                          thickness=1)

            cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
            cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        return img

    def mouth_detection(self, frame):
        frame = cv2.resize(frame, None, fx=self.ds_factor, fy=self.ds_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mouth_rects = self.mouth_cascade.detectMultiScale(gray, 1.7, 11)
        for (x, y, w, h) in mouth_rects:
            y = int(y - 0.15 * h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            break
        return frame

    def face_landmarks(self, frame):
        landmarks = face_recognition.face_landmarks(frame)
        counter = 0
        counter_bl = 0
        counter_tl = 0

        print(landmarks)
        try:
            chin = landmarks[0].get('chin')
            left_eyebrow = landmarks[0].get('left_eyebrow')
            left_eye = landmarks[0].get('left_eye')
            right_eyebrow = landmarks[0].get('right_eyebrow')
            right_eye = landmarks[0].get('right_eye')
            nose_bridge = landmarks[0].get('nose_bridge')
            nose_tip = landmarks[0].get('nose_tip')
            top_lip = landmarks[0].get('top_lip')
            bottom_lip = landmarks[0].get('bottom_lip')

            for points in chin:
                cv2.circle(frame, points, 1, (0, 0, 255), -1)
            for points in left_eye:
                counter+=1
                cv2.circle(frame, points, 1, (0, 0, 255), -1)
                # cv2.putText(frame, str(counter), points, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1,
                #             cv2.LINE_AA)
            for points in left_eyebrow:
                cv2.circle(frame, points, 1, (0, 0, 255), -1)
            for points in right_eye:
                cv2.circle(frame, points, 1, (0, 0, 255), -1)
            for points in right_eyebrow:
                cv2.circle(frame, points, 1, (0, 0, 255), -1)
            for points in top_lip:
                counter_tl += 1
                cv2.circle(frame, points, 1, (0, 0, 255), -1)
                # cv2.putText(frame, str(counter_tl), points, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1,
                #             cv2.LINE_AA)
            for points in bottom_lip:
                counter_bl += 1
                cv2.circle(frame, points, 1, (0, 0, 255), -1)
                # cv2.putText(frame, str(counter_bl), points, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1,
                #             cv2.LINE_AA)
            for points in nose_bridge:
                cv2.circle(frame, points, 1, (0, 0, 255), -1)
            for points in nose_tip:
                cv2.circle(frame, points, 1, (0, 0, 255), -1)
            """
            Eyes Open/Close
            """
            aspect_ratio = 0.30

            a = dist.euclidean(left_eye[1], left_eye[5])
            b = dist.euclidean(left_eye[2], left_eye[4])
            c = dist.euclidean(left_eye[0], left_eye[3])

            x = dist.euclidean(right_eye[1], right_eye[5])
            y = dist.euclidean(right_eye[2], right_eye[4])
            z = dist.euclidean(right_eye[0], right_eye[3])

            aspect_ratio_left = (a + b) / (2.0 * c)
            aspect_ratio_right = (x + y) / (2.0 * z)
            cv2.putText(frame, f"Left EAR: {aspect_ratio_left}", (5, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, f"Right EAR: {aspect_ratio_right}", (5, 55), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1,
                        cv2.LINE_AA)
            # if aspect_ratio < aspect_ratio_right:
            #     self.aspect_ratio_counter += 1
            # else:
            #     self.aspect_ratio_counter = 0
            # cv2.putText(frame, f"EAR Counter: {self.aspect_ratio_counter}", (5, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0),
            #             1, cv2.LINE_AA)

            """Frame with closed Eyes"""
            if aspect_ratio_left < 0.25 and aspect_ratio_right < 0.25:
                self.frames_closed_eyes += 1
                cv2.putText(frame, f"Frames_closed: {self.frames_closed_eyes}", (5, 70), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Eyes: Closed", (5, 25), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0), 1, cv2.LINE_AA)

                time = datetime.now()
                with open("datafile.txt", "a") as f:
                    f.write(f"{time} : eyes_closed\n")
            else:
                cv2.putText(frame, f"Eyes: Open", (5, 25), cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Frames_closed: {self.frames_closed_eyes}", (5, 70), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0), 1, cv2.LINE_AA)
            """
            Blink rate calculation for PERCLOS
            """


            """Mouth Open/Close"""
            aspect_ratio_mouth = 55
            mouth_status = None
            p = dist.euclidean(top_lip[3], bottom_lip[5])
            q = dist.euclidean(top_lip[5], bottom_lip[3])

            average = p + q / 2
            cv2.putText(frame, f"mouth_average: {str(average)}", (5, 85), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 255, 0),1, cv2.LINE_AA)
            if average < aspect_ratio_mouth:
                cv2.putText(frame, "Mouth: Close", (5, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0), 1, cv2.LINE_AA)
                mouth_status = 1
            elif average > aspect_ratio_mouth:

                time = datetime.now()
                with open("datafile.txt", "a") as f:
                    f.write(f"{time} : Mouth_open\n")

                cv2.putText(frame, "Mouth: Open", (5, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0), 1, cv2.LINE_AA)
                mouth_status = 0

            if self.mouth_status_prev == mouth_status:
                print()


            self.mouth_status_prev = mouth_status

            """
            Other Calculations
            """
            # Average Blink Rate
            # cv2.putText(frame, f"BPM : {counter} >= 12 to 19", (5, 75), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)

            # Yawns Per Minute
            # cv2.putText(frame, f"Yawns p/m : {counter} ", (5, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)


        except Exception:
            print('No landmarks Detected!')
        # for points in chin:
        #     cv2.circle(frame, points, 5, (0,0,255), -1)
        # for points in left_eye:
        #     cv2.circle(frame, points, 5, (0,0,255), -1)
        # for points in left_eyebrow:
        #     cv2.circle(frame, points, 5, (0,0,255), -1)
        # for points in right_eye:
        #     cv2.circle(frame, points, 5, (0,0,255), -1)
        # for points in right_eyebrow:
        #     cv2.circle(frame, points, 5, (0,0,255), -1)
        # for points in top_lip:
        #     cv2.circle(frame, points, 5, (0,0,255), -1)
        # for points in bottom_lip:
        #     cv2.circle(frame, points, 5, (0,0,255), -1)
        # for points in nose_bridge:
        #     cv2.circle(frame, points, 5, (0,0,255), -1)
        # for points in nose_tip:
        #     cv2.circle(frame, points, 5, (0,0,255), -1)

        # for points in landmarks:
        #     print(f"points : {points}")
        #     cv2.circle(frame, subpoints, 5, (0,0,255), -1)


        # print(f"Left Eye Aspect Ration: {aspect_ratio_left}")
        return frame

    def headpose(self, img):
        face3Dmodel = world.ref3DModel()
        faces = face_recognition.face_locations(img, model="cnn")
        predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        for face in faces:

            # Extracting the co cordinates to convert them into dlib rectangle object
            x = int(face[3])
            y = int(face[0])
            w = int(abs(face[1] - x))
            h = int(abs(face[2] - y))
            u = int(face[1])
            v = int(face[2])

            newrect = dlib.rectangle(x, y, u, v)
            cv2.rectangle(img, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), newrect)

            draw(img, shape)

            refImgPts = world.ref2dImagePoints(shape)

            height, width, channels = img.shape
            # focalLength = args["focal"] * width
            focalLength = 1 * width
            cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

            mdists = np.zeros((4, 1), dtype=np.float64)

            # calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv2.solvePnP(
                face3Dmodel, refImgPts, cameraMatrix, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

            #  draw nose line
            p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
            cv2.line(img, p1, p2, (110, 220, 0),
                     thickness=2, lineType=cv2.LINE_AA)

            # calculating euler angles
            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            print('*' * 80)
            # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
            x = np.arctan2(Qx[2][1], Qx[2][2])
            y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1]) + (Qy[2][2] * Qy[2][2])))
            z = np.arctan2(Qz[0][0], Qz[1][0])
            # print("ThetaX: ", x)
            print("ThetaY: ", y)
            # print("ThetaZ: ", z)
            print('*' * 80)
            if angles[1] < -15:
                GAZE = "Looking: Left"
            elif angles[1] > 15:
                GAZE = "Looking: Right"
            else:
                GAZE = "Forward"

        return img, GAZE

    def face_recognition(self, image):
        face = DeepFace.detectFace(image)
        result = DeepFace.find(img_path=face, db_path='data/data/db')
        return result

    def facial_comparison(self, images):
        for (img1, img2) in itertools.combinations(images, 2):
            # print(img1.shape, img2.shape)
            matrix_distance_1 = Compare_faces.getRep(img1)
            matrix_distance_2 = Compare_faces.getRep(img2)

            print("Comparing {} with {}.".format(img1, img2))
            # print("  + Squared l2 distance between representations: {:0.3f}".format(np.dot(d, d)))
            print(type(matrix_distance_1), type(matrix_distance_2))

class frontCameraFeatures:

    def objectDetection(self, frame):
        bbox, label, conf = cvx.detect_common_objects(frame, model='yolov4')

        outputImage = draw_bbox(frame, bbox, label, conf)
        return outputImage, label

    def midpoint(a1, a2, b1, b2):
        return (int((a1 + b1) / 2), int((a2 + b2) / 2))

    def object_distance(self, image):
        bbox, label, conf = cvx.detect_common_objects(frame, model='yolov4')
        image = draw_bbox(frame, bbox, label, conf)
        center = (round(image.shape[0]/2), image.shape[1])
        cv2.circle(image, center, 2, (0,255,0),1)
        for item in bbox:
            x1, x2, x3, x4 = item
            mid = frontCameraFeatures.midpoint(x1, x2, x3, x4)
            output_image = cv2.circle(image, mid, radius=1, color=(0, 255, 0), thickness=1, )
            D = ((dist.euclidean(mid, center)) * 0.0002645833) * 100
            print(D)
            cv2.putText(image, "{:.1f}m".format(D), mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return image
    def traffic_light_color(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # lower mask (0-10)
        lower_red = np.array([0, 70, 50])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(image, lower_red, upper_red)

        # upper mask (170-180)
        lower_red1 = np.array([170, 70, 50])
        upper_red1 = np.array([180, 255, 255])
        mask1 = cv2.inRange(image, lower_red1, upper_red1)

        # defining the Range of yellow color
        lower_yellow = np.array([21, 39, 64])
        upper_yellow = np.array([40, 255, 255])
        mask2 = cv2.inRange(image, lower_yellow, upper_yellow)

        # red pixels' mask
        mask = mask0 + mask1 + mask2

        # Compare the percentage of red values
        # rate = np.count_nonzero(mask) / (desired_dim[0] * desired_dim[1])
        #
        # if rate > Threshold:
        #     return True
        # else:
        #     return False

    def draw_lines(self, img, lines):
        img = np.copy(img)
        blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

        # Exception Handling if no lines are detected
        try:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except TypeError:
            print("Type Error Triggered due to not lines detected")

        img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
        return img

    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        # channel_count = img.shape[2]
        match_mask_color = (255)
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def lanesDetection(self, img):
        # img = cv.imread("./img/road.jpg")
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # print(img.shape)
        height = img.shape[0]
        width = img.shape[1]

        region_of_interest_vertices = [
            (200, height), (width / 2, height / 1.37), (width - 300, height)
        ]
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray_img, 50, 100, apertureSize=3)
        cropped_image = self.region_of_interest(
            edge, np.array([region_of_interest_vertices], np.int32))

        lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180,
                               threshold=50, lines=np.array([]), minLineLength=10, maxLineGap=30)
        image_with_lines = self.draw_lines(img, lines)
        # plt.imshow(image_with_lines)
        # plt.show()
        return image_with_lines

class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """

    # Get image ready for feeding into model
    size = image.shape
    small_img = imresize(image, (80, 160,3))
    print(small_img.shape)
    print(type(small_img))
    # small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model_lane.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = imresize(lane_drawn, size)
    print(type(lane_image))
    print(lane_image.shape)

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result


"""
Running Code
"""
# Video File
# cap = cv2.VideoCapture("data/data/02_65_6504_311618672082.mp4")
# cap = cv2.VideoCapture("data/data/traffic-trimmed.mp4")

# From Webcam
cap = cv2.VideoCapture(2)
driver_facing = functionalityDrowsiness()
model = 'models/2018_12_17_22_58_35.h5'
model_lane = load_model('models/full_CNN_model.h5')
shape_predictor = 'models/shape_predictor_68_face_landmarks.dat'
front_facing = frontCameraFeatures()
lanes = Lanes()
counter = 0
start_time = datetime.now()

while True:
    ret, frame = cap.read()
    counter += 1
    print(counter)

    # Face Landmarks
    outputImage = driver_facing.face_landmarks(frame)
    cv2.imshow("Driver Facing Features", outputImage)


    # outputImage = driver_facing.eyes_detection(frame, model, shape_predictor)
    # cv2.imshow("Eyes Detection", outputImage)

    # Lane Detection witout DL
    # img = front_facing.lanesDetection(frame)
    # cv2.imshow("Lane Detection", img)

    # Lane detection with DL
    # img = road_lines(frame)
    # objects, labels = front_facing.objectDetection(img)
    # cv2.imshow("lane Detection", img)

    # Objects with distance
    # output = front_facing.object_distance(frame)
    # cv2.imshow("objects distance detection", frame)


    # Head pose
    # landmarks = driver_facing.face_landmarks(frame)
    # landmarks, Gaze = driver_facing.headpose(frame)
    # cv2.putText(landmarks, Gaze, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
    # cv2.imshow('Fatigue Detection', landmarks)

    # facial Landmark 468
    # face = face_landmarks_mp.mp_face_landmarks(frame)
    # cv2.imshow('window', face)

    # Facial Comparison

    # path = 'data/data/db/Jiannan/1.png'
    # db_image = cv2.imread(path)
    # faces_compare = [frame, path]
    # driver_facing.facial_comparison(faces_compare)

    # print(landmarks)
    # Timestamp
    # cv2.putText(frame, f"time : {datetime.datetime.now()}", (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)

    # cv2.imshow('Fatigue Detection', landmarks)
    cv2.waitKey(1)
