import mediapipe as mp
import cv2
import time
from scipy.spatial import distance as dst

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
draw = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture("data/data/02_65_6504_311618672082.mp4")
# cap = cv2.VideoCapture(2)
# img = cv2.imread('data/data/face.jpg')
def mp_face_landmarks(image):
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    # print(results.multi_face_landmarks)
    faces = []
    visual_image = image.copy()
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            face = []
            counter = 0
            for lm in facelms.landmark:
                # print(lm)
                h, w, c = image.shape
                x, y, z = int(lm.x*w), int(lm.y*h), int(lm.z*c)
                # print(x, y, z)
                # cv2.putText(image, str(counter + 1), list[x, y],cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                face.append([x, y])
                visual_image = image.copy()
            mpDraw.draw_landmarks(visual_image, facelms, mpFaceMesh.FACE_CONNECTIONS,draw,draw)

            faces.append(face)
    return image, faces,visual_image

def EAR_Compute(faces):
    left_eye = [faces[0][34], faces[0][162], faces[0][159], faces[0][134], faces[0][154], faces[0][164]]
    right_eye = [faces[0][363], faces[0][385], faces[0][388], faces[0][264], faces[0][374], faces[0][382]]
    a = dst.euclidean(left_eye[1], left_eye[5])
    b = dst.euclidean(left_eye[2], left_eye[4])
    c = dst.euclidean(left_eye[0], left_eye[3])

    x = dst.euclidean(right_eye[1], right_eye[5])
    y = dst.euclidean(right_eye[2], right_eye[4])
    z = dst.euclidean(right_eye[0], right_eye[3])

    aspect_ratio_left = (a + b) / (2.0 * c)
    aspect_ratio_right = (x + y) / (2.0 * z)
    EAR = (aspect_ratio_right + aspect_ratio_left) / 2.0
    # print(EAR)

    return round(EAR,2), round(aspect_ratio_right,2), round(aspect_ratio_left,2)
ear_thresh = 0.3
counter_ear = 0
ear_consec_frame= 3
total = 0


while True:
    ret, frame = cap.read()
    img, faces, vi = mp_face_landmarks(frame)
    # print(len(faces))

    if (len(faces) != 0):
        ear_value, arr, arl = EAR_Compute(faces)
        # print(ear_value, arr, arl)
        left_eye = [faces[0][34], faces[0][162], faces[0][159], faces[0][134], faces[0][154], faces[0][164]]
        right_eye = [faces[0][363], faces[0][385], faces[0][388], faces[0][264], faces[0][374], faces[0][382]]
        # upper_lip = faces[0][15]
        # lower_lip = faces[0][16]
        # lip_distance = dst.euclidean(upper_lip,lower_lip)
        # print(lip_distance)
        cv2.rectangle(frame, left_eye[0], left_eye[3], (0, 255, 0), 2)
        cv2.putText(vi, f"Eyes_ratio: {str(ear_value)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
        # cv2.putText(vi, f"Mouth: {str(lip_distance)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

        upper_lip_u, upper_lip_l = faces[0][2], faces[0][14]
        lower_lip_u, lower_lip_l = faces[0][17], faces[0][19]

        upper_lip_distance = dst.euclidean(upper_lip_u, upper_lip_l)
        lower_lip_distance = dst.euclidean(lower_lip_u, lower_lip_l)
        between_lips = dst.euclidean(upper_lip_l, lower_lip_u)

        lips_average = (upper_lip_distance + lower_lip_distance) / 2
        if upper_lip_distance < between_lips:
            print("Mouth Open")
            cv2.putText(vi, f"Mouth: {str(upper_lip_distance)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vi, f"Mouth: {str(between_lips)}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print("Mouth Close")
            cv2.putText(vi, f"Mouth: {str(upper_lip_distance)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vi, f"Mouth: {str(between_lips)}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # if ear_value < ear_thresh:
    #     counter_ear += 1
    # else:
    #     if counter_ear >= ear_consec_frame:
    #         total += 1
    #     counter_ear = 0
    # cv2.putText(img, str(total), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    # cv2.putText(img, str(counter_ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # if len(faces) != 0:
    #     print(len(faces[0]))
    cv2.imshow('windows', vi)
    cv2.waitKey(1)

# ima, faces = mp_face_landmarks(img)
"""
    Analyzing Landmarks
"""
"""

counter = 0
c_x, c_y = 0, 0
c_set = [34,162,159,134,154,164,363,385,388,264,374,382]
for face in faces:
    for landmark in face:
        counter += 1
        if c_set.__contains__(counter):
            # cv2.putText(ima, str(counter), landmark,cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
            cv2.circle(ima, landmark, 5, (0,255,255), 5)
            print(landmark)
        else:
            cv2.circle(ima, landmark, 5, (0, 255, 0), 5)
            # cv2.putText(ima, str(counter), landmark, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
# print(faces[0][1])

# left_eye_upper = [faces[0][29], faces[0][27], faces[0][28]]
# left_eye_lower = [faces[0][24], faces[0][23], faces[0][22]]

# right_eye_upper = [faces[0][385], faces[0][386], faces[0][387]]
# right_eye_lower = [faces[0][380], faces[0][374], faces[0][373]]

# upper_lip = [faces[0][13]]
# lower_lip = [faces[0][17]]

# print(faces[0][34],faces[0][162],faces[0][159],faces[0][134],faces[0][154],faces[0][164],)
# print(faces[0][363],faces[0][385],faces[0][388],faces[0][264],faces[0][374],faces[0][382],)
# left_eye_status = dst.euclidean(left_eye_upper[1], left_eye_lower[1])
# right_eye_status = dst.euclidean(right_eye_lower[1], right_eye_upper[1])

# print(left_eye_status, right_eye_status)

left_eye = [faces[0][34],faces[0][162],faces[0][159],faces[0][134],faces[0][154],faces[0][164]]
right_eye = [faces[0][363],faces[0][385],faces[0][388],faces[0][264],faces[0][374],faces[0][382]]

a = dst.euclidean(left_eye[1], left_eye[5])
b = dst.euclidean(left_eye[2], left_eye[4])
c = dst.euclidean(left_eye[0], left_eye[3])

x = dst.euclidean(right_eye[1], right_eye[5])
y = dst.euclidean(right_eye[2], right_eye[4])
z = dst.euclidean(right_eye[0], right_eye[3])

aspect_ratio_left = (a + b) / (2.0 * c)
aspect_ratio_right = (x + y) / (2.0 * z)

print((aspect_ratio_right + aspect_ratio_left)/2.0)

cv2.imshow('Image',ima)
cv2.waitKey(0)
"""
