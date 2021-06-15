import cv2
import mediapipe as mp
from scipy.spatial import distance as dst

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
draw = mpDraw.DrawingSpec(thickness=3, circle_radius=3)

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
                if counter > 0 and counter < 460:
                    cv2.putText(image, str(counter), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    # cv2.circle(image, (x,y),2,(0,255,0),2)
                counter +=1
                face.append([x, y])
                visual_image = image.copy()
            mpDraw.draw_landmarks(visual_image, facelms, mpFaceMesh.FACE_CONNECTIONS,draw,draw)

            faces.append(face)
    return image, faces, visual_image

image = cv2.imread("data/data/face.jpg")
lm, faces, vi = mp_face_landmarks(image)
counter = 1
for face in faces:
    for landmark in face:
        counter += 1
        cv2.putText(vi, str(counter), landmark,cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
upper_lip_u, upper_lip_l = faces[0][13], faces[0][15]
lower_lip_u, lower_lip_l = faces[0][17],faces[0][19]

upper_lip_distance = dst.euclidean(upper_lip_u, upper_lip_l)
lower_lip_distance = dst.euclidean(lower_lip_u, lower_lip_l)
between_lips = dst.euclidean(upper_lip_l, lower_lip_u)

lips_average = (upper_lip_distance + lower_lip_distance )/ 2
if lips_average < between_lips:
    print(f"Mouth Open - {lower_lip_distance, between_lips, upper_lip_distance}")

else:
    print(f"Mouth close - {lower_lip_distance, between_lips, upper_lip_distance}")

cv2.imshow('features', lm)
cv2.waitKey(0)
