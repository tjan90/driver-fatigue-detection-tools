import cv2
from deepface import DeepFace
from PIL import Image

"""
 Only for one Image
"""
# img = 'data/data/db/TJ/01.jpg'
# img = cv2.imread(img)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY,)
# face = DeepFace.detectFace(img_path=img)
# cv2.imwrite(f"data/data/db/TJ/{}.jpg", face)
# cv2.imshow('face', face)
# cv2.waitKey()


"""
  From video stream
"""
cap = cv2.VideoCapture('data/data/02_65_6504_311618672082.mp4')
# cap = cv2.VideoCapture(2)
counter = 0
while True:
    counter += 1
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    print("converting to Gray")
    face = DeepFace.detectFace(img_path=frame)
    print("detecting face...")
    cv2.imshow('face', face)
    cv2.waitKey()
    cv2.imwrite("data/data/db/Jiannan/"+str(counter)+".png", face * 255)
    print(f"writing Image {counter}...")
    if cv2.waitKey(1) & 0xFF == 'q':
        break

cap.release()
cv2.destroyAllWindows()

#
# cap = cv2.VideoCapture(2)
# while True:
#     ret, frame = cap.read()
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(0) & ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()