"""
Object detection with pre developed python package
is working
"""
import cv2
import matplotlib.pyplot as plt
import cvlib as cvx
from cvlib.object_detection import draw_bbox
from scipy.spatial import distance as dist

def midpoint(a1, a2, b1, b2):
    return (int((a1 + b1)/2), int((a2 + b2)/2))

cap = cv2.VideoCapture("data/data/traffic.MOV")
while True:
    ret, frame = cap.read()
    im = frame
    bbox, label, conf = cvx.detect_common_objects(im)
    output_image = draw_bbox(im, bbox, label, conf)
    mids = []
    counter = 0
    width = frame.shape[1]
    height = frame.shape[0]

    center_carhood = (round(width/2), height)
    cv2.circle(output_image, center_carhood, radius=1, color=(0, 255, 0), thickness=1, )
    # print(center_carhood)
    for item in bbox:
        x1, x2, x3, x4 = item
        mid = midpoint(x1, x2, x3, x4)
        # print(mid)
        mids.append(mid)
        # cv2.putText(frame, f"{counter + 1}", mid, cv2.FONT_HERSHEY_PLAIN, 1,
        #             (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.line(output_image,mid,center_carhood,(0,255,0),1)
        output_image = cv2.circle(output_image, mid, radius= 1, color=(0, 255, 0), thickness=1, )
        D = ((dist.euclidean(mid,center_carhood)) * 0.0002645833)*100
        print(D)
        cv2.putText(output_image, "{:.1f}m".format(D), mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
    cv2.imshow('frame', output_image)
    if cv2.waitKey(1) & 0xff == 'q':
        break
cap.release()
cv2.destroyAllWindows()