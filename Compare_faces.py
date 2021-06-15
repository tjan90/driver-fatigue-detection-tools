import time
import torch as torch
from data.openface import openface
import torch
import argparse
import cv2
import itertools
import os
import numpy as np
np.set_printoptions(precision=2)



modelDir = 'data/openface/models'
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

start = time.time()
align = openface.AlignDlib(os.path.join(dlibModelDir, 'shape_predictor_68_face_landmarks.dat'))
net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), 96)
def getRep(imgPath):

    if type(imgPath) == str:
        print("Image file is String")
        bgrImg = cv2.imread(imgPath)
    else:
        print("image is array")
        bgrImg = imgPath
        # return 0
    print("Processing {}.".format(imgPath))
    # bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    try:
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    except Exception:
        print("Image is already in RGB format")
        rgbImg = bgrImg
    print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(rgbImg))
        # return "no faces detected"
    print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(96, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))

    print("  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)

    print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
    print("Representation:")
    print(rep)
    print("-----\n")
    return rep
# imgs = ['data/data/db/TJ/02.jpg', 'data/data/image-face.jpg']
# for (img1, img2) in itertools.combinations(imgs, 2):
#     d = getRep(img1) - getRep(img2)
#     print("Comparing {} with {}.".format(img1, img2))
#     print("  + Squared l2 distance between representations: {:0.3f}".format(np.dot(d, d)))


