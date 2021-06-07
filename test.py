import os
import numpy as np
import dlib
import cv2
from PIL import Image
import datetime
import pandas as pd

def loaddata():
    return pd.read_csv("training.csv")

def imgneedregco():
    test_path = "D:/Dlib/image/102180157/71.jpg"
    return test_path
sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


pdf = loaddata()

arr = pdf.to_numpy()



def findEuclid(source_represent, test_represent):
    euclid_distance = source_represent - test_represent
    euclid_distance = np.sum(np.multiply(euclid_distance, euclid_distance))
    euclid_distance = np.sqrt(euclid_distance)
    return euclid_distance

img1 = dlib.load_rgb_image(imgneedregco())
try:
    height, width, channels = img1.shape
    rec = dlib.rectangle(0, 0, height, width)
    img1_shape = sp(img1, rec)
    # print("shape" + str(img1_shape))
    img1_aligned = dlib.get_face_chip(img1, img1_shape)
    img1_represent = model.compute_face_descriptor(img1_aligned)
    # print("represent" + str(img1_represent))
    img1_represent = np.array(img1_represent)
    location = 0
    min = 1
    for i in range(arr.shape[0]):
        distance = findEuclid(img1_represent, arr[i][:-1])
        if distance < min:
            min = distance
            name = arr[i][-1]
    thresh_hold = 0.393587 #thresh_hold được tính từ các dữ liệu dataset trước đó
    if min < thresh_hold:
        print(min)
        print(name)
    if min > thresh_hold:
        print("diffirent")
except:
    print("cant reg")

