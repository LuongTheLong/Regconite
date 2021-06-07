import os
import numpy as np
import dlib
import cv2
from PIL import Image
import datetime
import pandas as pd


path = "D:/Dlib/" #file Nguyen gui

def imgneedregco():
    test_path = "D:/Dlib/Datasetneedtotrain/TheLong.jpg"
    return test_path


def findEuclid(source_represent, test_represent):
    euclid_distance = source_represent - test_represent
    print(euclid_distance)
    euclid_distance = np.sum(np.multiply(euclid_distance, euclid_distance))
    print(euclid_distance)
    euclid_distance = np.sqrt(euclid_distance)
    return euclid_distance

def savedata(df):
    df.to_csv(r'D:/Dlib/checkaccurate.csv', index=None, header=True)


img1 = dlib.load_rgb_image(imgneedregco())
def getImageWithName(Path, imgreg):
    sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    accurates = []

    for imagePath in imagePaths:

        name = str(imagePath)

        img2 = dlib.load_rgb_image(name)
        height, width, channels = img2.shape
        rec = dlib.rectangle(0, 0, height, width)
        img2_shape = sp(img2, rec)

        img2_aligned = dlib.get_face_chip(img2, img2_shape)

        img2_represent = model.compute_face_descriptor(img2_aligned)

        # img2_embeeding = np.array(img2_represent)

        # names.append(name.split("\\")[1].split(".")[0])

        # img2_embeedings.append(img2_embeeding)

        height1, width1, channels = imgreg.shape
        rec = dlib.rectangle(0, 0, height1, width1)
        img1_shape = sp(imgreg, rec)
        img1_aligned = dlib.get_face_chip(imgreg, img1_shape)
        img1_represent = model.compute_face_descriptor(img1_aligned)
        img1_represent = np.array(img1_represent)
        cv2.waitKey(10)
        distance = findEuclid(img1_represent, img2_represent)
        accurate = 1 - distance
        accurates.append(accurate)
        df = pd.DataFrame(accurates)
    savedata(df)
    return df

df = getImageWithName(path, img1)
print(df)



