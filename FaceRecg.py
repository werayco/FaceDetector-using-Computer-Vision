import cv2 as cv
import numpy as np
from typing import List
import os
import matplotlib.pyplot as plt

faceRec=cv.face.LBPHFaceRecognizer.create()

direc="IMAGG"

models= os.listdir(direc)

faceRec.read(r"TrainedRecg.yml")

faceDec=cv.CascadeClassifier(r"TheFace.xml")

'''This Function returns the region of interest of the detected image using the Haar Cascade Classifier'''

img = cv.imread(r"C:\Users\LENOVO-PC\Downloads\download (8).png") # Opening the image in grayscale and converting it to Matrix

gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

facList=faceDec.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=3)

for (x,y,w,h) in facList:

    regOfInt = gray_img[y:y+h,x:x+w]

    label,conf = faceRec.predict(regOfInt)

    cv.putText(img,str(models[label]),(x,h),fontFace=cv.FONT_HERSHEY_PLAIN,fontScale=1,color=(0,255,0),thickness=1)

    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=1)

    print(conf)

    cv.imshow("iimg",img)


cv.waitKey(0)

# print(len(roi))
