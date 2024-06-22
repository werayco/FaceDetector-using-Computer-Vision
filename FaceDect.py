import cv2 as cv
import numpy as np
import os


direc=r"C:\Users\LENOVO-PC\Downloads\Images\IMAGG"

face_dect = cv.CascadeClassifier("TheFace.xml")  # instantiating the haar algo

models= os.listdir(direc)

features =[]

labels = []

def FaceDetect():     

    for imgFold in models:

        path = os.path.join(direc,imgFold)

        label = models.index(imgFold)

        for ind_img in os.listdir(path):

            img_path = os.path.join(path,ind_img)

            img_arr = cv.imread(img_path)

            # img_arr = cv.resize(img_arr,(90,90),interpolation=cv.INTER_AREA)
            
            gray = cv.cvtColor(img_arr,cv.COLOR_BGR2GRAY)

            faces_recg = face_dect.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

            for (x,y,w,h) in faces_recg:

                faces_roi = gray[y:y+h,x:x+w]

                features.append(faces_roi)

                labels.append(label)

FaceDetect()

feat = np.array(features,dtype="object")

labelss=np.array(labels)

face_rec = cv.face.LBPHFaceRecognizer.create()

face_rec.train(feat,labelss)

face_rec.save("TrainedRecg.yml")

np.save("features.npy",feat)

np.save("label.npy",labelss)

print(f"length of the features ={len(features)}")
print(f"length of the labels ={len(labels)}")

    




