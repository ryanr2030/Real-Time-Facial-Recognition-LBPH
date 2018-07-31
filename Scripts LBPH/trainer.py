import cv2
import numpy as np
from PIL import Image
import os
import sys

#Path to the face profiles
path = "../profiles/"

#define face_recognizer right not blank
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#call teh face detector as in other classes
face_detector = cv2.CascadeClassifier("../classifier/haarcascade_frontalface_default.xml")

#stores the integer labels
img_lab = []

#stores the x, y pixel matrix for the images
faceSamples = []

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    ids = []
    for imagePath in imagePaths:
        if imagePath.endswith(".DS_Store"):
            print("")
        else:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            #stores image pixels as x and y matrix in a numpy array
            img_numpy = np.array(PIL_img,'uint8')
            #splits the name of the image to get the User Id # between the two dots
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            #applies the face detector classifier to the numpy array stores the face
            faces = face_detector.detectMultiScale(img_numpy)

            #for all the pixels in the face
            for (x,y,w,h) in faces:
                #stores the face pixel data in the face sample array
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                #stores id number of the user in the id array
                img_lab.append(id)
    return faceSamples,ids

#calles the function
faces,ids = getImagesAndLabels(path)
#takes the data applies the lbph algorithm and stores the identified facial features of each user
face_recognizer.train(faces, np.array(img_lab))
#write the trained data to a custom classifier to be called by the recognizer
face_recognizer.write("../classifier/CustomLBPH.yml")
