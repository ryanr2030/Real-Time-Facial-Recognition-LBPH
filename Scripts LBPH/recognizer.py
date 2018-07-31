import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("../classifier/CustomLBPH.yml")
cascadePath = "../classifier/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['Ryan Reynolds', 'Brett Dawson', '', '', '', ''] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.25*cam.get(3)
minH = 0.25*cam.get(4)

#loop to read image by image
while True:
    ret, img =cam.read()

     #converts the image to grayscales
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #detects faces in image feed from camera
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    #for each face found
    for(x,y,w,h) in faces:
        #draw a rectangle arround the face
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        #call our custom lbph classifier returns the ID # of the person it thinks it is and how confident it is
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])


        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 55):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  


    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n Exiting")
cam.release()
cv2.destroyAllWindows()
