

import cv2
import numpy as np
from scipy import ndimage
import sys
import os



#Creates the directory for a new face profile
def create_directory(face_profile):
 
    try:
        print ("Creating Directory")
        os.makedirs(face_profile)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            print ("The specified face profile already existed, it will be overwritten")
            raise
#Creates a new profile in the database
def create_profile_in_database(face_profile_name, database_path="../profiles/"):

    face_profile_path = database_path + face_profile_name + "/"
    create_directory(face_profile_path)
    return face_profile_path


#Face detection classifier 
face_detector= cv2.CascadeClassifier("../classifier/haarcascade_frontalface_default.xml") #create a cascade classifier
sideFace_detector = cv2.CascadeClassifier('../classifier/haarcascade_profileface.xml')



cam = cv2.VideoCapture(0)
#set height and width
cam.set(3, 640);
cam.set(4,480);



#number of faces to save counter for faces 
num_of_face_to_collect = 300
num_of_face_saved = 0


#  For saving face data to directory
profile_path= "../profiles/"

if len(sys.argv) == 1:
    print "\nError: No Name for Face Profile Specified\n"
    exit()
elif len(sys.argv) > 2:
    print "\nError: Only one profile can be made at a time\n"
    exit()
else:
    user_id = input('\nEnter User ID number 1 digit: ')
    name=sys.argv[1]

while(True):
    
    #computer reads the image
    ret, frame = cam.read()
    key = cv2.waitKey(1)
    #converts image from color to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #calls the front face detector and identifies the face in the image
    #look up what the 1.3 is scale factor 5 number of neighbors
    faces=face_detector.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        #create a rectangle the size of the face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)

        #saves file name and the path
        face_name = name+"."+str(user_id)+"."+str(num_of_face_saved)+".png"

        #saves the cropepd face
        cv2.imwrite(profile_path+face_name, gray[y:y+h,x:x+w])
        #tells user face was saved
        print ("Face Saved: "+face_name)
        num_of_face_saved += 1

     #signal handler        
    
    # if escape is hit exit
    if key in [27]: 
        break
    #breaks if number of desired faces are captured
    elif num_of_face_saved>=num_of_face_to_collect:
        break


  
    cv2.putText(frame, "Press ESC to quit.", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

    #displays the image to the screen
    cv2.imshow("Real Time Facial Recognition", frame)





cam.release()
cv2.destroyAllWindows()
