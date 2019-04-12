

import cv2
import numpy as np
from os import listdir
from os.path import isfile,join

training_data,labels=[],[]

data_path='./faces/'
data_path1=listdir(data_path)

def load_images(only_files,j):
    for i, files in enumerate(only_files):
        image_path=data_path2+only_files[i]
        images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        training_data.append(np.asarray(images,dtype=np.uint8))
        labels.append(j+1)

for j in range(0,len(data_path1)):
    data_path2='./faces/'+data_path1[j]+'/'
    print(data_path2)
    only_files=[f for f in listdir(data_path2) if isfile(join(data_path2,f))]
    load_images(only_files,j)
            
labels=np.asarray(labels,dtype=np.int32)

print(cv2.__version__)
#model=cv2.facecreateLBPHFaceRecognizer()

model=cv2.face.LBPHFaceRecognizer_create()


model.train(np.asarray(training_data),np.asarray(labels))

print("Model terained succesfully")

face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img,size=0.5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if faces is ():
        return img,[]
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))
    
    return img,roi

cap=cv2.VideoCapture(0)

name=['sarthak']


while True:
    ret,frame=cap.read()
    image,face=face_detector(frame)
    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        results=model.predict(face)
        print(results)
        if results[1]<500:
            confidence=int(100*(1-(results[1])/300))
            display_string=str(confidence)+'% Confident it is {}'.format(name[results[0]-1])
            cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        
        if confidence>75:
            cv2.putText(image,"UNLOCKED",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            cv2.imshow('Face Recognizer',image)
        else:
            cv2.putText(image,"LOCKED",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            cv2.imshow('Face Recognizer',image)
    except:
        cv2.putText(image,"No face found",(220,120),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        cv2.putText(image,"LOCKED",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        cv2.imshow('Face Recognizer',image)
        pass
    
    if cv2.waitKey():
        break

cap.release()
cv2.destroyAllWindows()
