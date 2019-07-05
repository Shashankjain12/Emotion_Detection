#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 02:00:28 2019

@author: shashank
"""
from statistics import mode
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Conv2D
#from keras.layers import MaxPool2D
from keras.models import load_model
import cv2
import numpy as np
import pandas as pd
#from keras.preprocessing.image import img_to_array
IP="YOUR_IP/video"
person_count=pd.read_pickle("/home/shashank/Desktop/mindmapperz/emotion_detection/object/obj.pkl")
classifier=load_model("/home/shashank/Desktop/mindmapperz/emotion_detection//models/emotion_model.hdf5")

face_cascade=cv2.CascadeClassifier("/home/shashank/Desktop/mindmapperz/emotion_detection/classifiers/face.xml")

emotion_labels={0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}

emotion_window = []

emotion_target_size = classifier.input_shape[1:3]
print(emotion_target_size)
emotion_offsets = (20, 40)

frame_window=10

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


sentiments={"happy":[],"sad":[],"angry":[],"disgust":[],"fear":[],"surprise":[],"neutral":[]}
def detect(gray,frame):
    faces=face_cascade.detectMultiScale(gray,1.3,8,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        face_coordinates=(x,y,w,h)
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        image = gray[y1:y2, x1:x2]
        #roi_gray=gray[y:y+h,x:x+w]
        #roi_color=frame[y:y+h,x:x+w]
        try:
            image=cv2.resize(image,(emotion_target_size))
        except:
            continue
        #image=image.astype("float")/255.0
        #image=img_to_array(image)
        image= preprocess_input(image, True)
        image=np.expand_dims(image,axis=0)
        image=np.expand_dims(image,axis=-1)

        #image=np.expand_dims(image,axis=-1)

        pred=classifier.predict(image)
        #print(pred)
        emotion_probability = np.max(pred)
        emotion_label_arg = np.argmax(pred)
        print("emotion_label_arg:",emotion_label_arg)
        print("emotion_prob:",emotion_probability)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue
        #print(emotion_window)
        #print(emotion_mode)
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #if(emotion_probability>0.65):
        cv2.putText(frame,emotion_mode,(x,y-10), font, 1,color,1,cv2.LINE_AA)
        if emotion_mode == 'angry':
            sentiments["angry"].append(1)
        elif emotion_mode == 'sad':
            sentiments["sad"].append(1)
        elif emotion_mode == 'happy':
            sentiments["happy"].append(1)
        elif emotion_mode == 'surprise':
            sentiments["surprise"].append(1)
        elif emotion_mode =='neutral':
            sentiments["neutral"].append(1)
        elif emotion_mode =='fear':
            sentiments["fear"].append(1)
        elif emotion_mode=='disgust':
            sentiments["disgust"].append(1)
        #x, y = coordinates[:2]
        #cv2.putText(image_array, text, (x + x_offset, y + y_offset),
        #        cv2.FONT_HERSHEY_SIMPLEX,
        #        font_scale, color, thickness, cv2.LINE_AA)

    return frame
#For Ip Webcam
video_cap=cv2.VideoCapture(IP)
#To use your own camera use:-
#video_cap=cv2.VideoCapture(0)

while True:
    ret,frame=video_cap.read()
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('image', 1366,768)
    frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LINEAR)
    if ret:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        canvas=detect(gray,frame)
        cv2.imshow("image",canvas)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    else:
        break


video_cap.release()
cv2.destroyAllWindows()

print("Total percentage of the students that are happy are:"+str(sentiments["happy"].count(1)/person_count)+"%")
print("Total percentage of the students that are sad are:"+str(sentiments["sad"].count(1)/person_count)+"%")
print("Total percentage of the students that are fear are:"+str(sentiments["fear"].count(1)/person_count)+"%")
print("Total percentage of the students that are disgust are:"+str(sentiments["disgust"].count(1)/person_count)+"%")
print("Total percentage of the students that are angry are:"+str(sentiments["angry"].count(1)/person_count)+"%")
print("Total percentage of the students that are neutral are:"+str(sentiments["neutral"].count(1)/person_count)+"%")
print("Total percentage of the students that are surprise are:"+str(sentiments["surprise"].count(1)/person_count)+"%")
