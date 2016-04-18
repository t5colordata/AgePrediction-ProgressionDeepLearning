import os
import re
import PIL
import cv2
from PIL import Image
import numpy as np 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Convolution1D,MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD,RMSprop
from keras.regularizers import l2, activity_l2,l1

file_train=open('morph_train.csv','r')
file_test=open('morph_test.csv','r')
training=file_train.readlines()
testing=file_test.readlines()

train_label=[]
train_image=[]
for i in training:
    train_image.append('./Album2/'+i.strip().split('!#!')[0])
    train_label.append(int(i.strip().split('!#!')[1]))

test_label=[]
test_image=[]
for i in testing:
    test_image.append('./Album2/'+i.strip().split('!#!')[0])
    test_label.append(int(i.strip().split('!#!')[1]))


img_train=[]
img_train[:]=[]
processed_train=[]
processed_train[:]=[]
size=90,90
for i in range(0,len(train_image)):
   img=cv2.imread(train_image[i])
   img=cv2.resize(img,(224,224))
   faceCascade=cv2.CascadeClassifier('1.xml')
   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
   print train_image[i]
   try:
      x,y,w,h=faces[0]
      crop_img=gray[y:y+h,x:x+w]
      processed_train.append(train_image[i])
      pil_image=Image.fromarray(crop_img)
      pl=pil_image.resize(size,Image.ANTIALIAS)
      #pil_image.thumbnail(size,Image.ANTIALIAS)
      data=np.asarray( pl, dtype="int32")
      data.shape
      img_train.append(data)
   except:
      print ''

train_label=[]
train_label[:]=[]
for i in processed_train:
    for j in  training: 
      l_name='./Album2/'+j.strip().split('!#!')[0]
      if i==l_name:
            train_label.append(int(j.strip().split('!#!')[1]))

img_test=[]
img_test[:]=[]
processed_test=[]
processed_test[:]=[]
size=90,90
for i in range(0,len(test_image)):
   img=cv2.imread(test_image[i])
   img=cv2.resize(img,(224,224))
   faceCascade=cv2.CascadeClassifier('1.xml')
   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
   try:   
      x,y,w,h=faces[0]
      crop_img=gray[y:y+h,x:x+w]
      pil_image=Image.fromarray(crop_img)
      pl=pil_image.resize(size,Image.ANTIALIAS)
      processed_test.append(test_image[i])
      data=np.asarray( pl, dtype="int32")
      data.shape
      img_test.append(data)
   except:
      print ''


test_label=[]
test_label[:]=[]
for i in processed_test:
    for j in  testing: 
        l_name='./Album2/'+j.strip().split('!#!')[0]
        if i==l_name:
              test_label.append(int(j.strip().split('!#!')[1]))



img_np_train=np.asarray(img_train)
img_np_test=np.asarray(img_test)
print img_np_train.shape
print img_np_test.shape


img_np_train=np.reshape(img_np_train,(45007,120,120,1)) 
X_train=np.reshape(img_np_train,(45007,1,120,120)) 

img_np_test=np.reshape(img_np_test,(5265,120,120,1)) 
X_test=np.reshape(img_np_test,(5265,1,120,120)) 

Y_train=np.asarray(train_label)
Y_test=np.asarray(test_label)
y_train2=np_utils.to_categorical(Y_train,10)
y_test2=np_utils.to_categorical(Y_test,10)