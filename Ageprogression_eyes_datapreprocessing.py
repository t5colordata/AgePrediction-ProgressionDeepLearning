import os
import numpy as np
import cv2
from PIL import Image
from PIL import Image
import numpy as np 
from keras.utils import np_utils
from keras.models import Sequential,Graph
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Convolution1D,MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD,RMSprop
from keras.regularizers import l2, activity_l2,l1

file=os.listdir('.')
images=[]
for i in file:
    images.append('./'+i.strip())

faceCascade=cv2.CascadeClassifier('../eye.xml')
img_train=[]
img_train[:]=[]
processed_train=[]
processed_train[:]=[]
size=30,30
for i in range(0,len(images)):
    img=cv2.imread(images[i])
    img=cv2.resize(img,(224,224))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.4,minNeighbors=3,minSize=(30,30),flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    try:   
      x,y,w,h=faces[0]
      crop_img=gray[y:y+h,x:x+w]
      pil_image=Image.fromarray(crop_img)
      pl=pil_image.resize(size,Image.ANTIALIAS)
      processed_train.append(images[i])
      data=np.asarray( pl, dtype="int32")
      data.shape
      img_train.append(data)
    except Exception as e:
      print e

len(processed_train)

label=[]
X_train1=[]
X_train2=[]
X_train1[:]=[]
X_train2[:]=[]
label_1=[]
label_1[:]=[]
label_0=[]
label_0[:]=[]
it=0
for i in processed_train:
   for j in processed_train:
      if i.replace('./','')[0:3]==j.replace('./','')[0:3]:
         X_train1.append(i)
         label.append(1)
         X_train2.append(j)
         label_1.append(1)
      else:
         it=it+1
         if it%79==0:
            X_train1.append(i)
            label.append(0)
            X_train2.append(j)
            label_0.append(1)


X_train1_eye=[]
X_train1_eye[:]=[]
processed_x_train1_eye=[]
processed_x_train1_eye[:]=[]
size=30,30
for i in range(0,len(X_train1)):
   img=cv2.imread(X_train1[i])
   img=cv2.resize(img,(224,224))
   faceCascade=cv2.CascadeClassifier('../eye.xml')
   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   eyes = faceCascade.detectMultiScale(gray,scaleFactor=1.4,minNeighbors=3,minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
   try:
      x,y,w,h=eyes[0]
      crop_img=gray[y:y+h,x:x+w]
      processed_x_train1_eye.append(X_train1[i])
      pil_image=Image.fromarray(crop_img)
      pl=pil_image.resize(size,Image.ANTIALIAS)
      #pil_image.thumbnail(size,Image.ANTIALIAS)
      data=np.asarray( pl, dtype="int32")
      data.shape
      X_train1_eye.append(data)
   except:
      print ''


X_train2_eyes=[]
X_train2_eyes[:]=[]
processed_x_train2_eyes=[]
processed_x_train2_eyes[:]=[]
size=30,30
for i in range(0,len(X_train2)):
   img=cv2.imread(X_train2[i])
   img=cv2.resize(img,(224,224))
   faceCascade=cv2.CascadeClassifier('../eye.xml')
   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   eyes = faceCascade.detectMultiScale(gray,scaleFactor=1.4,minNeighbors=3,minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
   try:
      x,y,w,h=eyes[0]
      crop_img=gray[y:y+h,x:x+w]
      processed_x_train2_eyes.append(X_train2[i])
      pil_image=Image.fromarray(crop_img)
      pl=pil_image.resize(size,Image.ANTIALIAS)
      data=np.asarray( pl, dtype="int32")
      data.shape
      X_train2_eyes.append(data)
   except Exception as e:
      print e

img_tr1=X_train1_eye[0:12000]
len(img_tr1)
img_ts1=X_train1_eye[12000:14042]
len(img_ts1)
img_tr2=X_train2_eyes[0:12000]
len(img_tr2)
img_ts2=X_train2_eyes[12000:14042]
len(img_ts2)
img_np_train1=np.asarray(img_tr1)
img_np_train1.shape
img_np_train2=np.asarray(img_tr2)
img_np_train2.shape
img_np_test1=np.asarray(img_ts1)
img_np_test1.shape
img_np_test2=np.asarray(img_ts2)
img_np_test2.shape

XT1=np.reshape(img_np_train1,(12000,30,30,1)) 
XTrain1=np.reshape(XT1,(12000,1,30,30))
XT2=np.reshape(img_np_train2,(12000,30,30,1)) 
XTrain2=np.reshape(XT2,(12000,1,30,30)) 

XTS1=np.reshape(img_np_test1,(2042,30,30,1)) 
XTest1=np.reshape(XTS1,(2042,1,30,30))
XTS2=np.reshape(img_np_test2,(2042,30,30,1)) 
XTest2=np.reshape(XTS2,(2042,1,30,30)) 


XTrain1.shape
XTrain2.shape
XTest1.shape
XTest2.shape

y_train=label[0:12000]
y_test=label[12000:14042]
Y_train=np.asarray(y_train)
Y_test=np.asarray(y_test)
y_train2=np_utils.to_categorical(Y_train,2)
y_test2=np_utils.to_categorical(Y_test,2)

np.save('fgnet_xtrain1_eyes.npy',XTrain1)
np.save('fgnet_xtrain2_eyes.npy',XTrain2)
np.save('fgnet_xtest1_eyes.npy',XTest1)
np.save('fgnet_xtest2_eyes.npy',XTest2)
np.save('fgnet_ytrain_eyes.npy',y_train2)
np.save('fgnet_ytest_eyes.npy',y_test2)