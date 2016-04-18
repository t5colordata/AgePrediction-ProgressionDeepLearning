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

new_path=os.path.join('.','aligned')
files=os.listdir(new_path)
len(files)
file=open('aligned_age.txt','r')
label_text=file.readlines()
len(label_text)

folder_names=[]
for i in files:
    image_path=os.path.join(new_path,i)
    folder_names.append(i)

label_text_cleaned=[]
for i in label_text:
   label_text_cleaned.append(i.strip().split(' ')[0])

file_list=[]
file_list[:]=[]
for i in folder_names:  
    fdr_name=os.path.join('./aligned/',i+'/')
    list=os.listdir(os.path.join('./aligned/',i+'/'))
    for j in list:
        if j.replace('landmark_aligned_face.','') in label_text_cleaned:
           file_list.append(fdr_name+j.strip())


a=set()
a2=[]
a2[:]=[]
size=35,35
processed=[]
processed[:]=[]
for i in range(0,len(file_list)):
   img=cv2.imread(file_list[i])
   img=cv2.resize(img,(0,0),fx=0.25,fy=0.25)
   faceCascade=cv2.CascadeClassifier('1.xml')
   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   gray.shape
   faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
   try:
      x,y,w,h=faces[0]
      processed.append(file_list[i])
      crop_img=gray[y:y+h,x:x+w]
      pil_image=Image.fromarray(crop_img)
      pil_image.thumbnail(size,Image.ANTIALIAS)
      data=np.asarray( pil_image, dtype="int32")
      a2.append(data)
   except:
      dummyvar=1




label_4=[]
label_others=[]
label_4[:]=[]
label_others[:]=[]
for i in processed:
    name=i.replace('face.','face~').split('~')[1]
    for j in label_text:
        l_name=j.strip().split(' ')[0]
        label=j.strip().split(' ')[1]
        if l_name==name:
            if int(label)==4:
                   label_4.append(i)
                   #label_4.append(int(label))
            else:
                   label_others.append(i)


for i in range(0,1500):
    label_others.append(label_4[i])

labels=[]
labels[:]=[]
for i in label_others:
    name=i.replace('face.','face~').split('~')[1]
    for j in label_text:
        l_name=j.strip().split(' ')[0]
        label=j.strip().split(' ')[1]
        if l_name==name:
                   labels.append(int(label))



a2=[]
a2[:]=[]
#size=35,35
s=120,120
for i in range(0,len(label_others)):
   img=cv2.imread(label_others[i])
   img=cv2.resize(img,(0,0),fx=0.25,fy=0.25)
   faceCascade=cv2.CascadeClassifier('1.xml')
   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   gray.shape
   faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
   try:
      x,y,w,h=faces[0]
      crop_img=gray[y:y+h,x:x+w]
      pil_image=Image.fromarray(crop_img)
      pl=pil_image.resize(s,Image.ANTIALIAS)
      #pil_image.thumbnail(size,Image.ANTIALIAS)
      data=np.asarray( pl, dtype="int32")
      data.shape
      a2.append(data)
   except:
      dummyvar=1



img_np_array=np.asarray(a2)
print img_np_array.shape

img_np_array_new=np.reshape(img_np_array,(10967,120,120,1)) 
img_np_array_new=np.reshape(img_np_array_new,(10967,1,120,120)) 

X_train=img_np_array_new[0:8000]
X_test=img_np_array_new[8000:10967]
label_np_array=np.asarray(labels)
label_np_array.shape

Y_train=label_np_array[0:8000]
Y_test=label_np_array[8000:10967]
y_train2=np_utils.to_categorical(Y_train,8)
y_test2=np_utils.to_categorical(Y_test,8)