from django.shortcuts import render
import os
import json
from django.http import HttpResponse
from django.conf import settings
from keras.utils import np_utils
from keras.models import Sequential,Graph
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD,RMSprop
from keras.regularizers import l2, activity_l2,l1
import cv2, numpy as np
import PIL
import re
from PIL import Image

# Create your views here.
#age prediction model
model_from_json(open('/Users/jiangjiangzhu/Downloads/my_arch_510pm_0414_latest.json').read())
model=model_from_json(open('/Users/jiangjiangzhu/Downloads/my_arch_510pm_0414_latest.json').read())
model.load_weights('/Users/jiangjiangzhu/Downloads/my_arch_510pm_0414_weights_lates.h5')

#age progression model
# graph = Graph() 
# graph.add_input(name='input1', input_shape=(1,120,120)) 
# graph.add_input(name='input2', input_shape=(1,120,120))
# graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_1',input='input1')
# graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_2',input='conv1_1')
# graph.add_node(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),name='max_conv1_1',input='conv1_2')
# graph.add_node(BatchNormalization(),name='batchnorm_conv1_1',input='max_conv1_1')
# graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv1_3',input='batchnorm_conv1_1')
# graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv1_4',input='conv1_3')
# graph.add_node(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),name='max_conv1_2',input='conv1_4')
# graph.add_node(BatchNormalization(),name='batchnorm_conv1_2',input='max_conv1_2')
# graph.add_node(Convolution2D(256, 3, 3, border_mode='valid',activation='relu'),name='conv1_5',input='batchnorm_conv1_2')
# graph.add_node(Convolution2D(256, 3, 3, border_mode='valid',activation='relu'),name='conv1_6',input='conv1_5')
# graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_1',input='input2')
# graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_2',input='conv2_1')
# graph.add_node(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),name='max_conv2_1',input='conv2_2')
# graph.add_node(BatchNormalization(),name='batchnorm_conv2_1',input='max_conv2_1')
# graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv2_3',input='batchnorm_conv2_1')
# graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv2_4',input='conv2_3')
# graph.add_node(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),name='max_conv2_2',input='conv2_4')
# graph.add_node(BatchNormalization(),name='batchnorm_conv2_2',input='max_conv2_2')
# graph.add_node(Convolution2D(256, 3, 3, border_mode='valid',activation='relu'),name='conv2_5',input='batchnorm_conv2_2')
# graph.add_node(Convolution2D(256, 3, 3, border_mode='valid',activation='relu'),name='conv2_6',input='conv2_5')
# graph.add_shared_node(Convolution2D(256,3,3,activation='relu',border_mode="valid"),name='combined_layer_1',inputs=['conv1_6','conv2_6'],merge_mode='sum')
# graph.add_node(Flatten(), name='flatten_layer',input='combined_layer_1')
# graph.add_node(Dense(256,activation='relu'),name='combined_dense_layer_1',input='flatten_layer')
# graph.add_node(Dense(2,activation='softmax'),name='output_layer',input='combined_dense_layer_1')
# graph.add_output(name='output1',input='output_layer')
# graph.load_weights('/Users/jiangjiangzhu/Downloads/graph_improved_1_weights.h5')
# graph.compile('sgd',{'output1':'binary_crossentropy'})
graph = Graph() 
graph.add_input(name='input1', input_shape=(1,120,120)) 
graph.add_input(name='input2', input_shape=(1,120,120))
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_1',input='input1')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_1a',input='conv1_1')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv1_1b',input='conv1_1a')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv1_2',input='conv1_1b')
graph.add_node(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),name='max_conv1_1',input='conv1_2')
graph.add_node(BatchNormalization(),name='batchnorm_conv1_1',input='max_conv1_1')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv1_3',input='batchnorm_conv1_1')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv1_4',input='conv1_3')
graph.add_node(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),name='max_conv1_2',input='conv1_4')
graph.add_node(BatchNormalization(),name='batchnorm_conv1_2',input='max_conv1_2')
graph.add_node(Convolution2D(256, 3, 3, border_mode='valid',activation='relu'),name='conv1_5',input='batchnorm_conv1_2')
graph.add_node(Convolution2D(256, 3, 3, border_mode='valid',activation='relu'),name='conv1_6',input='conv1_5')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_1',input='input2')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_1a',input='conv2_1')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv2_1b',input='conv2_1a')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv2_2',input='conv2_1b')
graph.add_node(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),name='max_conv2_1',input='conv2_2')
graph.add_node(BatchNormalization(),name='batchnorm_conv2_1',input='max_conv2_1')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv2_3',input='batchnorm_conv2_1')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv2_4',input='conv2_3')
graph.add_node(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),name='max_conv2_2',input='conv2_4')
graph.add_node(BatchNormalization(),name='batchnorm_conv2_2',input='max_conv2_2')
graph.add_node(Convolution2D(256, 3, 3, border_mode='valid',activation='relu'),name='conv2_5',input='batchnorm_conv2_2')
graph.add_node(Convolution2D(256, 3, 3, border_mode='valid',activation='relu'),name='conv2_6',input='conv2_5')
 
graph.add_shared_node(Convolution2D(256,3,3,activation='relu',border_mode="valid"),name='combined_layer_1',inputs=['conv1_6','conv2_6'],merge_mode='sum')
graph.add_node(Flatten(), name='flatten_layer',input='combined_layer_1')
graph.add_node(Dense(256,activation='relu'),name='combined_dense_layer_1',input='flatten_layer')
graph.add_node(Dense(2,activation='softmax'),name='output_layer',input='combined_dense_layer_1')
graph.add_output(name='output1',input='output_layer')
graph.load_weights('/Users/jiangjiangzhu/Downloads/grap_improved_latest.h5')
graph.compile('sgd',{'output1':'binary_crossentropy'})

image1 = []
image2 = []
def home(request):
    return render(request, 'home.html')
    
def prediction(image):
    img_predict=[]
    img_predict[:]=[]
    size = 120,120
    im = cv2.resize(np.asarray(image), (224, 224))
    faceCascade=cv2.CascadeClassifier('/Users/jiangjiangzhu/Downloads/1.xml')
    try:
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    except:
        gray=im
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    print faces
    try:
        x,y,w,h=faces[0]
        crop_img=gray[y:y+h,x:x+w]
        pil_image=Image.fromarray(crop_img)
        pl=pil_image.resize(size,Image.ANTIALIAS)
        data=np.asarray( pl, dtype="int32")
        data.shape
        img_predict.append(data)
        print img_predict
    except:
        pil_image_e = Image.fromarray(gray)
        pl_e = pil_image_e.resize(size,Image.ANTIALIAS)
        data_e=np.asarray(pl_e,dtype="int32")
        data_e.shape
        img_predict.append(data_e)
        img_np_predict_e=np.asarray(img_predict)
        img_np_predict_e=np.reshape(img_np_predict_e,(1,120,120,1))
        sample=np.reshape(img_np_predict_e,(1,1,120,120))
        out_e=model.predict_classes(sample)
        print out_e
        return out_e
    img_np_predict=np.asarray(img_predict)
    img_np_predict=np.reshape(img_np_predict,(1,120,120,1)) 
    X_train=np.reshape(img_np_predict,(1,1,120,120)) 
    out=model.predict_classes(X_train)
    out_propability = model.predict(X_train)
    print out_propability
    print out
    print out[0]
    return out[0]
    
    
def age_prediction(request):
  age_list=["16~18","19~21","22~24","25~27","29~32","33~35","36~39","40~43","44~49","50~77"]
  if request.method == 'POST':
    data = request.FILES.get('file')
    print data
    image = Image.open(data)
    print image
  # if image.mode == 'RGBA':
#         image = image.convert('RGB')
#         print "ddd"
#     print image
    out = prediction(image)
    return HttpResponse(
        json.dumps(age_list[out]),
        content_type='application/json')
        
def age_prediction_1(request):
    image = Image.open("/Users/jiangjiangzhu/colordata/deeplearning/static/1.jpg")
    age_list=["16~18","19~21","22~24","25~27","29~32","33~35","36~39","40~43","44~49","50~77"]
    out = prediction(image)
    return HttpResponse(
        json.dumps(age_list[out]),
        content_type='application/json')
        
def age_prediction_2(request):
    image = Image.open("/Users/jiangjiangzhu/colordata/deeplearning/static/2.jpg")
    age_list=["16~18","19~21","22~24","25~27","29~32","33~35","36~39","40~43","44~49","50~77"]
    out = prediction(image)
    return HttpResponse(
        json.dumps(age_list[out]),
        content_type='application/json')
        
def age_prediction_3(request):
    image = Image.open("/Users/jiangjiangzhu/colordata/deeplearning/static/3.jpg")
    age_list=["16~18","19~21","22~24","25~27","29~32","33~35","36~39","40~43","44~49","50~77"]
    out = prediction(image)
    return HttpResponse(
        json.dumps(age_list[out]),
        content_type='application/json')
        
def age_prediction_4(request):
    image = Image.open("/Users/jiangjiangzhu/colordata/deeplearning/static/4.jpg")
    age_list=["16~18","19~21","22~24","25~27","29~32","33~35","36~39","40~43","44~49","50~77"]
    out = prediction(image)
    return HttpResponse(
        json.dumps(age_list[out]),
        content_type='application/json')
        
def age_prediction_5(request):
    image = Image.open("/Users/jiangjiangzhu/colordata/deeplearning/static/5.jpg")
    age_list=["16~18","19~21","22~24","25~27","29~32","33~35","36~39","40~43","44~49","50~77"]
    out = prediction(image)
    return HttpResponse(
        json.dumps(age_list[out]),
        content_type='application/json')
        
def age_prediction_6(request):
    image = Image.open("/Users/jiangjiangzhu/colordata/deeplearning/static/6.jpg")
    age_list=["16~18","19~21","22~24","25~27","29~32","33~35","36~39","40~43","44~49","50~77"]
    out = prediction(image)
    return HttpResponse(
        json.dumps(age_list[out]),
        content_type='application/json')
        
def age_prediction_7(request):
    image = Image.open("/Users/jiangjiangzhu/colordata/deeplearning/static/7.jpg")
    age_list=["16~18","19~21","22~24","25~27","29~32","33~35","36~39","40~43","44~49","50~77"]
    out = prediction(image)
    return HttpResponse(
        json.dumps(age_list[out]),
        content_type='application/json')
        
def age_prediction_8(request):
    image = Image.open("/Users/jiangjiangzhu/colordata/deeplearning/static/8.jpg")
    age_list=["16~18","19~21","22~24","25~27","29~32","33~35","36~39","40~43","44~49","50~77"]
    out = prediction(image)
    return HttpResponse(
        json.dumps(age_list[out]),
        content_type='application/json')
        
def age_prediction_9(request):
    image = Image.open("/Users/jiangjiangzhu/colordata/deeplearning/static/9.jpg")
    age_list=["16~18","19~21","22~24","25~27","29~32","33~35","36~39","40~43","44~49","50~77"]
    out = prediction(image)
    return HttpResponse(
        json.dumps(age_list[out]),
        content_type='application/json')
        
def age_prediction_10(request):
    image = Image.open("/Users/jiangjiangzhu/colordata/deeplearning/static/10.jpg")
    age_list=["16~18","19~21","22~24","25~27","29~32","33~35","36~39","40~43","44~49","50~77"]
    out = prediction(image)
    return HttpResponse(
        json.dumps(age_list[out]),
        content_type='application/json')
        
def crop(image):
    img_predict=[]
    img_predict[:]=[]
    size = 120,120
    im = cv2.resize(np.asarray(image), (224, 224))
    faceCascade=cv2.CascadeClassifier('/Users/jiangjiangzhu/Downloads/1.xml')
    try:
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    except:
        gray = im
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    print faces
    try:
        x,y,w,h=faces[0]
        crop_img=gray[y:y+h,x:x+w]
        pil_image=Image.fromarray(crop_img)
        pl=pil_image.resize(size,Image.ANTIALIAS)
        data=np.asarray( pl, dtype="int32")
        data.shape
        img_predict.append(data)
        print img_predict
    except:
        pil_image_e = Image.fromarray(gray)
        pl_e = pil_image_e.resize(size,Image.ANTIALIAS)
        data_e=np.asarray(pl_e,dtype="int32")
        data_e.shape
        img_predict.append(data_e)
        img_np_predict_e=np.asarray(img_predict)
        img_np_predict_e=np.reshape(img_np_predict_e,(1,120,120,1))
        sample=np.reshape(img_np_predict_e,(1,1,120,120))
        return sample
    img_np_predict=np.asarray(img_predict)
    img_np_predict=np.reshape(img_np_predict,(1,120,120,1)) 
    X_train=np.reshape(img_np_predict,(1,1,120,120)) 
    print X_train
    return X_train
    
    
        
def age_progression(request):
    global image1, image2
    if len(image1) == 0:
        print len(image1)
        data = request.FILES.get('file')
        print data
        image1 = Image.open(data)
        print image1
        image1 = crop(image1)
        return HttpResponse(
              json.dumps(''),
              content_type='application/json')
    else:
        data = request.FILES.get('file')
        print data
        image2 = Image.open(data)
        print image2
        image2 = crop(image2)
        print image2
        print image1
        predictions=graph.predict({'input1':image1,'input2':image2}) 
        print predictions
        help = predictions['output1']
        out = help[0]
        output = out[0]
        print output
        image1 = []
        image2 = []
        return HttpResponse(
              json.dumps(output),
              content_type='application/json')
        
        









