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

graph = Graph() 
graph.add_input(name='input1', input_shape=(1,120,120)) 
graph.add_input(name='input2', input_shape=(1,120,120))
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_1',input='input1')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_2',input='conv1_1')
graph.add_node(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),name='max_conv1_1',input='conv1_2')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_3',input='max_conv1_1')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_4',input='conv1_3')
graph.add_node(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),name='max_conv1_2',input='conv1_4')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_5',input='max_conv1_2')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l1(0.01),activation='relu'),name='conv1_6',input='conv1_5')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_1',input='input2')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_2',input='conv2_1')
graph.add_node(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),name='max_conv2_1',input='conv2_2')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_3',input='max_conv2_1')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_4',input='conv2_3')
graph.add_node(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),name='max_conv2_2',input='conv2_4')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_5',input='max_conv2_2')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',W_regularizer=l1(0.01),activation='relu'),name='conv2_6',input='conv2_5')

graph.add_shared_node(Convolution2D(64,3,3,activation='relu',border_mode="valid"),name='combined_layer_1',inputs=['conv1_6','conv2_6'],merge_mode='ave')
graph.add_node(Flatten(), name='flatten_layer',input='combined_layer_1')
graph.add_node(Dense(256,activation='relu'),name='combined_dense_layer_1',input='flatten_layer')
graph.add_node(Dense(2,activation='softmax'),name='output_layer',input='combined_dense_layer_1')
graph.add_output(name='output1',input='output_layer')
graph.compile('sgd',{'output1':'binary_crossentropy'})
graph.get_config(verbose=1)
graph.fit({'input1':XTrain1,'input2':XTrain2,'output1':y_train2},batch_size=32,nb_epoch=5)