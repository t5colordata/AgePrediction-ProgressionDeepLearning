from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
from PIL import Image
from keras.utils import np_utils
import numpy
import os

#Data input

age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['m','f']
with open("   ") as f:
	lines=f.readlines()
lines.pop(0)
testList=[]
for each in lines:
	dic={}
	subjectDir=each.split('\t')[0]
	imageSubject=each.split('\t')[2]
	imageName='landmark_aligned_face.{0}.{1}'.format(imageSubject, each.split('\t')[1])
    imageAge=each.split('\t')[3]
    if imageAge=='(25 23)':
    	imageAge='(25 32'
    imageGender=each.split('\t')[4]
    dic['subjectDir']=subjectDir
    dic['imageName']=imageName
    dic['imageSubject']=imageSubject
    dic['imageAge']=imageAge
    dic['imageGender']=imageGender
    testList.append(dic)
img = Image.open(open('age.jpg'))
img = numpy.asarray(img,dtype='float64') / 256
img_ = img.transpose(2,0,1).reshape(1,3,816,816)

nb_classes = 8
y_train = [13]
Y_train = np_utils.to_categorical(y_train, nb_classes)


model = Sequential()


model.add(Convolution2D(96, 7, 7, border_mode='valid', input_shape=(3, 816, 816), subsample=(4, 4), W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization(epsilon=1e-04, mode=0, axis=3, momentum=0.75, weights=None))


model.add(Convolution2D(256, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization(epsilon=1e-04, mode=0, axis=3, momentum=0.75, weights=None))


model.add(Convolution2D(384, 3, 3, border_mode='same', W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())


model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(1))

model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=0.0005, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)



