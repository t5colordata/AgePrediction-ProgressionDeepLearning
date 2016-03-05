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

###########################Data input
data = numpy.empty((2,3,816,816),dtype='float32')
>>> imgs = os.listdir('./age')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'os' is not defined
>>> import os
imgs = os.listdir('./age')
num = len(imgs)
num
j=0
for i in range(num):
   img = Image.open('./aligned/7153718@N04/'+imgs[i])
   arr = numpy.asarray(img,dtype='float32')
   data[j,:,:,:]=[arr[:,:,0],arr[:,:,1],arr[:,:,2]]
   j=j+1
>>> print (data.shape[0],'samples')
(2, 'samples')
>>> label=[13,14]
>>> label = np_utils.to_categorical(label,2)

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

nb_classes = 8
Y_train = np_utils.to_categorical(y_train, nb_classes)
######################################################################################################

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


model.add(Dense(8))

model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=0.0005, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)



