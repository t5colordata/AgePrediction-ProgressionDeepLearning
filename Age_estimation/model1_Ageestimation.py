import pandas as pd
import numpy as np
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Convoltion1D,MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD,RMSprop
from keras.regularizers import l2, activity_l2,l1

X_train=img_np_array[0:14000]
X_test=img_np_array[14000:17393]
label_np_array=np.asarray(labels)
label_np_array.shape
Y_train=label_np_array[0:14000]
Y_test=label_np_array[14000:17393]
y_train2=np_utils.to_categorical(Y_train,8)
y_test2=np_utils.to_categorical(Y_test,8)


#Keras CNN
batch_size = 10 #larger batch size gives error i.e. memory error "you might consider using 'theano.shared(..., borrow=True)'")
nb_epoch = 25
nb_classes = 2

# input image dimensions
X_rows, X_cols = X_train3.shape[2],X_train3.shape[3]

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

model = Sequential()

model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, X_rows, X_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
model.add(Convolution2D(32,5,5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(3,3)))
model.add(Convolution2D(32,4,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 3),strides=(2,2)))
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
model.add(Convolution2D(1,1,1))
model.add(Activation('relu'))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='Adam')
#model.compile(loss='categorical_crossentropy', optimizer='adadelta') #rmsprop
model.fit(X_train, y_train2, batch_size=32, nb_epoch=1, show_accuracy=True, verbose=2)
score = model.evaluate(X_test, y_test2, show_accuracy=True, verbose=0)


valid_preds = model.predict_proba(X_test3, verbose=0)
print(valid_preds[0:5])
preds = model.predict_classes(X_test, verbose=0)
#print(preds.shape)
print(preds)

#model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
#score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])