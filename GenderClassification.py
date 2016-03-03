from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD

#Data input





model = Sequential()

#Cov1
model.add(Convolution2D(96, 7, 7, border_mode='valid', input_shape=(3, 227, 227), subsample=(4, 4), W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization(epsilon=1e-04, mode=0, axis=3, momentum=0.75, weights=None))

#Cov2
model.add(Convolution2D(256, 5, 5, border_mode='same', W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization(epsilon=1e-04, mode=0, axis=3, momentum=0.75, weights=None))

#Cov3
model.add(Convolution2D(384, 3, 3, border_mode='same', W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())

#FC1
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#FC2
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#FC3
model.add(Dense(2))

model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=0.0005, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)


