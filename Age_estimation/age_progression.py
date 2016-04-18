model = Sequential()


model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(3, 180, 130), subsample=(2, 2), W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

model.add(Convolution2D(32, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())


model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(801))

model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=0.0005, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train, batch_size=32, nb_epoch=100)
