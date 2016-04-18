graph = Graph() 
graph.add_input(name='input1', input_shape=(1,120,120)) 
graph.add_input(name='input2', input_shape=(1,120,120))
graph.add_input(name='input3', input_shape=(1,30,30)) 
graph.add_input(name='input4', input_shape=(1,30,30))
graph.add_input(name='input5', input_shape=(1,30,30)) 
graph.add_input(name='input6', input_shape=(1,30,30))
graph.add_input(name='input7', input_shape=(1,30,30)) 
graph.add_input(name='input8', input_shape=(1,30,30))

## mouth

graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_1m',input='input3')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_2m',input='conv1_1m')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv1_3m',input='conv1_2m')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv1_4m',input='conv1_3m')
graph.add_node(Convolution2D(256, 3, 3, border_mode='valid',activation='relu'),name='conv1_5m',input='conv1_4m')
graph.add_node(Convolution2D(256, 3, 3, border_mode='valid',activation='relu'),name='conv1_6m',input='conv1_5m')

graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_1m',input='input4')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_2m',input='conv2_1m')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_3m',input='conv2_2m')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_4m',input='conv2_3m')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv2_5m',input='conv2_4m')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv2_6m',input='conv2_5m')

graph.add_shared_node(Convolution2D(256,3,3,activation='relu',border_mode="valid"),name='combined_layer_1m',inputs=['conv1_6m','conv2_6m',],merge_mode='sum')

## eyes

graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_1e',input='input5')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_2e',input='conv1_1e')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv1_3e',input='conv1_2e')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv1_4e',input='conv1_3e')
graph.add_node(Convolution2D(256, 3, 3, border_mode='valid',activation='relu'),name='conv1_5e',input='conv1_4e')
graph.add_node(Convolution2D(256, 3, 3, border_mode='valid',activation='relu'),name='conv1_6e',input='conv1_5e')

graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_1e',input='input6')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_2e',input='conv2_1e')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_3e',input='conv2_2e')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_4e',input='conv2_3e')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv2_5e',input='conv2_4e')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv2_6e',input='conv2_5e')

graph.add_shared_node(Convolution2D(256,3,3,activation='relu',border_mode="valid"),name='combined_layer_1e',inputs=['conv1_6e','conv2_6e',],merge_mode='sum')

## nose

graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_1n',input='input5')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv1_2n',input='conv1_1n')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv1_3n',input='conv1_2n')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv1_4n',input='conv1_3n')
graph.add_node(Convolution2D(256, 3, 3, border_mode='valid',activation='relu'),name='conv1_5n',input='conv1_4n')
graph.add_node(Convolution2D(256, 3, 3, border_mode='valid',activation='relu'),name='conv1_6n',input='conv1_5n')

graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_1n',input='input6')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_2n',input='conv2_1n')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_3n',input='conv2_2n')
graph.add_node(Convolution2D(64, 3, 3, border_mode='valid',activation='relu'),name='conv2_4n',input='conv2_3n')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv2_5n',input='conv2_4n')
graph.add_node(Convolution2D(128, 3, 3, border_mode='valid',activation='relu'),name='conv2_6n',input='conv2_5n')

graph.add_shared_node(Convolution2D(256,3,3,activation='relu',border_mode="valid"),name='combined_layer_1n',inputs=['conv1_6n','conv2_6n',],merge_mode='sum')

## face 

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

graph.add_shared_node(Convolution2D(256,3,3,activation='relu',border_mode="valid"),name='combined_layer_1',inputs=['conv1_6','conv2_6',],merge_mode='sum')

## combine all features 

graph.add_shared_node(Convolution2D(256,3,3,activation='relu',border_mode="valid"),name='combined_layer_total',inputs=['combined_layer_1n','combined_layer_1e','combined_layer_1m','combined_layer_1'],merge_mode='sum')
graph.add_node(Flatten(), name='flatten_layer',input='combined_layer_total')
graph.add_node(Dense(512,activation='relu'),name='combined_dense_layer_1',input='flatten_layer')
graph.add_node(Dense(512,activation='relu'),name='combined_dense_layer_2',input='combined_dense_layer_1')
graph.add_node(Dense(2,activation='softmax'),name='output_layer',input='combined_dense_layer_2')
graph.add_output(name='output1',input='output_layer')

graph.fit({'input1':XTrain1,'input2':XTrain2,'output1':y_train2},batch_size=64,nb_epoch=5)

graph.evaluate({'input1':XTest1,'input2':XTest2,'output1':y_test2}, batch_size=128, show_accuracy=True, verbose=0)
predictions=graph.predict({'input1':XTest1,'input2':XTest2})
predictions