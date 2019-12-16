import numpy as np
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],28,28,1).astype('float32')/255
x_test=x_test.reshape(x_test.shape[0],28,28,1).astype('float32')/255

from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

from keras.models import Sequential#序贯模型，线性组织不同神经网络
model=Sequential()#模型构建

from keras.layers import Activation,Dropout,Flatten,Dense
from keras.layers.convolutional import Conv2D,MaxPooling2D
model.add(Conv2D(20,(5,5),input_shape=(28,28,1)))
model.add(Activation('relu'))#relu激活函数
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))#第一次Dropout
model.add(Conv2D(50,(5,5)))
model.add(Activation('relu'))#relu激活函数
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))#第二次Dropout
model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))#relu激活函数
model.add(Dense(10))
model.add(Activation('softmax'))

from keras.optimizers import RMSprop
model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=200,verbose=2)
score=model.evaluate(x_test,y_test,verbose=0)
print('loss',score[0])#0.0286
print('accu',score[1])#0.99以上