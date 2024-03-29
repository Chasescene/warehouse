from keras.datasets import mnist#由于兼容性问题，在jupyter中使用需从tensorflow调用keras，如tensorflow.python.keras.datasets
(x_train,y_train),(x_test,y_test)=mnist.load_data()#数据集下载，并定义x和y
x_train=x_train.reshape(60000,28*28).astype('float32')/255#修改输入形状、类型、正规化
x_test=x_test.reshape(10000,28*28).astype('float32')/255

from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train,num_classes=10)#转换为0-1编码输出类别标签
y_test=np_utils.to_categorical(y_test,num_classes=10)

from keras.models import Sequential#序贯模型，线性组织不同神经网络
from keras.layers import Dense, Activation, Dropout
model=Sequential()#模型构建
model.add(Dense(100,input_shape=(28*28,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

from keras.optimizers import RMSprop#比SGD优化效果更好，可选Adam优化器
model.compile(loss='categorical_crossentropy',optimizer=RMSprop (),metrics=['accuracy'])#模型装配，装配模型训练部分。交叉熵损失函数，RMSprop优化器，评价准确度。交叉熵衡量预测概率分布准确度。
model.fit(x_train,y_train,batch_size=128,epochs=10)#模型训练。batch_size是优化器进行权重更新前要观察的训练样例数。每次训练epoch迭代中，优化器调整权重，损失函数最小化。epoch=200时拟合效果最好
score=model.evaluate(x_test,y_test,verbose=1)#模型评估准确度，verbose为日志显示，1为显示进度条
print("Test score:",score[0])
print("Test accuracy:",score[1])#epoch=10时，testscore=0.0872，testaccuracy=0.9772