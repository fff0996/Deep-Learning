#내 구글드라이브 마운트 하기 
from google.colab import drive
drive.mount('/content/gdrive')


import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.models import load_model

seed = 0
np.random.seed(seed)
tf.random.set_seed(3)
df = pd.read_csv("/content/gdrive/MyDrive/DL/00_input_DL_BMI_grid.csv")


dataset = df.values
X = dataset[:,0:14]
Y = dataset[:,14]

#X_train = X
#Y_train = Y 
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=seed)


#학습모델 세팅하기 
model =Sequential()
model.add(Dense(12,input_dim=14,activation='relu'))
model.add(Dense(14,activation='relu'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=10,batch_size=10)
print("\n Test Accurcy: %.4f" % (model.evaluate(X_test,Y_test)))
model.save("/content/gdrive/My Drive/deep_learning_bmi_obesity_8:2.h200")
print("\n Test Accurcy: %.4f" % (model.evaluate(X_train,Y_train)[1]))
print("\n Test Accurcy: %.4f" % (model.evaluate(X_test,Y_test)))
