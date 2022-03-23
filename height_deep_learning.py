import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import load_model

import numpy as np
import tensorflow as tf


seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)

#학습시킬 데이터 가져오기
df = pd.read_csv("/content/Queen.csv")

dataset = df.values
#atttribute
X = dataset[:,0:14]
#class
Y = dataset[:,14]
#training set 과 validation set으로 나누기
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=seed)


#model setting
model =Sequential()
model.add(Dense(30,input_dim=14,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')
#Training
model.fit(X_train,Y_train,epochs=200,batch_size=10)


#model 저장
model.save("/content/gdrive/My Drive/deep_learning.h200")

#구글 드라이브 마운트하기
from google.colab import drive
drive.mount('/content/gdrive')


#학습 모델로 예측하기 
Y_prediction = model.predict(X_test).flatten()

#결과 데이터 프레임 만들기
result = pd.DataFrame(X_test,columns=["age","sex","array","pc1","pc2","pc3",
                                      "pc4","pc5","pc6","pc7","pc8","pc9","pc10",
                                      "pred_inf"])
#result[["age","sex"]].astype(int64)
#for i in range(35477):
 # resultX_test[i]
result["Predict"] = Y_prediction
result["Observe"]= Y_test
result.to_csv("/content/discovery_result.csv",index=False)



for i in range(177329):
  label = Y_test[i]
  prediction = Y_prediction[i]
  #print("관찰 값 {:.3f}, 예측 값: {:.3f}".format(label,prediction))
  print(prediction-label)
  
weights = model.get_weights()
print(weights[0].shape)
print(weights[1].shape)
print(weights[2].shape)
print(weights[3].shape)
print(weights[4].shape)
print(weights[5].shape)



print(weights[0])
zz = pd.DataFrame(weights[0])

#데이터 프레임 각 행의 평균 구하기 
zz.mean(axis=1)
