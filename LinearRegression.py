import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 공부시간에 따른 점수 표
df = pd.DataFrame({'Study time':[3,4,5,8,10,5,8,6,3,6,10,9,7,0,1,2],
		   'Score':[76,74,74,89,92,75,84,82,73,81,89,88,83,40,70,69]})

# 데이터셋 분리
train, test = train_test_split(df, test_size = 0.4, random_state = 2)
X_train = train[['Study time']]
y_train = train['Score']
X_test = test[['Study time']]
y_test = test['Score']

# LinearRegression 진행
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# 잔차 구하기
y_mean = np.mean(y_test) # y 평균값

# $\sum(y 예측값 - y 평균값)^2$ = 예측값에 대한 편차
nomerator = np.sum(np.square(y_test - y_pred)) 

# $sum(y 관측값 - y 평균값)^2$
denominator = np.sum(np.square(y_test - y_mean))
r2 = 1 - nomerator / denominator
r2
