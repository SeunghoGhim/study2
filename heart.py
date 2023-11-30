import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Url_B_cancer="https://raw.githubusercontent.com/SeunghoGhim/study2/main/heart.csv"

df = pd.read_csv(Url_B_cancer)

df.head()

df.tail(n=10)

df=df.drop([1],axis=0)

df.head()

df.isnull().sum()

y_original = df['target']

y_original.head()

y_original.tail()

y_original.unique()

y_original.value_counts()

y_original_2=pd.get_dummies(y_original)

y_original_2.head()

x_train, x_test, y_train, y_test = train_test_split(
    df.iloc[:,1:], y_original_2['B'],stratify=y_original_2['B'], random_state=3)

x_train.head()

std = StandardScaler()

std.fit(x_train.iloc[:,0:])

x_train.iloc[:,0:]= std.transform(x_train.iloc[:,0:])

x_train.head()

x_train.describe()

x_test.iloc[:,0:]= std.transform(x_test.iloc[:,0:])

x_test.head()

x_test.describe()

y_train.describe()

y_test.describe()

df_cat = pd.concat([x_train,y_train], axis=1)

corr_mat=np.corrcoef(df_cat,rowvar=False)

sns.heatmap(corr_mat,linewidth=1,cmap='RdYlGn_r')
plt.show()

df_cat.head()

x_train.shape

sns.pairplot(df_cat.iloc[:,21:], hue='B');
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()                                                
model.add(Dense(13, input_dim=13, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])  # 딥러닝 모델을 실행합니다.

historyA=model.fit(x_train, y_train, epochs=180, batch_size=15)

y_loss = historyA.history['loss']
y_accu = historyA.history['binary_accuracy']

x_len = np.arange(len(y_loss))

plt.plot(x_len,y_loss)
plt.plot(x_len,y_accu, marker='.', c="red")

plt.show()

A=x_test.iloc[3,0:].to_numpy()

B=A.reshape(1, 30)

model.predict(B,verbose=0)#[:10]

model.predict(x_test,verbose=0)[:10]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random

random.seed(5)

model = Sequential()                    
model.add(Dense(13, input_dim=13, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])  # 딥러닝 모델을 실행합니다.

history=model.fit(x_train, y_train, validation_split=0.25, epochs=100, batch_size=15, verbose=1)

y_val_loss = history.history['val_loss']

plt.plot(x_len,y_loss, marker='.', c="blue", label='Train-set loss')
plt.plot(x_len,y_val_loss, marker='.', c="red",label='Validation-set loss')

plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

y_accu = history.history['binary_accuracy']
y_val_accu = history.history['val_binary_accuracy']

plt.plot(x_len,y_accu, marker='.', c="blue", label='Train-set accuracy')
plt.plot(x_len,y_val_accu, marker='.', c="red",label='Validation-set accuracy')

plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

model.evaluate(x_test,y_test)
