from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

targetUrl = "https://raw.githubusercontent.com/gundaminpde/2022/main/sonar3.csv"

Data_set = pd.read_csv(targetUrl, sep=',')

x = Data_set.iloc[:,0:60]
y = Data_set.iloc[:,60]

print(Data_set.shape)

Data_set.haed()

x.head()

y.head()

model = Sequential()
model.add(Dense(50, input_dim=60, activation='relu'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
history=model.fit(x,y, epochs=1000, batch_size=60)

y_loss = history.history['loss']

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_loss)
