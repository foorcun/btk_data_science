# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 21:41:31 2023

@author: msnif
"""

# 1 # libraries
import pandas as pd # Read data
import numpy as np
import matplotlib.pyplot as plt



#kodlar
# print(234)


# 2 # veri on islemece
# 2.1 # veri yuklemece
veriler = pd.read_csv('satislar.csv')

print(veriler)


# veri on islemece

# bagimsiz degisken
aylar = veriler[['Aylar']]

# bagimli degisken
satislar = veriler[['Satislar']]



# verilerin egitim ve test icin bolunmesi
# model_selection
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()


X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)


# modelling
from sklearn.linear_model import LinearRegression

lr = LinearRegression()


#lr.fit(X_train,Y_train) # with StandardScaler
lr.fit(x_train,y_train) # without StandardScaler


tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))