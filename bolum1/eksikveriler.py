# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 21:41:31 2023

@author: msnif
"""

# libraries
import pandas as pd # Read data
import numpy as np
import matplotlib.pyplot as plt



#kodlar
# print(234)

# veri yuklemece

veriler = pd.read_csv('eksikveriler.csv')

print(veriler)

print (veriler[['boy']]) # boy column print edilmesi
print(type(veriler))

boyKilo = veriler[['boy','kilo']]
print(boyKilo)

# veri on isleme

from sklearn.impute import SimpleImputer

from enum import Enum

class ImputerStrategies(Enum):
    MEAN ="mean"


imputer = SimpleImputer(missing_values=np.nan,strategy= ImputerStrategies.MEAN.value)


yas = veriler.iloc[:,1:4].values # 1 dahil 4 haric sutunlar

imputer = imputer.fit(yas[:,1:4]) # fit = ogren. burda ortalamasini ogren
print(imputer.fit(yas[:,1:4]))

yas[:,1:4] = imputer.transform(yas[:,1:4]) # fitte ogrendigini Uygula

print(yas)


## kategorik verilerin encoding edilmesi

ulke = veriler.iloc[:,0:1].values # values yazmazsak DataFrame, values ise Array

print(ulke) # burda orijinal datadan kesilen column tr,us,fr gibi

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke) # burda tek column da 0,1,2 gibi sayilar verilmis hali

ohe = preprocessing.OneHotEncoder() 
ulke = ohe.fit_transform(ulke).toarray()
print(ulke) # burda yan yana 3 column matrix oluyor artik




