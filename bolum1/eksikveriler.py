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
veriler = pd.read_csv('eksikveriler.csv')

print(veriler)

print (veriler[['boy']]) # boy column print edilmesi
print(type(veriler))

boyKilo = veriler[['boy','kilo']]
print(boyKilo)



# eksik veriler

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


#  
## kategorik verilerin encoding edilmesi ile numeric verilere cevirmek

ulke = veriler.iloc[:,0:1].values # values yazmazsak DataFrame, values ise Array

print(ulke) # burda orijinal datadan kesilen column tr,us,fr gibi

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke) # burda tek column da 0,1,2 gibi sayilar verilmis hali

ohe = preprocessing.OneHotEncoder() 
ulke = ohe.fit_transform(ulke).toarray()
print(ulke) # burda yan yana 3 column matrix oluyor artik


print(list(range(22)))
sonuc = pd.DataFrame(data=ulke , index =range(22), columns=['fr','tr','us'])
print(sonuc)


sonuc2 = pd.DataFrame(data=yas, index = range(22),columns =['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1]
print(cinsiyet)

sonucCin = pd.DataFrame(data = cinsiyet,index =range(22), columns = ['cinsiyet'])


# data frame birlestirmek
s=  pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2 = pd.concat([s,sonucCin], axis=1)


# verilerin egitim ve test icin bolunmesi
# model_selection
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, sonucCin, test_size=0.33, random_state=0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()


X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)