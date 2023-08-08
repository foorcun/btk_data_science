# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# libraries
import pandas as pd # Read data
import numpy as np
import matplotlib.pyplot as plt



#kodlar
# print(234)

# veri yuklemece

veriler = pd.read_csv('veriler.csv')

print (veriler[['boy']]) # boy column print edilmesi
print(type(veriler))

boyKilo = veriler[['boy','kilo']]
print(boyKilo)

# veri on isleme