# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 11:21:58 2020

@author: Long
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# doc file
d1=pd.read_csv("student-mat.csv",sep=";")
#d2=pd.read_csv("student-por.csv",sep=";")
#d1=pd.concat([d1,d2])
# sửa dữ liệu chữ thành dang int

d1["school"]=d1["school"].replace({"GP": "0","MS": "1"})
d1["sex"]=d1["sex"].replace({"M": "0","F": "1"})
d1["address"]=d1["address"].replace({"U": "0","R": "1"})
d1["famsize"]=d1["famsize"].replace({"LE3": "0","GT3": "1"})
d1["Pstatus"]=d1["Pstatus"].replace({"T": "0","A": "1"})
d1["Mjob"]=d1["Mjob"].replace({"at_home": "0","health": "1","services":"2","teacher":"3","other":"4"})
d1["Fjob"]=d1["Fjob"].replace({"at_home": "0","health": "1","services":"2","teacher":"3","other":"4"})
d1["reason"]=d1["reason"].replace({"course": "0","home": "1","reputation": "2","other": "3"})
d1["guardian"]=d1["guardian"].replace({"father": "0","mother": "1","other": "2"})
d1["schoolsup"]=d1["schoolsup"].replace({"no": "0","yes": "1"})
d1["famsup"]=d1["famsup"].replace({"no": "0","yes": "1"})
d1["paid"]=d1["paid"].replace({"no": "0","yes": "1"})
d1["activities"]=d1["activities"].replace({"no": "0","yes": "1"})
d1["nursery"]=d1["nursery"].replace({"no": "0","yes": "1"})
d1["higher"]=d1["higher"].replace({"no": "0","yes": "1"})
d1["internet"]=d1["internet"].replace({"no": "0","yes": "1"})
d1["romantic"]=d1["romantic"].replace({"no": "0","yes": "1"})

###################
data = d1.drop(columns=['G3'])
#lay 80% train và 20% test
train =int(len(data) * 0.8)
test = int(len(data) * 0.2)
data_train = data.loc[:train]
data_test = data.loc[train+1:]
#print(data_train)
#print(data_test)
# gan label train va test
labels = d1['G3']
labels_train = labels.loc[:train]
labels_test = labels.loc[train+1:]
#print(labels_train)
#print(labels_test)
##################
## khoi tao mo hinh LinearRegression
model = LinearRegression()
#fit mo hinh
model.fit(data_train, labels_train)
#print(model.coef_)
#print(model.intercept_)
#Diem cua model
print(model.score(X=data_train, y=labels_train))
#Ham du doan ket qua
def myfuntion(x):
    x = model.predict(x)
    df = pd.DataFrame({'KetQua':x})   
    for i in range(df.size):
        if(df['KetQua'][i]>=10):
            df['KetQua'][i] = 'dau'
        else:
            df['KetQua'][i] = 'rot'
    return df
# test 
print(myfuntion(data_test))
# voi diem lon hon bang 10 thi dau va nho hon 10 thi rot
print(labels_test)
