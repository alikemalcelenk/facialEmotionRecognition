#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:23:09 2020

@author: alikemalcelenk
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import *
import os
print(os.listdir("./input/ck"))

dataPath = './input/ck/CK+48'
data_dir_list = os.listdir(dataPath)
#print(data_dir_list) #['happy', 'contempt', 'fear', 'surprise', 'sadness', 'anger', 'disgust']

imgDataList=[]

for dataset in data_dir_list:
    img_list=os.listdir(dataPath+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    #print(img_list)  Btüün image datası
    for img in img_list:
        inputImg=cv2.imread(dataPath + '/'+ dataset + '/'+ img )
        inputImgResize=cv2.resize(inputImg,(48,48))
        #fotolar zaten 48x48. Emin olmak için resize yapıyorum
        imgDataList.append(inputImgResize)
    
imgData = np.array(imgDataList)
imgData = imgData/255  #Normalization

#Resimleri aldığıma dair bir tane örnek resim
plt.imshow(imgData[0],cmap='gray')
#resmi alıp gray scale yaptık. Zaten gray ama ne olur ne olmaz diye.
plt.axis("off")
# x ve y çizgilerini kapattık ki grafik gibi olmasın
plt.show()

num_classes = 7

num_of_samples = imgData.shape[0] #981  #(shape, dtype, order) arr nin içi o yüzden 0 ı alıyoruz. Kaç tane img datası oldugunu gösteriyor

labels = np.ones((num_of_samples), dtype='int64') #1lerden olusan 981 elemanlık array olustrudum
#print(labels) 

labels[0:134]=0 #135
labels[135:188]=1 #54
labels[189:365]=2 #177
labels[366:440]=3 #75
labels[441:647]=4 #207
labels[648:731]=5 #84
labels[732:981]=6 #249
#print(labels) 


from keras.utils.np_utils import to_categorical 
Y = to_categorical(labels, num_classes) 
#Çoklu sınıflandırma ile ilgilendiğimiz için etiketleri kategorik olarak etiketlememiz gerekiyor
#Elimizde bulunan sayıları(yani resimlerdeki sayıları 0 dan 6 ya kadar) encode ederek 
#başka formata cevirdik
#0 => [1,0,0,0,0,0,0,0,0,0]
#2 => [0,0,1,0,0,0,0,0,0,0]
#9 => [0,0,0,0,0,0,0,0,0,1]
#Burada bizim yaptığımız encoding, one-hot-encoding olarak geçer
#print(Y)

#Shuffle the dataset
x,y = shuffle(imgData, Y, random_state=2)
#Shuffle işlemi arrlerin içinde indexlerin yerlerini değiştiriyor. Burda imgData içindeki imageların sırasını rastgele değiştirip x e yazdık.
#Aynı şekilde Y içindekileri rastgele değiştirip y ye yazdık. 
#print(x)
#print(y)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)
#Şimdi burada x in bir kısmını test olarak alıcam ve uyg içinde test olarak onu kullanıcam.
# test_size = datasetimin %85 train %15 validation a ayır demek.
#random_state ise yapılan işlemin belli bir sırada yapılmasını sağlıyor.  

input_shape=(48,48,3)
#png datayı direk kullandığımız için 3 kanalı var. digitRecognition da 1 demiştim çünkü ordaki datalar png değil csv içindeydi.

model = Sequential() #modeli oluşturduk.
#1
model.add(Conv2D(filters = 6, kernel_size = (5, 5), input_shape = input_shape, padding='Same', activation = 'relu'))
#filters = convolution layerdaki filtre sayısı
#kernel_size = filtrenin boyutu. 
#SamePadding kullanıyoruz(kenarlara 0 koyarak, input size = output size).
#Activation func. = relu
model.add(MaxPooling2D(pool_size=(2, 2)))

#2
model.add(Conv2D(filters = 16, kernel_size = (5, 5), padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#3
model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


#Flatten -> matrixi tek sütun haline getirme işlemi. ANN için 
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))

#Adam Optimizer
#Learning rateimiz normalde  sabittir. Adam Optimizer kullanarak değiştirebiliyoruz.
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#loss functionı categorical_crossentropy ile buluyoruz. Eğer yanlış predict ederse loss yüksek, doğru predict ederse loss 0.
model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#Fit the model
hist = model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_test, y_test))
#Epoch and Batch Size
#981 resmimiz var. batch__size = 64 dedim.  981/64 = 15,33 kez batch yaparız.
#Bu da epoch olarak adlandırılır. Her epochta 15,33 kez batch yapıyoruz demek.

#Evaluate model
score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])





