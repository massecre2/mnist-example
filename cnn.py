# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:35:57 2017

@author: Oğuzhan
"""


from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

#   import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from numpy import * 
#SKLEARN
#   from sklearn.utils import shuffle 
from sklearn.cross_validation import train_test_split 
import theano

from keras import backend as K
K.set_learning_phase(1)

#____________________________________________________________________________________________________________________
import os
    
non_converted_image_file_path = 'C:\\Users\\Oğuzhan\\Desktop\\DeepLearningProject\\mnist\\data\\input_data'
converted_image_file_path = 'C:\\Humansdataresized'

#_____________________________________________________________________________________________________________________
from numpy import size

list_non_converted_images = os.listdir(non_converted_image_file_path) 
number_of_non_converted_images = size(list_non_converted_images)
    
#_____________________________________________________________________________________________________________________
from PIL import Image

img_rows , img_cols = 200 , 200

for file in list_non_converted_images:
    imop = Image.open(non_converted_image_file_path + '\\' + file)
    img = imop.resize((img_rows,img_cols))
    gray= img.convert('L')
        #When translating a color image to black and white (mode “L”), the library uses the ITU-R 601-2 luma transform:
        #L = R * 299/1000 + G * 587/1000 + B * 114/1000
    gray.save(converted_image_file_path +'\\'+ file, "JPEG")
    #-- file , img , img_cols, img_rows , imop , gray
#_____________________________________________________________________________________________________________________
list_of_converted_images = os.listdir(converted_image_file_path)
number_of_converted_images = size(list_of_converted_images)

if (number_of_converted_images != number_of_non_converted_images):
    print('NON EUQALS ' , number_of_converted_images , " = " , number_of_non_converted_images)
    raise SystemExit
else:
    print('EQUALS ' , number_of_converted_images , " = " , number_of_non_converted_images)
#_____________________________________________________________________________________________________________________
import numpy as np

imatrix = np.array([np.array(Image.open(converted_image_file_path+ '\\'+ im2)).flatten() for im2 in list_of_converted_images], 'f')

label = np.ones((number_of_non_converted_images,),dtype=int)
label[0:84] = 0 #cat
label[85:170] = 1 #kopek
label[170:] = 2 #human 

#_____________________________________________________________________________________________________________________
from sklearn.utils import shuffle 

shuffle_imatrix , shuffle_label = shuffle(imatrix,label,random_state=2) #shuffle_imatrix=data , shuffle_label = Label
train_data = [shuffle_imatrix,shuffle_label]

"""
import matplotlib.pyplot as plt 
img = imatrix[167].reshape(img_rows,img_cols)
plt.imshow(img)
plt.imshow(img,cmap='gray')
"""
print ("train data array[0] , train values : \n" ,train_data[0] , "\n")
print ("train data array[0] , shape train values : \n " ,train_data[0].shape ,"\n")

print ("train data array[1] , label values : \n" ,train_data[1] , "\n")
print ("train data array[1] , shape label values :\n ", train_data[1].shape, "\n")
#_____________________________________________________________________________________________________________________!TAMAM!
batch_size = 32 #batch size to train 
nb_classes = 3 # we have 3 class in non processing data folder. first of cat , second dog , thirt people.
nb_epoch = 1 # number of epoch to train 

img_rows , img_cols = 200, 200

img_channels = 1
nb_filters = 32 #convolutional filter 
nb_pool = 2 #pooling
nb_conv = 3 #conv.
#_____________________________________________________________________________________________________________________
from sklearn.cross_validation import train_test_split 

X_data_set , y_label_set = (train_data[0],train_data[1]) #train_data[0] = shuffle_imatrix , train_data[1] = shuffle_label

X_train, X_test, y_train , y_test = train_test_split(X_data_set , y_label_set , test_size=0.2 , random_state=4)

X_train = X_train.reshape(X_train.shape[0], 1 , img_rows , img_cols)#X_train.shape[0] = number of trained data , 1 = siyah beyaz olduğu için tek resim kanalı
X_test = X_test.reshape(X_test.shape[0], 1 , img_rows , img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255 

print('X_train shape :', X_train.shape)
print(X_train.shape[0] , 'train samples' )
print(X_test.shape[0] , 'test samples') 

from keras import utils
from keras.utils import np_utils

#Convert class vectors to binary class matrices 
Y_train = np_utils.to_categorical(y_train , nb_classes)
Y_test = np_utils.to_categorical(y_test , nb_classes)

i=100 
import matplotlib.pyplot as plt 
plt.imshow(X_train[i , 0] , interpolation='nearest')
print("label : " , Y_train[i,:])
#_____________________________________________________________________________________________________________________
from keras.models import Sequential 
from keras.layers.convolutional import MaxPooling2D

model = Sequential()

from keras.layers import Conv2D

model.add(Conv2D(nb_filters, (nb_conv , nb_conv) , 
                        activation = 'relu' , 
                        input_shape=(1,200,200), 
                        data_format='channels_first'))


model.add(Conv2D(nb_filters, (nb_conv , nb_conv) , 
                        activation = 'relu'))


model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy' , optimizer='adadelta')
#_____________________________+________________________________________________________________________________________

model.fit(X_test , Y_test , batch_size = batch_size , epochs = nb_epoch ,
           verbose = 1 , validation_data=(X_test , Y_test))

model.fit(X_train, Y_train , batch_size=batch_size , epochs = nb_epoch,
          verbose = 1 , validation_split = 0.2 )
#_____________________________________________________________________________________________________________________
score=model.evaluate(X_train , Y_train , verbose = 0)
print('Test score : ' , score)
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])
#_____________________________________________________________________________________________________________________
#%% 

#visualizing intermediate Layers

output_layer = model.layers[0].output 
output_fn = K.function([model.layers[0].input], [output_layer])
#_____________________________________________________________________________________________________________________

input_image= X_train[0:1,:,:,:]
print(input_image.shape)

plt.imshow(input_image[0,0,:,:], cmap='gray')
plt.imshow(input_image[0,0,:,:])


output_image = output_fn([input_image])
output_image = np.array([output_image])
output_image = output_image.reshape(1,32,198,198)
print(output_image.shape)


# Rearrnge dimension so we can plot the result as RGB images
output_image = np.rollaxis(np.rollaxis(output_image , 3 , 1) , 3 , 1)
print(output_image.shape)

fig = plt.figure(figsize=(8,8))
for i in range(32):
    ax = fig.add_subplot(6, 6, i+1)
    ax.imshow(output_image[0,:,:,i],interpolation='nearest') 
    ax.imshow(output_image[0,:,:,i],cmap=matplotlib.cm.gray)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
plt