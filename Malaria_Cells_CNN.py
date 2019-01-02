import os 
import cv2
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Activation
import random 
import numpy as np


img_dir="cell_images"  
img_size=70
def create_training(path):
    training=[]
    for category in os.listdir(path):
        path_img=os.path.join(path,category) 
        num_class=os.listdir(path).index(category) 
        for img in os.listdir(path_img):
            try:
                img_arr=cv2.imread(os.path.join(path_img,img),cv2.IMREAD_GRAYSCALE) 
                img_new=cv2.resize(img_arr,(img_size,img_size)) 
                training.append([img_new,num_class])
            except Exception as e:
                pass
    return training  
data_training=create_training(img_dir)
andom.shuffle(data_training)    
x=[]
y=[]
for i,j in data_training:
    x.append(i)
    y.append(j)  
x=np.array(x).reshape(-1,img_size,img_size,1)    
x=x/255 
model=Sequential() 
model.add(Conv2D(64,(3,3),input_shape=x.shape[1:])) 
model.add(Activation("relu")) 
model.add(MaxPool2D(pool_size=(2,2))) 

model.add(Conv2D(64,(3,3))) 
model.add(Activation("relu")) 
model.add(MaxPool2D(pool_size=(2,2))) 


model.add(Conv2D(64,(3,3))) 
model.add(Activation("relu")) 
model.add(MaxPool2D(pool_size=(2,2)))  

model.add(Flatten()) 

model.add(Dense(64))
model.add(Activation("relu")) 

model.add(Dense(64))
model.add(Activation("relu")) 

model.add(Dense(1)) 
model.add(Activation("sigmoid")) 

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"]) 
model.fit(x,y,batch_size=64,epochs=3,validation_split=0.2)

model.save("malaria-cell-cnn.model")