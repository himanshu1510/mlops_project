#!/usr/bin/env python
# coding: utf-8

# In[17]:


from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Model


# In[18]:


model = load_model('disease.h5')


# In[19]:


model.layers[0].input


# 

# In[20]:


top_model=model.layers[-2].output


# In[25]:


top_model=Dense(units=64,
           activation='relu'
           )(top_model)
top_model=Dense(units=1,
           activation='sigmoid'
           )(top_model)
model=Model(input =model.input,output=top_model)


# In[26]:


model.layers


# In[27]:


model.summary()


# In[ ]:


from keras_preprocessing.image import ImageDataGenerator
model.compile(optimizer= 'Adam' , loss ='binary_crossentropy' , metrics=['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'C:/Users/hp19tu/Desktop/project/cell_images/malaria/cell_image_train/',
        target_size=(64, 64),
        batch_size=6,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'C:/Users/hp19tu/Desktop/project/cell_images/malaria/cell_image_test/',
        target_size=(64, 64),
        batch_size=6,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=27358,
        epochs=1,
        validation_data=test_set,
        validation_steps=200)
acc=model.history
accuracy=acc.history['val_accuracy'][0]
accuracy=accuracy*100
accuracy


# In[ ]:




