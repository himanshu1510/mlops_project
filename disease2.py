#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D


# In[2]:


from keras.layers import MaxPooling2D


# In[3]:


from keras.layers import Flatten


# In[4]:


from keras.layers import Dense


# In[5]:


from keras.models import Sequential


# In[6]:


model = Sequential()


# In[7]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))


# In[8]:


model.summary()


# In[9]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[10]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))


# In[11]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:





# In[12]:


model.summary()


# In[13]:


model.add(Flatten())


# In[14]:


model.summary()


# In[15]:


model.add(Dense(units=128, activation='relu'))


# In[16]:


model.summary()


# In[17]:


model.add(Dense(units=1, activation='sigmoid'))


# In[18]:


model.summary()


# In[19]:


from keras_preprocessing.image import ImageDataGenerator


# In[20]:


model.compile(optimizer= 'Adam' , loss ='binary_crossentropy' , metrics=['accuracy'])


# In[23]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        '/root/workstation/malaria/cell_image_train/',
        target_size=(64, 64),
        batch_size=6,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        '/root/workstation/malaria/cell_image_test/',
        target_size=(64, 64),
        batch_size=6,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=27560,
        epochs=1,
        validation_data=test_set,
        validation_steps=200)


# In[87]:


acc=model.history


# In[1]:


acc.history


# In[4]:


accuracy=acc.history['val_accuracy'][0]


# In[ ]:


accuracy=int(accuracy*100)
f=open("accuracy.txt","w+")
f.write(str(accuracy))
f.close()
print("Accuracy is:" ,accuracy , "%")


# In[2]:


model.save('disease.h5')
