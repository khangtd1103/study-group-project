# %% [markdown]
#region title
# # Gesture Recognition
# In this group project, you are going to build a 3D Conv model that will be able to predict the 5 gestures correctly. Please import the following libraries to get started.
#endregion title
# %%
#region import
import numpy as np
import os

# from scipy.misc import imread, imresize 
# ImportError: cannot import name 'imread' from 'scipy.misc'
from scipy.ndimage import zoom
# ImportError: cannot import name 'imresize' from 'scipy.misc

import imageio
import datetime
import os
#endregion import

# %% [markdown]
#region explain_random_seed
# We set the random seed so that the results don't vary drastically.
#endregion explain_random_seed

# %%
#region random_seed
np.random.seed(30)
import random as rn
rn.seed(30)
from keras import backend as K
import tensorflow as tf
# tf.set_random_seed(30) #AttributeError: module 'tensorflow' has no attribute 'set_random_seed
tf.random.set_seed(30)
#endregion random_seed

# %% [markdown]
#region explain_read_folders
# In this block, you read the folder names for training and validation. You also set the `batch_size` here. Note that you set the batch size in such a way that you are able to use the GPU in full capacity. You keep increasing the batch size until the machine throws an error.

#endregion explain_read_folders
# %%
#region read_folders
train_doc = np.random.permutation(open('Project_data/train.csv').readlines())
val_doc = np.random.permutation(open('Project_data/val.csv').readlines())
batch_size = 32 #experiment with the batch size
#endregion read_folders

# %%
#region set_folder_path
train_path = "Project_data/train"
folders = [f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path,f))]
#endregion set_folder_path
# %% [markdown]
#region generator_title
# ## Generator
# This is one of the most important part of the code. The overall structure of the generator has been given. In the generator, you are going to preprocess the images as you have images of 2 different dimensions as well as create a batch of video frames. You have to experiment with `img_idx`, `y`,`z` and normalization such that you get high accuracy.
#endregion generator_title
 
# %% [markdown]
# %%
#region generator_test
def generator(source_path, folder_list, batch_size):
    print( 'Source path = ', source_path, '; batch size =', batch_size)
    # img_idx =  #create a list of image numbers you want to use for a particular video
    t = np.random.permutation(folder_list)
    num_batches = len(folder_list)/batch_size # calculate the number of batches
    print("num_batches = ", num_batches)
    print("range(round(num_batches)) = ",range(round(num_batches)))
    # create empty containers for the batch
    counter=0
    print("\npermutated folder_list, first 5 elements t[:5]: \n", t[:5],"\n")
    for batch in range(round(num_batches)): # we iterate over the number of batches
        # batch_data = np.zeros((batch_size,30,100,100,3)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
        # batch_labels = np.zeros((batch_size,5)) # batch_labels is the one hot representation of the output
        if counter==5:break
        for folder in range(batch_size): # iterate over the batch_size
            print(f"for folder in range(batch_size): print ( t[folder {folder} + (batch {batch} * batch_size {batch_size})]):", t[folder + (batch*batch_size)])
            if counter==5:break
            counter+=1
            
            # imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')

generator(train_path, folders, 10)                
#endregion generator_test
# %% [markdown]
# %%
# %%
#region generator_sample
def generator(source_path, folder_list, batch_size):
    print( 'Source path = ', source_path, '; batch size =', batch_size)
    img_idx =  #create a list of image numbers you want to use for a particular video
    while True:
        t = np.random.permutation(folder_list)
        num_batches = len(folder_list)/batch_size # calculate the number of batches
        print(num_batches)
        # create empty containers for the batch
        for batch in range(num_batches): # we iterate over the number of batches
            batch_data = np.zeros((batch_size,x,y,z,3)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((batch_size,5)) # batch_labels is the one hot representation of the output
            for folder in range(batch_size): # iterate over the batch_size
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate over the frames/images of a folder to read them in
                    image = imageio(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    
                    #crop the images and resize them. Note that the images are of 2 different shape 
                    #and the conv3D will throw error if the inputs in a batch have different shapes
                    
                    batch_data[folder,idx,:,:,0] = #normalise and feed in the image
                    batch_data[folder,idx,:,:,1] = #normalise and feed in the image
                    batch_data[folder,idx,:,:,2] = #normalise and feed in the image
                    
                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels #you yield the batch_data and the batch_labels, remember what does yield do

        
        # write the code for the remaining data points which are left after full batches

#region generator_sample
# %% [markdown]
# Note here that a video is represented above in the generator as (number of images, height, width, number of channels). Take this into consideration while creating the model architecture.

# %%
curr_dt_time = datetime.datetime.now()
train_path = '/notebooks/storage/Final_data/Collated_training/train'
val_path = '/notebooks/storage/Final_data/Collated_training/val'
num_train_sequences = len(train_doc)
print('# training sequences =', num_train_sequences)
num_val_sequences = len(val_doc)
print('# validation sequences =', num_val_sequences)
num_epochs = # choose the number of epochs
print ('# epochs =', num_epochs)

# %% [markdown]
# ## Model
# Here you make the model using different functionalities that Keras provides. Remember to use `Conv3D` and `MaxPooling3D` and not `Conv2D` and `Maxpooling2D` for a 3D convolution model. You would want to use `TimeDistributed` while building a Conv2D + RNN model. Also remember that the last layer is the softmax. Design the network in such a way that the model is able to give good accuracy on the least number of parameters so that it can fit in the memory of the webcam.

# %%
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers

#write your model here

# %% [markdown]
# Now that you have written the model, the next step is to `compile` the model. When you print the `summary` of the model, you'll see the total number of parameters you have to train.

# %%
optimiser = #write your optimizer
model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model.summary())

# %% [markdown]
# Let us create the `train_generator` and the `val_generator` which will be used in `.fit_generator`.

# %%
train_generator = generator(train_path, train_doc, batch_size)
val_generator = generator(val_path, val_doc, batch_size)

# %%
model_name = 'model_init' + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'
    
if not os.path.exists(model_name):
    os.mkdir(model_name)
        
filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

LR = # write the REducelronplateau code here
callbacks_list = [checkpoint, LR]

# %% [markdown]
# The `steps_per_epoch` and `validation_steps` are used by `fit_generator` to decide the number of next() calls it need to make.

# %%
if (num_train_sequences%batch_size) == 0:
    steps_per_epoch = int(num_train_sequences/batch_size)
else:
    steps_per_epoch = (num_train_sequences//batch_size) + 1

if (num_val_sequences%batch_size) == 0:
    validation_steps = int(num_val_sequences/batch_size)
else:
    validation_steps = (num_val_sequences//batch_size) + 1

# %% [markdown]
# Let us now fit the model. This will start training the model and with the help of the checkpoints, you'll be able to save the model at the end of each epoch.

# %%
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)

# %%



