# %% [markdown]
#region title[markdown]
# # Gesture Recognition
# In this group project, you are going to build a 3D Conv model that will be able to predict the 5 gestures correctly. Please import the following libraries to get started.
#endregion title[markdown]
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

#tag user_add
import matplotlib.pyplot as plt
#endregion import

# %% [markdown]
#region explain_random_seed[markdown]
# We set the random seed so that the results don't vary drastically.
#endregion explain_random_seed[markdown]

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
#region explain_read_folders[markdown]
# In this block, you read the folder names for training and validation. You also set the `batch_size` here. Note that you set the batch size in such a way that you are able to use the GPU in full capacity. You keep increasing the batch size until the machine throws an error.

#endregion explain_read_folders[markdown]
# %%
#region read_folders
#tag info # csv files contains folders' data
train_doc = np.random.permutation(open('Project_data/train.csv').readlines())
train_dir = 'Project_data/train'
val_doc = np.random.permutation(open('Project_data/val.csv').readlines())
val_dir = 'Project_data/val'
batch_size = 10 #experiment with the batch size
#endregion read_folders

# %% [markdown]
#region generator_title[markdown]
# ## Generator - Sample Function
# This is one of the most important part of the code. The overall structure of the generator has been given. In the generator, you are going to preprocess the images as you have images of 2 different dimensions as well as create a batch of video frames. You have to experiment with `img_idx`, `y`,`z` and normalization such that you get high accuracy.
#endregion generator_title[markdown]
# %% 
%%script echo "skipped: sample generator" 
#region generator_sample
import math
def generator(source_path, folder_list, batch_size):
    print( 'Source path = ', source_path, '; batch size =', batch_size)
    img_idx = #create a list of image numbers you want to use for a particular video
    while True:
        t = np.random.permutation(folder_list)
        num_batches = # calculate the number of batches
        print(num_batches)
        # create empty containers for the batch
        for batch in range(math.floor(num_batches)): # we iterate over the number of batches
            batch_data = np.zeros((batch_size,x,y,z,3)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((batch_size,5)) # batch_labels is the one hot representation of the output
            for folder in range(batch_size): # iterate over the batch_size
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate over the frames/images of a folder to read them in
                    image = imageio(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    
                    #crop the images and resize them. Note that the images are of 2 different shape 
                    #and the conv3D will throw error if the inputs in a batch have different shapes
                    #tag explain #batch_data
                    ''' 
                    - Use different image values for different channels => normalize each image's channel separately
                    - ":,:," means loading the image into the entire 2D space of the tensor at the specific location 
                    '''
                    batch_data[folder,idx,:,:,0] = #normalise and feed in the image
                    batch_data[folder,idx,:,:,1] = #normalise and feed in the image
                    batch_data[folder,idx,:,:,2] = #normalise and feed in the image
                    

                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels #you yield the batch_data and the batch_labels, remember what does yield do

        
        # write the code for the remaining data points which are left after full batches

#endregion generator_sample
# %% [markdown]
# ## Data Preparation     
# %% [markdown]
# ### Get all image paths
# %% [markdown]
# #### Function
# %%
#region get_image_paths[function]
# get the paths to all the images in a given directory
def get_image_paths(data_dir):
    # Get a list of folder names in the data directory
    folder_names = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,f))]
    # Create a list of paths to each folder
    folder_paths = [os.path.join(data_dir,folder_names[i]) for i in range(len(folder_names))]
    # Create a list of paths to each image
    img_paths = [os.path.join(folder_path, img) for folder_path in folder_paths for img in os.listdir(folder_path)]
    return folder_paths, img_paths
#endregion get_image_paths[function]
# %% [markdown]
# #### Execution
# %%
#region get_train_image_paths
# Get the paths to all the images in the training directory
train_folder_paths, train_img_paths = get_image_paths(train_dir)
#endregion get_train_image_paths
# %% [markdown]
# ### Get unique sizes
# %% [markdown]
# #### Function: get image size
# %%
#region get_all_image_sizes
from PIL import Image
# Define a function to get the size of an image file
def get_image_size(img_path):
  # Open the image file
  img = Image.open(img_path)
  # Get the image dimensions
  width, height = img.size
  # Return the dimensions as a tuple
  return (height, width)
# %% [markdown]
# #### Execution: get unique sizes of all images
# %%
# Create an empty set to store the unique image sizes
sizes = set()
# Iterate over the list of image paths
for image_path in train_img_paths:
    # Get the size of the image
    size = get_image_size(image_path)
    # Add the size to the set
    sizes.add(size)

print("Unique sizes of images: ",sizes)

#endregion get_all_image_sizes
# %% [markdown]
# ### Get unique image count per folder

# %%
#region uniq_frame_num
# Initialize a dictionary to store the number of images per folder
uniq_num_imgs_per_folder = set()

# Iterate through each folder path
for folder_path in train_folder_paths:
    # Get the list of image file names in the folder
    image_files = os.listdir(folder_path)

    # Add the number of images to the dictionary
    uniq_num_imgs_per_folder.add(len(image_files))

# Print the number of images per folder
print("Unique image count per folder:", uniq_num_imgs_per_folder)
#endregion uniq_frame_num
# %% [markdown]
# ### Group images by size
# %%
#region group_images_by_size
# Define a function to group images by size
def group_images_by_size(folder_paths):
    # Create an empty dictionary to store the grouped images
    grouped_images = {}
    # Iterate over the list of image paths
    for folder in folder_paths:
        # Get the first image from current folder
        sample_image_path = os.path.join(folder, os.listdir(folder)[0])
        # Get the size of the current image
        size = get_image_size(sample_image_path)
        # Add image to grouped images dictionary
        grouped_images.setdefault(size, []).append((sample_image_path, folder, size))

    # Return the grouped images dictionary
    return grouped_images

train_grouped_imgs = group_images_by_size(train_folder_paths)

dict_lengths = {key: len(value) for key, value in train_grouped_imgs.items()}
print("[train_grouped_imgs] Number of images for each size:", str(dict_lengths).replace('{', '').replace('}', ''))
#endregion group_images_by_size
# %% [markdown]
#region init_gen_params
# ## Initialize Generator Parameters 
# %%

# init_gen_param
def init_gen_params(num_frames, y, z):
    # Initialize img_idx using num_frames
    img_idx = [int(num) for num in np.linspace(0,29,num_frames)]
    # Define x,y,z
    x, y, z = len(img_idx), 120, 120
    return img_idx, x, y, z
img_idx, x, y, z = init_gen_params(7, 120, 120)

print("img_idx =", img_idx)
print("x =", x)
print("y =", y)
print("z =", z)
#endregion init_gen_params
# %% [markdown]
# ## Data Preprocessing
# %% [markdown]
# ### Function: crop images to square dimensions 
# %%
from skimage.util import crop # import to use crop() function
#region crop_image[function]
## Crop the image to a square shape with dimensions equal to the minimum of its width and height
def crop_image(origin_image, height_width):
    # If the width and height are equal, no cropping is needed
    if min(height_width) == max(height_width):
        top_bottom = left_right = (0, 0)
    else:
        # If the width is greater than the height, crop the top and bottom
        top_bottom = (int((max(height_width) - min(height_width)) / 2), int((max(height_width) - min(height_width)) / 2)) if height_width.index(max(height_width))==0 else (0, 0)
        # If the height is greater than the width, crop the left and right
        left_right = (0, 0) if height_width.index(max(height_width)) == 0 else (int((max(height_width) - min(height_width)) / 2), int((max(height_width) - min(height_width)) / 2))
    
    # Define the cropping dimensions
    crop_width = (top_bottom, left_right, (0,0))
    # Crop the image
    image_cropped = crop(origin_image, crop_width)
    # Return the cropped image
    return image_cropped
#endregion crop_image[function]
# %% [markdown]
# ### Experiments
# %% [markdown]
# #### test resize + crop
# %%
#%%script echo "skipped: open later"
#region test_resize
# Test the resize and crop for both group of image sizes
from skimage.transform import resize # import to use resize() function

# Get a dictionary of 5 sample images from each group
sample_imgs_by_size = {key:values[:5] for key, values in train_grouped_imgs.items()}

# For each group of image sizes, resize and crop the first image in the group
for key, value in sample_imgs_by_size.items():
    # Initialize the figure and subplots
    fig, axs = plt.subplots(nrows=2, ncols=len(value), figsize=(12, 5))
    # Add a title to the figure
    fig.suptitle(f"Size={key}")  

    # Loop through each image in the current group
    for value_idx, value in enumerate(sample_imgs_by_size[key]):

        # Display original image in the first row
        image_orig = imageio.v3.imread(value[0])
        axs[0, value_idx].imshow(image_orig)
        axs[0, value_idx].axis('off')
        axs[0, value_idx].set_title("Original")

        # Crop the resized image to a square shape
        image_cropped = crop_image(image_orig, key)
        # Resize the image 
        image_resized = resize(image_cropped, (120, 120),preserve_range=True, mode='reflect')

        # Display cropped and resized image in the second row
        image_resized = image_resized/255.0
        axs[1, value_idx].imshow(image_resized)
        axs[1, value_idx].axis('off')
        axs[1, value_idx].set_title(f"Resized\n{image_resized.shape[:2]}")

    # Show the figure
    plt.show()
#endregion test_resize
# %% [markdown]
# ### Test image normalization
# %% [markdown]
# #### Normalization functions 
# %%
#region test_image_normalization
import numpy as np # import to use np.clip()
random_folder = np.random.permutation(train_folder_paths)[0]
sample_imgs = sorted([os.path.join(random_folder, image_name.decode()) for image_name in os.listdir(random_folder)])
img_idx = [x-1 if x!=0 else x for x in range(0, 31, 5)]  #create a list of image numbers you want to use for a particular video
selected_imgs = [sample_imgs[idx] for idx in img_idx]
    
# Extract the red, green, and blue channels
def ext_rgb(image):
    image = image/255.0
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    return image, r, g, b

def create_img(args):
    # Create a new image with the normalized channels
    norm_image = np.zeros_like(args[0])
    norm_image[:, :, 0] = args[1]
    norm_image[:, :, 1] = args[2]
    norm_image[:, :, 2] = args[3]
    return np.clip(norm_image, 0, 1)

# inspired from https://akash0x53.github.io/blog/2013/04/29/RGB-Normalization/
def ratio_norm (args):
    # Calculate the ratio of the red, green, and blue channels
    sum_rgb = args[1] + args[2] + args[3]
    # Avoid division by zero
    sum_rgb[sum_rgb == 0] = 1

    r_norm = args[1] / sum_rgb 
    g_norm = args[2] / sum_rgb 
    b_norm = args[3] / sum_rgb 

    return args[0], r_norm, g_norm, b_norm   

# inspired from https://groups.google.com/g/comp.soft-sys.matlab/c/nXMVvJ6OhZY?pli=1
def vector_norm (args):
    # Calculate the normalized pixel intensity of the image
    pixel_vector_norm = np.sqrt(args[1]**2 + args[2]**2 + args[3]**2)
    # Avoid division by zero
    pixel_vector_norm[pixel_vector_norm==0] = 1
    # pixel_vector_norm = np.where(pixel_vector_norm == 0, 1, pixel_vector_norm)
    # Normalize the vector to have a magnitude of 1
    r_norm = args[1] / pixel_vector_norm 
    g_norm = args[2] / pixel_vector_norm 
    b_norm = args[3] / pixel_vector_norm 

    return args[0], r_norm, g_norm, b_norm

def std_mean(args):
    # Calculate the mean and standard deviation of the red, green, and blue channels
    r_mean = np.mean(args[1])
    r_std = np.std(args[1])
    g_mean = np.mean(args[2])
    g_std = np.std(args[2])
    b_mean = np.mean(args[3])
    b_std = np.std(args[3])

    # Standardize the red, green, and blue channels
    r_std = (args[1] - r_mean) / r_std 
    g_std = (args[2] - g_mean) / g_std
    b_std = (args[3] - b_mean) / b_std

    return args[0], r_std, g_std, b_std

# %% [markdown]
# #### Execution
# %%
# %%script echo "skipped: open later"
# Load the sample images
selected_imgs_read = [imageio.v3.imread(sample_img) for sample_img in selected_imgs]

# Test the ratio_norm function
ratio_norm_imgs = [create_img(ratio_norm(ext_rgb(img))) for img in selected_imgs_read]

# Test the vector_norm function
vector_norm_imgs = [create_img(vector_norm(ext_rgb(img))) for img in selected_imgs_read]

# Test the standardization function
std_mean_imgs = [create_img(std_mean(ext_rgb(img))) for img in selected_imgs_read]

def display_images(imgs, title):
    plt.figure(figsize=(25, 10))
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(img)
        plt.title(f"{title} Image {i+1}\nfirst pixel rgb={tuple(round(v,2) if not v.is_integer() else int(v) for v in img[0,0,:3])}")
        plt.axis('off')
    plt.show()

# Display the original images
display_images(selected_imgs_read, 'Original')

# Display the ratio_norm images
display_images(ratio_norm_imgs, 'Ratio Norm')

# Display the vector_norm images
display_images(vector_norm_imgs, 'Vector Norm')

# Display the standardization images
display_images(std_mean_imgs, 'Standardization')

#endregion test_image_normalization
# %% [markdown]
# ### Test remaining data points
# %%
#region test_remaining_data
num_full_batches = len(train_doc)//batch_size
remaining_num = len(train_doc) % batch_size
t = np.random.permutation(train_doc)

print(f"Number of full batches: {num_full_batches}")
print(f"Number of remaining images: {remaining_num}")
# %%
# %%script echo "skipped, open later"
import matplotlib.pyplot as plt # import to use plt.imshow() function
from skimage.transform import resize # import to use resize() function

batch_data = np.zeros((remaining_num,x,y,z,3))
batch_labels = np.zeros((remaining_num, 5))
for folder in range(remaining_num):
    fig, axs = plt.subplots(nrows=4, ncols=x, figsize=(30, 15))
    imgs = os.listdir(train_dir+'/'+ t[folder + (num_full_batches*batch_size)].split(';')[0]) # read all the images in the folder
    imgs = sorted(imgs)

    for idx,item in enumerate(img_idx):
        '''fig, axs = plt.subplots(ncols=2, figsize=(12, 5))'''
        image_orig = imageio.v3.imread(train_dir+'/'+t[folder + (num_full_batches*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)

        axs[0,idx].imshow(image_orig/255)
        axs[0,idx].axis('off')
        # axs[0,idx].set_title(f"Image {imgs[item]}\nindex={item}; size={image_orig.shape[:2]}")
        axs[0,idx].set_title(f"original; size={image_orig.shape[:2]}")
        
        #scale image
        image_scaled = image_orig/255

        # crop the image
        image_cropped = crop_image(image_scaled, image_scaled.shape[:2])
        # display cropped image
        axs[1,idx].imshow(image_cropped)
        axs[1,idx].axis('off')
        axs[1,idx].set_title(f"Cropped; size={image_cropped.shape[:2]}")

        # resize the image
        image_resized = resize(image_cropped, (120, 120))
        # display resized image
        axs[2,idx].imshow(image_cropped)
        axs[2,idx].axis('off')
        axs[2,idx].set_title(f"Resized; size={image_resized.shape[:2]}")

        # Calculate the magnitude of the vector
        magnitude = np.sqrt(image_resized[:,:,0]**2 + image_resized[:,:,1]**2 + image_resized[:,:,2]**2)
        # Avoid division by zero
        magnitude[magnitude==0] = 1

        batch_data[folder,idx,:,:,0] = image_resized[:,:,0] / magnitude #normalise and feed in the image
        batch_data[folder,idx,:,:,1] = image_resized[:,:,1] / magnitude #normalise and feed in the image
        batch_data[folder,idx,:,:,2] = image_resized[:,:,2] / magnitude #normalise and feed in the image

        image_norm = create_img([image_resized, image_resized[:,:,0]/magnitude, image_resized[:,:,1]/magnitude, image_resized[:,:,2]/magnitude])
        axs[3,idx].imshow(image_norm)
        axs[3,idx].axis('off')
        axs[3,idx].set_title(f"Normalized: size={image_resized.shape[:2]}")
    plt.show()

    batch_labels[folder, int(t[folder + (num_full_batches*batch_size)].strip().split(';')[2])] = 1
print("batch_labels:\n", batch_labels)  
#endregion test_remaining_data
# %% [markdown]
# ## Final Generator
# %%
#region final_generator
from IPython.display import display # import to use display() function
import pandas as pd # import to use pandas.DataFrame() function
# from scipy.misc import imresize
# ImportError: cannot import name 'imresize' from 'scipy.misc' (/home/khang/.local/lib/python3.10/site-packages/scipy/misc/__init__.py)
from skimage.transform import resize # import to use resize() function

# initialize generator parameters
img_idx, x, y, z = init_gen_params(9,120,120)

def generator(source_path, folder_list, batch_size):
    while True:
        '''print( 'Source path =', source_path, '; batch size =', batch_size)
        print("source_path:", source_path)
        print("folder_list:", folder_list.shape)
        print("batch_size:", batch_size)'''
        #tag todo # examine representative folders to determine similarities between images
        #tag todo # I think 5 would be suffice for img_idx
        # img_idx = img_idx  #create a list of image numbers you want to use for a particular video
        '''print("img_idx =", img_idx)'''
        t = np.random.permutation(folder_list)
        num_batches = len(folder_list)/batch_size # calculate the number of batches
        num_full_batches = len(folder_list)//batch_size
        #tag newly_added
        remaining_num = len(folder_list) % batch_size
        '''print("num_batches =", num_batches)'''
        # create empty containers for the batch
        for batch in range(num_full_batches): # we iterate over the number of batches
            batch_data = np.zeros((batch_size,x,y,z,3)) # x is the number of images you use for each video,(y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((batch_size,5)) # batch_labels is the one hot representation of the output
            for folder in range(batch_size): # iterate over the batch_size

                #tag explain # imgs use listdir to acquire all files (images) within the folder
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
                imgs = sorted(imgs)
                #tag examine # imgs
                # for i, img in enumerate(imgs):
                    # display(f"Index: {i}, Image: {img}")
                #tag remove_later # display(pd.DataFrame(imgs, columns=['imgs']))
                #tag improve # img_idx = [i - 1 for i in range(0, len(imgs), 5)]
                for idx, item in enumerate (img_idx): #  Iterate over the frames/images of a folder to read them in
                    image_orig = imageio.v3.imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    #tag solved # Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
                    # normalize the image
                    # image_orig = image_orig/255

                    #tag todo # crop and resize the images
                    '''
                    crop the images and resize them. Note that the images are of 2 different shape 
                    and the conv3D will throw error if the inputs in a batch have different shapes
                    '''
                    #scale image
                    image_scaled = image_orig/255
                    # crop the image
                    #tag updated # add crop function
                    image_cropped = crop_image(image_scaled, image_scaled.shape[:2])
                    # resize the image
                    image_resized = resize(image_cropped, (120, 120))

                    # Calculate the normalized pixel intensity of the image
                    pixel_vector_norm = np.sqrt(image_resized[:,:,0]**2 + image_resized[:,:,1]**2 + image_resized[:,:,2]**2)
                    # Avoid division by zero
                    pixel_vector_norm[pixel_vector_norm==0] = 1

                    batch_data[folder,idx,:,:,0] = image_resized[:,:,0] / pixel_vector_norm #normalise and feed in the image
                    batch_data[folder,idx,:,:,1] = image_resized[:,:,1] / pixel_vector_norm #normalise and feed in the image
                    batch_data[folder,idx,:,:,2] = image_resized[:,:,2] / pixel_vector_norm #normalise and feed in the image

                    #tag counter_end 
                    
                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels #you yield the batch_data and the batch_labels, remember what does yield do

            # handle remaining data points after full batches
        if remaining_num > 0:
            batch_data = np.zeros((remaining_num,len(img_idx),120,120,3))
            batch_labels = np.zeros((remaining_num, 5))
            for folder in range(remaining_num):
                imgs = os.listdir(source_path+'/'+ t[folder + (num_full_batches*batch_size)].split(';')[0]) # read all the images in the folder
                imgs = sorted(imgs)

                for idx,item in enumerate(img_idx):
                    image_orig = imageio.v3.imread(source_path+'/'+t[folder + (num_full_batches*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    
                    #scale image
                    image_scaled = image_orig/255
                    # crop the image
                    image_cropped = crop_image(image_scaled, image_scaled.shape[:2])
                    # resize the image
                    image_resized = resize(image_cropped, (120, 120))

                    # Calculate the magnitude of the vector
                    pixel_vector_norm = np.sqrt(image_resized[:,:,0]**2 + image_resized[:,:,1]**2 + image_resized[:,:,2]**2)
                    # Avoid division by zero
                    pixel_vector_norm[pixel_vector_norm==0] = 1

                    batch_data[folder,idx,:,:,0] = image_resized[:,:,0] / pixel_vector_norm #normalise and feed in the image
                    batch_data[folder,idx,:,:,1] = image_resized[:,:,1] / pixel_vector_norm #normalise and feed in the image
                    batch_data[folder,idx,:,:,2] = image_resized[:,:,2] / pixel_vector_norm #normalise and feed in the image

                batch_labels[folder, int(t[folder + (num_full_batches*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels
 
# generator(train_path, folders, 10)      
# %% [markdown]
#region tesing
# ### Test the Generator
# %%
# %%script echo "skipped: open later"
# Test the generator function
import itertools
import math
# Initialize the generator
#tag explain # folder_list is basically the data within the csv file(folder_name;gesture_name;int_value)
gen = generator(train_dir, train_doc, batch_size)

# Create an empty dataframe to store batch data and labels
df = pd.DataFrame(columns=['batch_data shape', 'batch_labels shape'])

df = pd.concat([pd.DataFrame({
    'batch_data shape': [list(batch_data.shape)],
    'batch_labels shape': [list(batch_labels.shape)],
}) for batch_data, batch_labels in itertools.islice(gen, math.ceil(len(train_doc) / batch_size))], ignore_index=True)
display(df)

#endregion tesing
#endregion final_generator
# %% [markdown]
# Note here that a video is represented above in the generator as (number of images, height, width, number of channels). Take this into consideration while creating the model architecture.
# %%
curr_dt_time = datetime.datetime.now()
train_path = 'Project_data/train'
val_path = 'Project_data/val'
num_train_sequences = len(train_doc)
print('# training sequences =', num_train_sequences)
num_val_sequences = len(val_doc)
print('# validation sequences =', num_val_sequences)
num_epochs = 10 # choose the number of epochs
print ('# epochs =', num_epochs)
# %% [markdown]
# ## Model Architecture
# Here you make the model using different functionalities that Keras provides. Remember to use `Conv3D` and `MaxPooling3D` and not `Conv2D` and `Maxpooling2D` for a 3D convolution model. You would want to use `TimeDistributed` while building a Conv2D + RNN model. Also remember that the last layer is the softmax. Design the network in such a way that the model is able to give good accuracy on the least number of parameters so that it can fit in the memory of the webcam.

# %%
# import the necessary packages
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation
from keras.layers import Conv3D, MaxPooling3D, Activation, Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

# initialize the parameters
img_idx, x, y, z = init_gen_params(7,120,120)

input_shape = (x, y, z, 3)
num_classes = 5  # Example number of classes

import math

steps_per_epoch = math.ceil(num_train_sequences / batch_size)
validation_steps = math.ceil(num_val_sequences / batch_size)


print("steps_per_epoch: ", steps_per_epoch)
print("validation_steps: ", validation_steps)

train_gen = generator(train_dir, train_doc, batch_size)
val_gen = generator(val_path, val_doc, batch_size)
# %% [markdown]
#region test_model
# ## Test model
# %% [markdown]
# ### Initializing

# %% [markdown]
# ### Model 1
# %%
# Define the model

def create_3dcnn_m1 (input_shape, num_classes):
    keras.backend.clear_session()
    
    global train_gen
    global val_gen
    
    train_gen = generator(train_dir, train_doc, batch_size)
    val_gen = generator(val_path, val_doc, batch_size)

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(128, kernel_size=(3, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    #  Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # Summary of the model
    model.summary()

    return model

model = create_3dcnn_m1(input_shape, num_classes)
# %% [markdown]
# ### Model 2
# %%
def create_3d_cnn_m2 (input_shape, num_classes):
    keras.backend.clear_session()

    global train_gen
    global val_gen

    train_gen = generator(train_dir, train_doc, batch_size)
    val_gen = generator(val_path, val_doc, batch_size)

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv3D(16, kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(32, kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(64, kernel_size=(3, 3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    #  Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # Summary of the model
    model.summary()
    return model
model = create_3d_cnn_m2(input_shape, num_classes)

# %%
# Train the model
history = model.fit(train_gen, steps_per_epoch=steps_per_epoch, 
                    epochs=num_epochs, validation_data=val_gen, 
                    validation_steps=validation_steps, verbose=1)

# Evaluate the model
loss, metrics = model.evaluate(val_gen, steps=validation_steps)
print(f'Test loss: {loss:.3f}, Test metrics: {metrics}')

# %% [markdown]
# Implement batch normailzation
# %%
def create_3d_cnn_m2 (input_shape, num_classes):
    keras.backend.clear_session()

    global train_gen
    global val_gen

    train_gen = generator(train_dir, train_doc, batch_size)
    val_gen = generator(val_path, val_doc, batch_size)

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv3D(16, kernel_size=(3, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(32, kernel_size=(3, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))


    model.add(Conv3D(64, kernel_size=(3, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))


    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    #  Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # Summary of the model
    model.summary()
    return model
# %%
# Train the model
model = create_3d_cnn_m2(input_shape, num_classes)

history = model.fit(train_gen, steps_per_epoch=steps_per_epoch, 
                    epochs=num_epochs, validation_data=val_gen, 
                    validation_steps=validation_steps, verbose=1)

# Evaluate the model
loss, metrics = model.evaluate(val_gen, steps=validation_steps)
print(f'Test loss: {loss:.3f}, Test metrics: {metrics}')
# %%
%%script echo "skipped"
''' # %%
def create_3d_cnn_m2 (input_shape, num_classes):
    keras.backend.clear_session()

    global train_gen
    global val_gen

    train_gen = generator(train_dir, train_doc, batch_size)
    val_gen = generator(val_path, val_doc, batch_size)

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv3D(16, kernel_size=(3, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(32, kernel_size=(1, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))


    model.add(Conv3D(64, kernel_size=(1, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))


    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    #  Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # Summary of the model
    model.summary()
    return model
# %%
# Train the model
model = create_3d_cnn_m2(input_shape, num_classes)

history = model.fit(train_gen, steps_per_epoch=steps_per_epoch, 
                    epochs=num_epochs, validation_data=val_gen, 
                    validation_steps=validation_steps, verbose=1)

# Evaluate the model
loss, metrics = model.evaluate(val_gen, steps=validation_steps)
print(f'Test loss: {loss:.3f}, Test metrics: {metrics}')

# %%
img_idx, x, y, z = init_gen_params(11,120,120)
input_shape =(x,y,z,3)

def create_3d_cnn_m2 (input_shape, num_classes):
    keras.backend.clear_session()

    global train_gen
    global val_gen

    train_gen = generator(train_dir, train_doc, batch_size)
    val_gen = generator(val_path, val_doc, batch_size)

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv3D(16, kernel_size=(3, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(32, kernel_size=(3, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))


    model.add(Conv3D(64, kernel_size=(3, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))


    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    #  Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # Summary of the model
    model.summary()
    return model
# %%
# Train the model
model = create_3d_cnn_m2(input_shape, num_classes)

history = model.fit(train_gen, steps_per_epoch=steps_per_epoch, 
                    epochs=num_epochs, validation_data=val_gen, 
                    validation_steps=validation_steps, verbose=1)

# Evaluate the model
loss, metrics = model.evaluate(val_gen, steps=validation_steps)
print(f'Test loss: {loss:.3f}, Test metrics: {metrics}')'''
# %%
#region codellama_70b
# ## Saving the model
# Now that you have trained the model, it's time to save it. You can use the `save` function to save the model.
model.save('model.h5')
# %%
%%script echo "skipped"
'''# ## Loading the model
# Now that you have trained and saved the model, it's time to load it. You can use the `load_model` function to load the model.
from keras.models import load_model
model = load_model('model.h5')
# ## Summarizing the model
# Now that you have loaded the model, it's time to summarize it. You can use the `summary` function to summarize the model.
model.summary()
# ## Visualizing the model
# Now that you have loaded the model, it's time to visualize it. You can use the `plot_model` function to visualize the model.
from keras.utils import plot_model
plot_model(model, to_file='model.png')
# ## Predicting on new data
# Now that you have loaded the model, it's time to predict on new data. You can use the `predict` function to predict on new data.
model.predict(x_test)
# ## Evaluating the model
# Now that you have loaded the model, it's time to evaluate the model. You can use the `evaluate` function to evaluate the model.
model.evaluate(x_test, y_test)
# ## Saving the model
# Now that you have trained the model, it's time to save it. You can use the `save` function to save the model.
model.save('model.h5')'''
#endregion codellama_70b
#endregion test_model

# %% [markdown]
# ## Selected Model
# %%
def create_3d_cnn_m2 (input_shape, num_classes):
    keras.backend.clear_session()

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv3D(16, kernel_size=(3, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(32, kernel_size=(3, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))


    model.add(Conv3D(64, kernel_size=(3, 3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))


    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model
# %% [markdown]
# Now that you have written the model, the next step is to `compile` the model. When you print the `summary` of the model, you'll see the total number of parameters you have to train.

# %%
model = create_3d_cnn_m2(input_shape, num_classes)
optimiser = Adam(learning_rate=0.001) #write your optimizers
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
        
filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.weights.h5'

#tag error 
'''checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)'''
# TypeError: ModelCheckpoint.__init__() got an unexpected keyword argument 'period'

#tag error
'''
ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=model_init_2024-03-3115_06_39.122125/model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5
The error message is indicating that the filepath provided must end in .keras (Keras model format). However, in your code, you are trying to save the model in .h5 format.
To resolve this, you can either:
Change the file extension in your filepath variable to .keras if you want to save the entire model.
Set save_weights_only=True if you intend to save only the weights
'''

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', save_freq='epoch')



LR = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001) # write the REducelronplateau code here
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
'''# %% 
print("validation_steps:", validation_steps)
print("number_of_epochs:", num_epochs)
# %%
model.fit(train_generator, epochs=num_epochs, verbose=1,
                    validation_data=val_generator, 
                    validation_steps=validation_steps)'''
# %%
#tag error 
num_epochs = 20
'''The fit_generator method has been deprecated in TensorFlow 2.1.0 and later versions. Instead, you should use the fit method, which now supports generators'''
'''model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, initial_epoch=0)'''
'''
The error message indicates that the fit() method of the model object does not accept a workers argument. 
The workers argument is used to specify the number of worker subprocesses to use for data preloading. 
It is only available in the fit_generator() method, which has been deprecated in TensorFlow 2.x and replaced with the fit() method.
If you still want to use multiple worker processes for data preloading, you can use the use_multiprocessing argument instead:
'''
history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1,
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, initial_epoch=0)
# %%
# Extracting accuracy and loss values from the history object
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# Creating a range of epochs
epochs_range = range(num_epochs)

# Setting up the figure size
plt.figure(figsize=(8, 8))

# Creating the first subplot for accuracy
plt.subplot(1, 2, 1)

# Plotting the training and validation accuracy
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')

# Adding a legend to the plot
plt.legend(loc='lower right')

# Adding a title to the plot
plt.title('Training and Validation Accuracy')

# Creating the second subplot for loss
plt.subplot(1, 2, 2)

# Plotting the training and validation loss
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')

# Adding a legend to the plot
plt.legend(loc='upper right')

# Adding a title to the plot
plt.title('Training and Validation Loss')

# Displaying the plot
plt.show()
# %%
model.load_weights('./model_init_2024-04-0314_45_06.392266/model-00019-0.00724-0.99849-0.26904-0.94000.weights.h5')
model.evaluate(val_gen, steps=validation_steps)