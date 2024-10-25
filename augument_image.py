import os
import numpy as np
import keras
print(keras.__version__)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Input folder containing faces
input_folder = 'in_image'  # Folder with your cropped face images
augmented_folder = 'out_aug_image'  # Output folder for augmented images

# Create the output folder if it doesn't exist
if not os.path.exists(augmented_folder):
    os.makedirs(augmented_folder)

# Initialize the ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        
        # Load the image
        img = load_img(image_path)  # Load image
        x = img_to_array(img)  # Convert to numpy array
        x = np.expand_dims(x, axis=0)  # Reshape for the generator

        # Generate augmented images and save them
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_folder,
                                   save_prefix='aug', save_format='jpg'):
            i += 1
            if i > 10:  # Number of augmented images to generate per original image
                break

print('Data augmentation complete! Augmented images saved in', augmented_folder)
