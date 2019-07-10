import numpy as np 
# from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import cv2 
import random 
import os 
from ImageGenerator_v2 import ImageDataGenerator
from PIL import Image 

ROOT_DIR = os.path.abspath('../')

class Dataset():

    def __init__(self, frames_per_step=16, image_shape=(112,112), batch_size=10, sub_folder='UCF_2'):

        self.image_shape = image_shape
        self.frames_per_step = frames_per_step
        self.batch_size = batch_size
        self.sub_folder = sub_folder 
    
    def get_generators(self):

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            data_format='channels_first'
            )

        test_dataget = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            os.path.join(ROOT_DIR + '/data/' + self.sub_folder, 'train'),
            target_size=self.image_shape,
            batch_size = self.batch_size,
            frames_per_step = self.frames_per_step,
            shuffle=False,
            class_mode = 'categorical',
            classes=['get_on','get_off']
        )

        validation_generator = train_datagen.flow_from_directory(
            os.path.join(ROOT_DIR + '/data/' + self.sub_folder, 'test'),
            target_size=self.image_shape,
            batch_size = self.batch_size,
            frames_per_step = self.frames_per_step,
            shuffle=False,
            class_mode = 'categorical',
            classes=['get_on','get_off']
        )
    
        return train_generator, validation_generator 







