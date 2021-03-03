import numpy as np
import tensorflow as tf
import cv2
import os, random 

from functions import text_to_labels
from glob import glob
from keras.utils import plot_model, Sequence



class DataGenerator(Sequence):

    def __init__(self, image_path, image_width, image_height,
                 batch_size=0, downsample_factor=0, max_text_len=9, shuffle=True):
        '''Initialize data generator:
        @image_path [in]: directory contains images,
        @image_width [in]: normalized image width,
        @image_height [in]: normalized image height
        '''
        self.image_path = image_path
        self.image_list = glob(self.image_path + '/*')
        self.n = len(self.image_list)
        self.indexes = list(range(self.n))
        self.cur_index = 0        
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.downsample_factor = downsample_factor
        self.shuffle = shuffle
        self.max_text_len = max_text_len
        self.on_epoch_end()
        self.images = np.ones([self.n, self.image_width, self.image_height, 1])#np.zeros((self.n, self.image_height, self.image_width))
        self.texts = np.ones([self.n, self.max_text_len])#[]#np.zeros(self.n)
        self.input_length = np.ones((self.n, 1)) * (self.image_width // self.downsample_factor - 2)  # (bs, 1)
        self.label_length = np.zeros((self.n, 1)) 
    
    def buildDataset(self):
        '''
        Build dataset
        '''
        # images = []
        # texts = []
        for i, impath in enumerate(self.image_list):
            image = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (self.image_width, self.image_height))
            image = image.astype(np.float32)
            image = (image / 255.0)  * 2.0 - 1.0
            #image = image.T 
            image = np.expand_dims(image.T, -1)
            self.images[i] = image
            #self.images[i, :, :] = image 
            imname = os.path.basename(impath)
            #print(imname)
            #remove character(s) '_' in the end of text
            # text = imname[0 : 9]#.split('/')[-1]
            # #text = imname[0 : -4].split('\\')[-1]
            #print(imname[0 : 9])
            self.texts[i] = text_to_labels(imname[0 : 9])
            #self.texts[i] = text_to_labels(imname[0 : 9])
            #self.texts.append(imname[0 : 9])
            #self.texts.append(text_to_labels(imname[0 : 9]))
        
        inputs = {
            'the_input': self.images,  # (n, 128, 64, 1)
            'the_labels': self.texts,  # (n, 8)
            'input_length': self.input_length,  # (n, 1) -> 모든 원소 value = 30
            'label_length': self.label_length  # (n, 1) -> 모든 원소 value = 8
        }

        outputs = {'ctc': np.zeros([self.n])} # (n, 1)

        return (inputs, outputs)
        #return tf.data.Dataset.from_tensor_slices((inputs, outputs))
        #self.images = np.expand_dims(np.array(images), axis = 3)
        #self.texts = np.asarray(texts)
        #return tf.convert_to_tensor(images), tf.convert_to_tensor(texts, dtype=str)
        # images = tf.convert_to_tensor(images)
        # texts = tf.convert_to_tensor(texts)
        # inputs = tf.data.Dataset(images)
        # outputs = tf.data.Dataset(texts)
        # ds = tf.data.Dataset.zip((inputs, outputs))
        # #ds = ds.batch(self.batch_size)
        # return ds.batch(self.batch_size)

    def next_sample(self):      ## index max -> 0 으로 만들기
        '''
        Get next data sample:
        '''
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.images[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]
    
    def next_batch(self):       ## batch size만큼 가져오기
        while True:
            X_data = np.ones([self.batch_size, self.image_width, self.image_height, 1])     # (bs, 128, 64, 1)
            Y_data = np.ones([self.batch_size, self.max_text_len])             # (bs, 9)
            input_length = np.ones((self.batch_size, 1)) * (self.image_width // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                #print("text:", text)
                Y_data[i] = text#text_to_labels(text)
                #label_length[i] = len(text)

            # dict 형태로 복사
            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) -> 모든 원소 value = 30
                'label_length': label_length  # (bs, 1) -> 모든 원소 value = 8
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1) -> 모든 원소 0
            yield (inputs, outputs)
    
    def getSteps(self):
        '''
        Get number of steps
        '''
        return self.n // self.batch_size
    # def __len__(self):
    #     '''Denotes the number of batches per epoch'''
    #     return int(np.floor(len(self.image_list) / self.batch_size))
    
    # def __getitem__(self, index):
    #     '''Generate Single Batch of Data'''
        
    #     # Generate indexes of the batch
    #     indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

    #     # Find list of IDs
    #     list_IDs_temp = [self.image_list[k] for k in indexes]

    #     # Generate data
    #     inputs, outputs = self.__data_generation(list_IDs_temp)

    #     return inputs, outputs
    
    # def on_epoch_end(self):
    #     '''Updates indexes after each epoch'''
    #     self.indexes = np.arange(len(self.image_list))
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)
    
    # def __data_generation(self, list_IDs_temp):
    #     '''Generates data containing batch_size samples'''
        
    #     # Reading Data
    #     images = []
    #     texts = []
    #     label_length = np.zeros((self.batch_size, 1))
        
    #     for i, img_file in enumerate(list_IDs_temp):
    #         image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    #         image = cv2.resize(image, (self.image_width, self.image_height))
    #         image = image.astype(np.float32)
    #         image = (image / 255.0)
    #         images.append(image.T)
    #         text = img_file[0 : -4].split('/')[-1]
    #         text = img_file[0 : -4].split('\\')[-1]
    #         # texts.append()
    #         texts.append(text_to_labels(text))
    #         label_length[i] = len(text)
    #         # images[i, :, :] = image
        
    #     input_length = np.ones((self.batch_size, 1)) * (self.image_width // self.downsample_factor - 2)
    #     images = np.expand_dims(np.array(images), axis = 3)
        
    #     inputs = {
    #         'the_input': images,  # (bs, 128, 64, 1)
    #         'the_labels': np.array(texts),  # (bs, 8)
    #         'input_length': input_length,  # (bs, 1) 
    #         'label_length': np.array(label_length)  # (bs, 1)
    #     }

    #     #outputs = {'ctc': np.zeros([self.batch_size])} # (bs, 1) 

    #     return images, np.array(texts)
    #     #return inputs, outputs
    
