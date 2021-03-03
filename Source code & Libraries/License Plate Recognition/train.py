'''Importing Libraries'''
#import cv2
#import itertools
#import os, random
#import numpy as np
#from glob import glob
#from tqdm import tqdm_notebook
from matplotlib import pyplot as plt
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import SparseCategoricalCrossentropy
#from keras.utils import plot_model, Sequence
from utils import img_h, img_w, batch_size, val_batch_size, downsample_factor, max_text_len, epochs

from network import get_Model
from DataGeneration import DataGenerator

model = get_Model(training = True)
model.compile(loss = {'ctc': lambda y_true, y_pred: y_pred}, optimizer = Adadelta())

early_stop = EarlyStopping(
    monitor = 'val_loss', min_delta = 0.001,
    patience = 4, mode = 'min',
    verbose = 1
)

checkpoint = ModelCheckpoint(
    filepath = 'CuDNNLSTM+BN5--{epoch:02d}--{loss:.3f}--{val_loss:.3f}.hdf5',
    monitor = 'val_loss', verbose = 1,
    mode = 'min', period = 1
)
ada = Adadelta()

train_datagen = DataGenerator('Korean-license-plate-Generator/Train',  img_w, img_h, batch_size, downsample_factor)
valid_datagen = DataGenerator('Korean-license-plate-Generator/Val',  img_w, img_h, val_batch_size, downsample_factor)
# _steps_per_epoch = train_datagen.__len__()
# _validation_steps = valid_datagen.__len__()
val_dataset = valid_datagen.buildDataset()
train_dataset = train_datagen.buildDataset()
# steps_per_epoch = train_datagen.getSteps()
# validation_steps = valid_datagen.getSteps()

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = ada, metrics=['accuracy'])

# model.fit_generator(generator=train_datagen.next_batch(),
#                     steps_per_epoch = steps_per_epoch,
#                     epochs = epochs,
#                     callbacks=[checkpoint],
#                     validation_data=valid_datagen.next_batch(),
#                     validation_steps = validation_steps)

# import numpy as np
# print(np.typename(train_datagen[0]))
# print(np.typename(train_datagen[1]))

#print(train_datagen)
# model.compile(optimizer='adam',
#               loss = SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
history = model.fit(
    train_dataset,
    #generator = train_datagen,
    # x = train_dataset,
    # y = yTrain,
    #steps_per_epoch = _steps_per_epoch,#int(len(train_datagen.image_list) // batch_size),
    epochs = epochs, 
    callbacks = [checkpoint],
    batch_size = batch_size,
    validation_batch_size=val_batch_size,
    validation_data = val_dataset#,
    #validation_steps = _validation_steps#int(len(valid_datagen.image_list) // batch_size)
)

plt.plot(history.history['loss'], color = 'b')
plt.plot(history.history['val_loss'], color = 'r')
plt.show()