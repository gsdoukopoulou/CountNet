import tensorflow as tf
import keras
from keras import backend as K
from keras.models import load_model, Sequential
import numpy as np

def class_mae(y_true, y_pred): # calculate mean absolute error
    return K.mean(K.abs(K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)),axis=-1)

model = load_model('/home/gsdoukopoul/CountNet/models/CRNN.h5', custom_objects={'class_mae': class_mae})
weights = model.get_weights()

model.save('/home/gsdoukopoul/CountNet/models/weights', save_format='tf')