import tensorflow as tf
from keras.models import load_model, Sequential


def class_mae(y_true, y_pred): # calculate mean absolute error
    return K.mean(K.abs(K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)),axis=-1)

model = load_model('/home/gsdoukopoul/CountNet/models/CRNN.h5')
weights = model.get_weights()


