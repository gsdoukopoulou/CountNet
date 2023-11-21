import tensorflow as tf
from keras.models import load_model, Sequential

model = load_model('/home/gsdoukopoul/CountNet/models/CRNN.h5')
weights = model.get_weights()


