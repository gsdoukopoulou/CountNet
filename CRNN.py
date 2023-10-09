import os
import keras
from keras import backend as K

# load model
model = keras.models.load_model(
    os.path.join('models' , 'CRNN.h5') ,
)

model.summary()