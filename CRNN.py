import os
import keras
from keras import backend as K

# load model
model = keras.models.load_model(
    os.path.join('models' , 'CRNN.h5') ,
    custom_objects={
        'class_mae': class_mae ,
        'exp': K.exp
    }
)

model.summary()