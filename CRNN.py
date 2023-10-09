import os
import keras
from keras import backend as K


def class_mae(y_true, y_pred): # calculate mean absolute error
    return K.mean(
        K.abs(
            K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)
        ),
        axis=-1
    )


# load model
model = keras.models.load_model(
    os.path.join('models' , 'CRNN.h5') ,
    custom_objects={
        'class_mae': class_mae ,
        'exp': K.exp
    }
)

model.summary()