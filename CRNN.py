import os
import keras
from keras import backend as K
from keras import layers
from tensorflow.keras.models import Sequential

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

model = keras.Sequential([
    layers.ZeroPadding2D(padding=(0, 0), input_shape=(1, 500, 201), name='zeropadding2d_input_1'),
    layers.Conv2D(64, (3, 3), activation='relu', name='conv1'),
    layers.Conv2D(32, (3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D(pool_size=(3, 3), name='pool1'),
    layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
    layers.Conv2D(64, (3, 3), activation='relu', name='conv4'),
    layers.MaxPooling2D(pool_size=(3, 3), name='pool2'),
    layers.Dropout(0.5, name='dropout_1'),
    layers.Permute((2, 1, 3), name='permute_1'),
    layers.Reshape((53, 1280), name='reshape_1'),
    layers.LSTM(40, return_sequences=True, name='lstm_1'),
    layers.MaxPooling1D(pool_size=2, name='maxpooling1d_1'),
    layers.Flatten(name='flatten_1'),
    layers.Dense(11, name='dense_1'),
    layers.Activation('softmax', name='activation_1')
])

model.summary()

