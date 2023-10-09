import os
import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, LSTM, Permute, Reshape, Dropout, MaxPooling1D, ZeroPadding2D
from keras.models import Sequential

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

# Create a Sequential model
model = Sequential()

# Layer 1: ZeroPadding2D
model.add(ZeroPadding2D(padding=(0, 0), input_shape=(1, 500, 201), name='zeropadding2d_input_1'))

# Layer 2: Conv2D
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv1'))

# Layer 3: Conv2D
model.add(Conv2D(32, (3, 3), activation='relu', name='conv2'))

# Layer 4: MaxPooling2D
model.add(MaxPooling2D(pool_size=(3, 3), name='pool1'))

# Layer 5: Conv2D
model.add(Conv2D(128, (3, 3), activation='relu', name='conv3'))

# Layer 6: Conv2D
model.add(Conv2D(64, (3, 3), activation='relu', name='conv4'))

# Layer 7: MaxPooling2D
model.add(MaxPooling2D(pool_size=(3, 3), name='pool2'))

# Layer 8: Dropout
model.add(Dropout(0.5, name='dropout_1'))

# Layer 9: Permute
model.add(Permute((2, 1, 3), name='permute_1'))

# Layer 10: Reshape
model.add(Reshape((53, 1280), name='reshape_1'))

# Layer 11: LSTM
model.add(LSTM(40, return_sequences=True, name='lstm_1'))

# Layer 12: MaxPooling1D
model.add(MaxPooling1D(pool_size=2, name='maxpooling1d_1'))

# Layer 13: Flatten
model.add(Flatten(name='flatten_1'))

# Layer 14: Dense
model.add(Dense(11, name='dense_1'))

# Layer 15: Activation
model.add(Activation('softmax', name='activation_1'))

# Print the model summary
model.summary()

