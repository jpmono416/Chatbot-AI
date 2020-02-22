# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


"""
For future improvement of this class, implement a "test"
method that runds the pre-trained model on a particular dataset
and load this model from disk using keras.model.save("*.h5")
"""

class Model:
    @staticmethod
    def build(width, height, depth, classes, reg, init="he_normal"):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid",
                         kernel_initializer=init, kernel_regularizer=reg,
                         input_shape=inputShape))

        model.add(Conv2D(32, (3, 3), padding="same",
                         kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        # Adding another layer for improving the training function
        model.add(Conv2D(64, (3, 3), padding="same",
                    kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        # Final training layer. Learn 128 features this time
        model.add(Conv2D(128, (3, 3), padding="same",
                         kernel_initializer=init, kernel_regularizer=reg))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        # fully-connected layer
        model.add(Flatten())
        model.add(Dense(512, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        model.save("blank-model")
        return model
