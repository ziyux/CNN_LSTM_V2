from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import MaxPooling3D
from keras.layers import Bidirectional
from keras.models import load_model
import numpy as np

class model(object):
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = self.build_model()

    def build_model(self):
        if self.model_type == 'cnnlstm':
            model = Sequential()

            # Add two convolutional 2D layers
            model.add(TimeDistributed(Conv2D(filters=64, kernel_size=(1, 3), activation='relu'),
                                      input_shape=(240, 1, 25, 2)))
            model.add(TimeDistributed(MaxPooling2D(pool_size=(1, 1))))
            model.add(TimeDistributed(Conv2D(filters=32, kernel_size=(1, 3), activation='relu')))
            model.add(TimeDistributed(Flatten()))

            #Add three Bidirectional LSTM layers
            model.add(Bidirectional(LSTM(50, return_sequences=True)))
            model.add(Bidirectional(LSTM(50, return_sequences=True)))
            model.add(Bidirectional(LSTM(50)))

            # Add drop out layer
            model.add(Dropout(0.25))

            # Add fully connected output layer
            model.add(Dense(240, activation='sigmoid'))

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
            print(model.summary())
            return model

        # if self.model_type == 'c3d':
        #     model = Sequential()
        #     model.add(Conv3D(filters=32, kernel_size=(10, 1, 1), activation='relu', input_shape=(None, 240, 1, 2, 25)))
        #     model.add(TimeDistributed(MaxPooling3D(pool_size=(5, 1, 1))))
        #     model.add(Conv3D(filters=32, kernel_size=(10, 1, 1), activation='relu'))
        #     model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 1, 1))))
        #     model.add(TimeDistributed(Flatten()))
        #     model.add(Dropout(0.25))
        #     model.add(Dense(240, activation='sigmoid'))
        #     print(model.summary())
        #     return model

    def fit(self, x, y, batch_size=1, epochs=50):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, predict, label):
        predict[predict >= 0.5] = 1
        predict[predict < 0.5] = 0
        accuracy = 1 - np.sum(np.abs(predict - label), axis=1)/predict.shape[1]
        return accuracy

    def save(self):
        self.model.save('model_' + str(self.model_type) + '.h5')

    def load(self):
        self.model = load_model('model_' + str(self.model_type) + '.h5')
