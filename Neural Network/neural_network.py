"""
Neural network train file.
"""
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from config import SAVE_DIR_PATH
from config import MODEL_DIR_PATH


class TrainModel:

    @staticmethod
    def train_neural_network(X, y) -> None:
        """
        This function trains the neural network.
        """
        # splitting the dataset into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # Reshape array from 2D to 4D to use Conv2D
        x_traincnn = X_train[..., np.newaxis, np.newaxis]
        x_testcnn = X_test[..., np.newaxis, np.newaxis]

        print(x_traincnn.shape, x_testcnn.shape)
        # layer 1
        model = Sequential()
        model.add(Conv2D(64, (3,3), padding='same', activation='relu',
                         input_shape=(40, 1, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(1)))

        # layer 2
        model.add(Conv2D(32, (3,3), padding='same', activation='relu',))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(1)))

        # layer 3
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(1)))

        # dense layer connected to softmax output
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='softmax'))
        print(model.summary)

        # Compile the model using the model.compile
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # Training the model using model.fit()
        cnn_history = model.fit(x_traincnn, y_train,
                                batch_size=16, epochs=55,
                                validation_data=(x_testcnn, y_test))

        # Loss plotting
        plt.plot(cnn_history.history['loss'])
        plt.plot(cnn_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss.png')
        plt.show()
        plt.close()

        # Accuracy plotting
        plt.plot(cnn_history.history['accuracy'])
        plt.plot(cnn_history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('acc.png')
        plt.show()

        predictions = model.predict_classes(x_testcnn)
        new_y_test = y_test.astype(int)
        matrix = confusion_matrix(new_y_test, predictions)

        print(classification_report(new_y_test, predictions))
        print(matrix)

        model_name = 'Emo.h5'

        # Save model and weights
        if not os.path.isdir(MODEL_DIR_PATH):
            os.makedirs(MODEL_DIR_PATH)
        model_path = os.path.join(MODEL_DIR_PATH, model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)


if __name__ == '__main__':
    print('Training started')
    X = joblib.load(SAVE_DIR_PATH + '\\X.joblib')
    y = joblib.load(SAVE_DIR_PATH + '\\y.joblib')
    TrainModel.train_neural_network(X=X, y=y)
