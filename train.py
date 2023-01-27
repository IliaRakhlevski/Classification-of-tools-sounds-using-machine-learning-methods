# This module is used for model creation/training

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D 
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import datetime
import samples_processing
import settings



# Model configuration
BATCH_SIZE = 1
NO_EPOCHS = 20
PATIENCE = NO_EPOCHS
LEARNING_RATE = 0.00001
TEST_SIZE = settings.TEST_SIZE
VALIDATION_SPLIT = settings.TEST_SIZE



# set using CPU/GPU    
def set_cpu_gpu(CPU = True):
    
    import tensorflow as tf
    from keras import backend as K
    import psutil 
    
    num_cores = psutil.cpu_count(logical = False)
    
    #if GPU:
    num_GPU = 1
    num_CPU = 1
    
    if CPU:
        num_GPU = 0
    
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
            device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session(config=config)
    K.set_session(session)



# Create model
# classes - classification classes
def create_model(classes):
    model = Sequential()
    
    model.add(Conv2D(128, kernel_size=(3, 15), activation='relu', padding='same',
                     input_shape=(2, settings.SMPLS_NUM, 1)))
    model.add(Conv2D(128, kernel_size=(3, 9), activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 5), activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(len(classes), activation='softmax'))
       
    model.summary()

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=LEARNING_RATE),
                  metrics=['accuracy'])
    return model



# data training
# X - preprocessed audio data
# Y - predicted classes
# classes - classification classes
def data_training(X, Y, classes):
    
    set_cpu_gpu()
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, shuffle=True, random_state=0)
    
    # create new model
    #model = create_model(classes)
    
    # load existing model
    model = keras.models.load_model('model.h5') 
    
    earlystop = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    callbacks = [earlystop]

    # Fit data to model
    model.fit(x = X_train, y = y_train,
                batch_size=BATCH_SIZE,
                shuffle=True, 
                epochs=NO_EPOCHS,
                validation_split=VALIDATION_SPLIT,
                callbacks=callbacks)

    # Generate generalization metrics
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    y_pred = model.predict(X_test)
    
    y_pred = np.argmax(y_pred, axis = 1)
    y_test = np.argmax(y_test, axis = 1)
     
    print(classification_report(y_test, y_pred))
    
     
    ts = datetime.datetime.now().timestamp()
    filename = 'model_{0}.h5'.format(ts)    
    model.save(filename)
    




if __name__ == '__main__':

    # audio data preparation
    X, Y = samples_processing.create_data('Data', settings.CLASSES)
   
    # and model training
    data_training(X, Y, settings.CLASSES)
      
    
