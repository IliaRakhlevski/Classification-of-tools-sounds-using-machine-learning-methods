# Main module

import samples_processing
import settings
import train


if __name__ == '__main__':

    # audio data preparation 
    X, Y = samples_processing.create_data('Data', settings.CLASSES)
    
    # and model training
    train.data_training(X, Y, settings.CLASSES)
    