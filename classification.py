# Audio classification module

import settings
import samples_processing
import keras
import numpy as np
import os


# print results of prediction
# pred_res - results of prediction
def print_results(pred_res):
    
    pred_sum = sum(pred_res)                                      

    print("\n\n")
    for i in range(len(pred_res)):
        pred = pred_res[i]
        text = "{0} - {1} - {2} %".format(settings.CLASSES[i], pred, pred / (pred_sum / 100))
        print(text)
        
    print("\n")
    
    max_ind = np.argmax(pred_res)
    text = "Predicted: {0} - {1} %".format(settings.CLASSES[max_ind], pred_res[max_ind] / (pred_sum / 100))
    print(text)
    print("\n")
    


# classification of samples
# samples_list - audio samples list
# model - created/loaded model
def samples_classification(samples_list, model):
    smpls = np.asarray([samples_list])
    predictions = model.predict(smpls)
    return np.argmax(predictions[0])



# classification of audio stream 
# audio_file_name - audio file path
# model - created/loaded model
# sampling_rate - sampling rate (for conversion)
# plot - if must be plotted (for debug)
def audio_classification(audio_file_name, model, sampling_rate = settings.SAMPLING_RATE, plot = False):
    smpls, current_sampling_rate = samples_processing.read_samples_from_file(audio_file_name, plot)
    new_data = samples_processing.sample_rate_conversion(smpls, current_sampling_rate, sampling_rate)
    
    # find maximal absolute value in the samples
    max_new_data_val = abs(max(new_data, key=abs))
    
    # find value below which the samples chunk is ignored
    min_val_be_concerned = max_new_data_val * settings.VAL_IGNORED
    
    min_val_text = "Maximal value is {0}, minimal value to be concerned - {1}\n".format(max_new_data_val,
                                                                                  min_val_be_concerned)
    print(min_val_text)
    
    if plot == True:
        print("\nData after rate conversion:")
        samples_processing.plot_samples(new_data)
    
    win_pos = 0     # window position
    pred_res = [0]*len(settings.CLASSES)
    
    # windowing
    while win_pos < len(new_data):
        
        samples = None
        
        # get chunk starting at 'win_pos' and its lenght is settings.SMPLS_NUM
        # or less if the distance to the samples end is less then settings.SMPLS_NUM
        if (win_pos + settings.SMPLS_NUM) < len(new_data):
            samples = smpls[win_pos:(win_pos + settings.SMPLS_NUM)]
        else:
            samples = smpls[win_pos:]
            
        # get maximal absolute value of the current chunk
        max_val = abs(max(samples, key=abs))
        
        # if the maximal absolute value of the current chunk is larger
        # then value below which the samples chunk is ignored
        if max_val > min_val_be_concerned:
            
            # get processed data from samples
            pd = samples_processing.get_processed_data(samples, sampling_rate, head_pad_percent = 0, plot = False)
    
            # reshape the processed data
            pd = np.asarray(pd)
            pd = pd.reshape(2, settings.SMPLS_NUM, 1)
               
            # perform classification of the data
            class_ind = samples_classification(pd, model)
            pred_res[class_ind] += 1
            
            win_class_res = "Start - {0}, End - {1}, Class - {2}".format(win_pos, 
                                     win_pos + settings.SMPLS_NUM, 
                                     settings.CLASSES[class_ind])
            print(win_class_res)
            
        else:
            win_class_res = "Start - {0}, End - {1}, Max value - {2}, Ignored".format(win_pos, 
                                     win_pos + settings.SMPLS_NUM, 
                                     max_val)
            print(win_class_res)
            
        if plot == True:
            samples_processing.plot_samples(samples)
            print("\n")
    
        win_pos += settings.WIN_STEP
    
    return pred_res



# run classification on files in the test directory
# input_dir - input directory
def run_test_classification(input_dir):
    
    model = keras.models.load_model('model.h5') 
    
    files_list = os.listdir(input_dir)
     
    for f in files_list:
        file_path = os.path.join(input_dir, f)
        print(file_path.replace("\\","/")) 
        pred_res = audio_classification(file_path, model)   
        print_results(pred_res)
            
    

if __name__ == '__main__':
    # used for test
    
 
    #audio_file_name = 'Sounds/Test/hammer.wav'
    
    #audio_file_name = 'Sounds/Test/drill_1.wav'
    
    #audio_file_name = 'Sounds/Test/drill_2.wav'
    
    #audio_file_name = 'Sounds/Test/jackhammer_1.wav'
    
    #audio_file_name = 'Sounds/Test/jackhammer_2.wav'
    
#    model = keras.models.load_model('model.h5')   
#
#    pred_res = audio_classification(audio_file_name, model)                                  
#
#    print_results(pred_res)
    
    # classification of all the files are found in the test directory
    run_test_classification('Sounds/Test')
    
    
    
    