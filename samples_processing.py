# Audio samples processing

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
import scipy.signal as sps
import settings
import peaks_envelope
import sklearn.preprocessing
import os



# read samples from file and plot it
# path - path to the file
# plot - is must be plotted
def read_samples_from_file(path, plot = False):
    sampling_rate, data = read(path)

    if len(data.shape) > 1: # stereo
        data = data[:,0]

    if plot == True:
        print("\nLoaded: " + path)
        print(" -- Sampling Frequency is " + str(sampling_rate))
        print(" -- Number of samples: " + str(len(data)) + "\n")       
        plot_samples(data)
    
    return data, sampling_rate



# plot audio samples
# data - data to be plotted
def plot_samples(data):
    plt.figure()
    plt.plot(data)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('Waveform of Test Audio')
    plt.show()
    
   
    
# samples rate conversion
# data - samples data
# sampling_rate - current sampling rate
# new_rate - new sampling rate
def sample_rate_conversion(data, sampling_rate, new_rate = settings.SAMPLING_RATE):
    
    if sampling_rate == new_rate:
        return data
    
    # Resample data
    number_of_samples = round(len(data) * float(new_rate) / sampling_rate)
    new_data = sps.resample(data, number_of_samples)
    return new_data



# write samples to file
# path - path to the file
# rate - sampling rate
# data - samples to be written
def write_samples_to_file(path, rate, data):
    write(path, rate, data.astype(np.int16))



# normilize audio data
# audio - audio data
def normalize(audio):
    audio_scaled = sklearn.preprocessing.minmax_scale(audio, 
                                                      feature_range=(settings.NORM_RANGE_MIN, 
                                                                     settings.NORM_RANGE_MAX))
    return audio_scaled



# divide audio file into parts according to the intervals
# path - file path
# intervals - list of intervals, each interval has start/end samples
def divide_file(path, intervals):
    files_names = []
    data, sampling_rate = read_samples_from_file(path, True) 
    for i in range(0, len(intervals)):
        interval = intervals[i]
        ch = "_{0}.wav".format(i)
        new_path = path.replace(".wav", ch)
        smpls = data[interval[0]:interval[1]]
        write_samples_to_file(new_path, sampling_rate, smpls)
        files_names.append(new_path)
    return files_names   
    


# write samples to files accordint to the intervals    
# basis_path - basic name of the file
# smpls - samples to be written
# sampling_rate - sampling rate 
# intervals - list of intervals
def write_smpls_intervals_to_files(basis_path, smpls, sampling_rate, intervals):
    files_names = []
    for i in range(0, len(intervals)):
        interval = intervals[i]
        ch = "__{0}_{1}.wav".format(interval[0], interval[1])
        new_path = basis_path.replace(".wav", ch)
        new_smpls = smpls[interval[0]:interval[1]]
        write_samples_to_file(new_path, sampling_rate, new_smpls)
        files_names.append(new_path)
    return files_names   
 
 
    
# add pads to smples 
# smpls - samples 
# head_percent - percent of head pad
# new_num_smpls - new number of samples
def add_pad_to_smpls(smpls, head_percent, new_num_smpls):
    if len(smpls) < new_num_smpls:
        tail_percent = 100 - head_percent
        extra_smpls = new_num_smpls - len(smpls)
        head_smpls = int(extra_smpls / 100.0 * head_percent)
        tail_smpls = int(extra_smpls / 100.0 * tail_percent)
        if (tail_smpls + head_smpls + len(smpls)) < new_num_smpls:
            extra = new_num_smpls - (tail_smpls + head_smpls + len(smpls))
            tail_smpls += extra
        smpls = np.pad(smpls, (head_smpls, tail_smpls), 'constant')
    return smpls
        


# fix samples number 
# smpls - samples   
# head_pad_percent - head pad percent 
def fix_smpls_num(smpls, head_pad_percent = 0):
    
    # if number of samples larger than the defined one cut them
    if len(smpls) > settings.SMPLS_NUM:
        smpls = smpls[0:settings.SMPLS_NUM]
    # if number of samples smaller than the defined one add head/tail pads
    elif len(smpls) < settings.SMPLS_NUM:
        smpls = add_pad_to_smpls(smpls, head_pad_percent, settings.SMPLS_NUM)
    return smpls
        


# process row data
# smpls - samples 
# sampling_rate - sampling rate 
# head_pad_percent - head pad percent 
# plot - is must be plotted    
def get_processed_data(smpls, sampling_rate, head_pad_percent = 0, plot = False):
    new_data = sample_rate_conversion(smpls, sampling_rate)
    
    if plot == True:
        print("\nData after rate conversion:")
        plot_samples(new_data)

    scaled_data = normalize(new_data)
    
    if plot == True:
        print("\nNormalized data:")
        plot_samples(scaled_data)
        
    fix_smpls_len = fix_smpls_num(scaled_data, head_pad_percent)
     
    if plot == True:
        print("\nFixed samples number:")
        plot_samples(fix_smpls_len)

    pe = peaks_envelope.create_peaks_envelopes([fix_smpls_len], settings.PEAKS_ENV_ORDER, plot)
    return pe



# load audio data from file
# path - path to the file
# head_pad_percent - head pad percent 
# plot - is must be plotted 
def load_data_from_file(path, head_pad_percent = 0, plot = False):
    smpls, sampling_rate = read_samples_from_file(path, plot)
    pd = get_processed_data(smpls, sampling_rate, head_pad_percent, plot)
    return pd



# data preparation for training/testing
# input_dir - directory containing audio data
# classes - classification classes
def create_data(input_dir, classes):
    X = []  # audio data - list of video sequences
    Y = []  # list of classes for classification
     
    classes_list = os.listdir(input_dir)
     
    for c in classes_list:
        print(c)
        files_list = os.listdir(os.path.join(input_dir, c))
        for f in files_list:
            processed_data = load_data_from_file(os.path.join(os.path.join(input_dir, c), f))
            
            processed_data = np.asarray(processed_data)
            processed_data = processed_data.reshape(2, settings.SMPLS_NUM, 1)
            
            # add preprocessed data to the audio data
            X.append(processed_data)             
            y = [0]*len(classes)
            y[classes.index(c)] = 1
            Y.append(y)

    return X, Y


if __name__ == '__main__':
#   data, sampling_rate = read_samples_from_file('Sounds/Hammer/HAMMERN2.wav', True)  
#   new_data = sample_rate_conversion(data, sampling_rate)
#   plot_samples(new_data)
#   write_samples_to_file('out.wav', settings.SAMPLING_RATE, new_data)
#
#   audio_scaled = normalize(new_data)
#   plot_samples(audio_scaled)
#
#   pe = peaks_envelope.create_peaks_envelopes([audio_scaled], 2, True)
   #pe0 = np.array(pe[0])
   
   
#   locs = [[4000, 12000],[26000, 33000]]
#   divide_file('Sounds/Hammer/HAMMERN2.wav', locs)
#   data, sampling_rate = read_samples_from_file('Sounds/Hammer/HAMMERN2_0.wav', True) 
#   data, sampling_rate = read_samples_from_file('Sounds/Hammer/HAMMERN2_1.wav', True) 
    
  pe = load_data_from_file('Sounds/Hammer/smrpg_mario_hammer.wav', 0, True) 
  # pe = load_data_from_file('Sounds/Jackhammer/zvuk-otboynogo-molotka-26925.wav', 0, True)
  # pe = load_data_from_file('Sounds/Drill/pnevmaticheskaya-drel-35103.wav', 0, True)
   
   
   
   