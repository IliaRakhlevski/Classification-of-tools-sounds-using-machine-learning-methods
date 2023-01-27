# This module is used for WAV files preprocessing

import samples_processing
import settings




# plot samples from list of intervals
# smpls - samples list
# intervals - list of intervals, each interval has start/end samples
def plot_smpls(smpls, intervals):
    for i in range(0, len(intervals)):
        interval = intervals[i]
        samples = smpls[interval[0]:interval[1]]
        orig_int = "\nOriginal interval: {0} - {1}".format(interval[0], interval[1]) 
        print(orig_int)
        samples_processing.plot_samples(samples) 
        
        

# plot samples from list of WAV files      
def plot_files(paths):
    for path in paths:
        samples_processing.read_samples_from_file(path, True) 
    
    

if __name__ == '__main__': 
  
    # example how to devide an audio file into several files according to
    # the variable "intervals"
    video_name = 'Sounds/Process/25995.wav'
    
    # load original audio samples
    data, sampling_rate = samples_processing.read_samples_from_file(video_name, True)  
    
    # resample the loaded data
    rate_conv_data = samples_processing.sample_rate_conversion(data, sampling_rate)
    samples_processing.plot_samples(rate_conv_data) 
    
    # intervals table
    intervals = [
                    [25000, 30000],
                    [40000, 45000],
                    [50000, 55000],
                 ]
    plot_smpls(rate_conv_data, intervals) 
       
    # write the intervals to files
    files_names = samples_processing.write_smpls_intervals_to_files(video_name, 
                                                                  rate_conv_data, 
                                                                  settings.SAMPLING_RATE, 
                                                                  intervals)  
     
    # load newly created files and plot them
    plot_files(files_names)

