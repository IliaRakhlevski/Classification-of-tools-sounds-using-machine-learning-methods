# This module create peaks envelopes for audio data

import numpy as np
from numpy import sign, zeros
from scipy.interpolate import interp1d
from matplotlib.pyplot import plot,show,grid
import settings


# create peaks envelopes
def peaks_envelope(smpls):
    
    if type(smpls) is not np.array:
        s =  np.array(smpls)
    else:
        s = smpls

    q_u = zeros(s.shape)
    q_l = zeros(s.shape)
    
    #Prepend the first value of (s) to the interpolating values. 
    #This forces the model to use the same starting point for both the upper and lower envelope models.
    u_x = [0,]
    u_y = [s[0],]
    
    l_x = [0,]
    l_y = [s[0],]
    
    #Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.
    for k in range(1,len(s)-1):
        if (sign(s[k]-s[k-1])==1) and (sign(s[k]-s[k+1])==1):
            u_x.append(k)
            u_y.append(s[k])
    
        if (sign(s[k]-s[k-1])==-1) and ((sign(s[k]-s[k+1]))==-1):
            l_x.append(k)
            l_y.append(s[k])
    
    #Append the last value of (s) to the interpolating values. 
    #This forces the model to use the same ending point for both the upper and lower envelope models.
    u_x.append(len(s)-1)
    u_y.append(s[-1])
    
    l_x.append(len(s)-1)
    l_y.append(s[-1])
    
    #Fit suitable models to the data. 
    u_p = interp1d(u_x,u_y, kind = settings.INTER_1D_KIND, bounds_error = False, fill_value=0.0)
    l_p = interp1d(l_x,l_y, kind = settings.INTER_1D_KIND, bounds_error = False, fill_value=0.0)
    
    #Evaluate each model over the domain of (s)
    for k in range(0,len(s)):
        q_u[k] = u_p(k)
        q_l[k] = l_p(k)
        
    return q_u, q_l


# create peaks envelopes with specific order
def peaks_envelope_with_order(smpl, order):
    q_u, q_l = peaks_envelope(smpl)
    for i in range(1, order):
        q_uu, q_ul = peaks_envelope(q_u)
        q_lu, q_ll = peaks_envelope(q_l)
        q_u = q_uu
        q_l = q_ll
    return q_u, q_l
    

# plot samples with lower and upper peaks envelopes
def plot_data_with_envelopes(s, q_u, q_l):
    #Plot everything
    plot(s);
    plot(q_u,'r');
    plot(q_l,'g');
    grid(True);
    show()
    
    
# create peaks envelopes with specific order and plot them (optional)
def create_peaks_envelopes(data, order = 1, is_plot = False):
    peaks_envelope_testing_data = []

    for i in range(0, len(data)):
       q_u, q_l = peaks_envelope_with_order(data[i], order)  # create peaks envelopes
       if is_plot == True:
           print("\nPeaks envelopes:")
           plot_data_with_envelopes(data[i], q_u, q_l)      # plot data with envelopes, if required
       l = [q_u.tolist(), q_l.tolist()]
       peaks_envelope_testing_data.append(l)                # add this list to the 'peaks_envelope_testing_data'
    return peaks_envelope_testing_data




