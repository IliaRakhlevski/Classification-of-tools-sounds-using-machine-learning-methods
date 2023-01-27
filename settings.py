# Global parameters


# Default sampling rate
SAMPLING_RATE = 8000

# Peaks envelope order
# Indicates how many times is preformed the process
# of peaks envelopes creation. First time this process
# is apllied on the audio data. The next times the process
# is apllied in peaks envelopes are recieved at the previous stage
PEAKS_ENV_ORDER = 2

# Normalization range
NORM_RANGE_MIN = -1
NORM_RANGE_MAX = 1

# Number of samples
SMPLS_NUM = 4000

# Interpolation 1D kind, can be linear, quadratic, cubic,
# nearest, zero, ...
INTER_1D_KIND = "linear"

# Classification classes
CLASSES = ["Hammer", "Jackhammer", "Drill"]

# Size of test part
TEST_SIZE = 0.2

# Window step
WIN_STEP = SMPLS_NUM // 2

# if the maximal value of the checked samples chunk is bellow
# then specified value then such chunk must be ognored.
# this value is calculated from the maximal absolute value of
# all the samples: 
# maximal value * VAL_IGNORED
VAL_IGNORED = 0.25

