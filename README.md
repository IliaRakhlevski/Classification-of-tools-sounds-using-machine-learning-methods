# Classification of tools sounds using machine learning methods

The goal of this project to develop a model that classifies sounds of tools, for example: hammer, jackhammer, drill and etc.<br/>
The model uses 2D-CNN architecture.<br/>

Project files:<br/>
settings.py – global settings of the project: variables and constants.<br/>
wav_files_processing.py – WAV files preparation: dividing audio file into several files, extracting sequences<br/>
  from the original file and writingthem to new files, resampling.<br/>
samples_processing.py – processing samples loaded from audio files: resampling, padding, normalization, peaks envelopes creation,<br/>
  reading/writing samples from/to audio files.<br/>
peaks_envelope.py – peaks envelopes creation.<br/>
train.py – creation/training/testing of the model.<br/>
classification.py – classification of audio files.<br/>
main.py – main module of the project.<br/>
Training Results.html – results of the model training/testing. The model training was divided into several trainings.<br/>
  The results are presented in this file these are the results of the last training.<br/>
Test Classification Results.html – results of the test audio files classification.<br/>

See "Course project.pdf" for the details.<br/>
