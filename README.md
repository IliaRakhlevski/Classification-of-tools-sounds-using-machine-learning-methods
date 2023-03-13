# Classification of tools sounds using machine learning methods
## AFEKA - Tel-Aviv Academic College Of Engineering<br/>Department: Intelligent Systems<br/>Course: Machine Learning In Speech Processing Technologies

**Course project:** Implementation of a model that classifies sounds of tools: hammer, jackhammer, drill.<br/>
&emsp;&emsp;The model uses 2D-CNN architecture.<br/>
**Development Tools:** Python 3.7, Anaconda / Spyder / IPython.<br/>
**Libraries:** NumPy, SciPy, Pandas, Keras / TensorFlow.<br/>
**Data:** WAV files containing sounds of tools.<br/>

**Project files:**<br/>
* *settings.py* – global settings of the project: variables and constants.<br/>
* *wav_files_processing.py* – WAV files preparation: dividing audio file into several files, extracting sequences<br/>
&emsp;&emsp;from the original file and writing them to new files, resampling.<br/>
* *samples_processing.py* – processing samples loaded from audio files: resampling, padding, normalization,<br/>
&emsp;&emsp;peaks envelopes creation, reading/writing samples from/to audio files.<br/>
* *peaks_envelope.py* – peaks envelopes creation.<br/>
* *train.py* – creation/training/testing of the model.<br/>
* *classification.py* – classification of audio files.<br/>
* *main.py* – main module of the project.<br/>
* *Training Results.html* – results of the model training/testing. The model training was divided into several trainings.<br/>
&emsp;&emsp;The results are presented in this file these are the results of the last training.<br/>
* *Test Classification Results.html* – results of the test audio files classification.<br/>

In the directory “Data” are found audio files that are used for training/testing.<br/> 
Each directory contains audio files belong to the specific class:<br/> 
Drill – 32 files, Hammer – 23 files, Jackhammer – 35 files.<br/>
In the directory “Test” are found 5 audio files are used for testing.<br/>

See *"Course project.pdf"* for the details.<br/>
