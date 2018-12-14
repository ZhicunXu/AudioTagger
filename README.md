# AudioTagger

This audio event classifier can generate several recognized tags for given audio input, such as ‘acoustic guitar’, ‘dog bark’, and ‘male speech’. The classifier is trained using a dataset called Google AudioSet, and this dataset has an ontology which covers more than 600 classes of audio events. During the process of recognition for each second, the classifier would generate probabilities of all classes based on selected context window. Then, classes with probabilities higher than the selected threshold are outputted.

## Installation
###### For the folder with name 'train':
 - Dataset: 
    download converted numpy data from:
    https://drive.google.com/open?id=0B49XSFgf-0yVQk01eG92RHg4WTA
    tensorflow type data are from (no need to download):
    https://research.google.com/audioset/download.html

###### For the folder with name 'tagger':
  - Dowload the model from :
   https://drive.google.com/open?id=1DktaZdVf7LqFBeEgTzJedVIWxUvE8jWf
- Or train your own from the 'train' folder.
- Dowload the following auxiliary files from
 https://github.com/tensorflow/models/tree/master/research/audioset
   - mel_features.py
   - vggish_input.py
   - vggish_params.py
   - vggish_postprocess.py
   - vggish_slim.py
- Two additional data files:
    [VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt), in TensorFlow checkpoint format.
    [Embedding PCA parameters](https://storage.googleapis.com/audioset/vggish_pca_params.npz) in NumPy compressed archive format.
## Instruction
###### For training:
    python /train/main_thesis_multi_level_attention.py train --data_dir  /your_dataset_folder --workspace /your_workfolder --levels 2 3 --tag level_23
  - --data_dir: specify where you store your training dataset.
  - --workspace: specify where you want to store all the output, such as trained model, logs, probabilities
  - --selected levels for input to multi-level attention models
  - --tag give name for your model
###### For showing the training results:
    python /train/main_thesis_multi_level_attention.py get_avg_stats --workspace /your_workfolder --begin 1000 --step 1000 --end 10000
  - --workspace: specify your working folder
  - --begin/step/end: specify the range for number of iteration in order to visualize the trainining score output
 
###### For generating subtitles and probablities matrix:
    python /tagger/audio_tagging.py --fname test.mp4 --model model_27000\[0.344128\].h5 --left 3 --right 1 --threshold 0.01 --show_music_speech n
- --fname: name of the testing file, can either be .wav or .mp4 formats
- --model: give the model path
- --left/right: select the context window, for example, left 3 and right 1 means the recoginition for this current second is based on the context from previous 3 second to later 1 second.
- --threshold: classes with predicted probablities higher than the threhold will be outputted into subtitle files. Users can adjust them mannualy according to their need.
- --show_music_speech: decide if the two most common classes will be show in the subtitles. This is normally set to 'n/False/0'.

## Reference:
###### scientific paper:

Hershey, Shawn, et al. "CNN architectures for large-scale audio classification." Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on. IEEE, 2017.

Gemmeke, Jort F., et al. "Audio set: An ontology and human-labeled dataset for audio events." Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on. IEEE, 2017.

Kong, Qiuqiang, et al. "Audio set classification with attention model: A probabilistic perspective." 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018.

Yu, Changsong, et al. "Multi-level Attention Model for Weakly Supervised Audio Classification." arXiv preprint arXiv:1803.02353 (2018).

###### github:
https://github.com/tensorflow/models/tree/master/research/audioset
https://github.com/qiuqiangkong/ICASSP2018_audioset
