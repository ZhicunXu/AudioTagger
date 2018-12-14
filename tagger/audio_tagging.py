import moviepy.editor as mp
import numpy as np
import os
from scipy.io import wavfile
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import tensorflow as tf
from keras.models import load_model
import datetime
import argparse
import time
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import pickle
####################################################################
# audioSet feature extractor takes 0.960s audio
# truncate the remainig 0.040s for each second for better indexing
####################################################################
def truncateAudioFs(data, fs):
    i = 0
    new_data=[]
    while (i+1)*fs<len(data):
        new_data.append(data[i*fs:i*fs + int(fs*0.96)])
        i = i + 1
    new_data.append(data[i*fs::])
    new_data = np.vstack(new_data)
    return new_data

####################################################################
# convert the audio file to log mel spectrogram
####################################################################
def prepareAudio(WAV_FILE):
    fs, data = wavfile.read(WAV_FILE)
    #data = data[0]
    data = truncateAudioFs(data, fs)
    data = data / 32768.0
    return vggish_input.waveform_to_examples(data, fs)

###################################################################
# conver the mp4 file to wav audio (16bit, 44100)
###################################################################
def genWave(videoName):
    clip = mp.VideoFileClip(videoName)
    if clip.audio is None:
        print('No audio to recognize in the video.')
        return 0
    audioName = videoName[:-4]+'.wav'
    # 16 bit 44100 fs PCM wav
    #print(audioName)
    clip.audio.write_audiofile(audioName, codec='pcm_s16le', verbose=1)
    return audioName

###################################################################
# get log-mel spectrograms from wav or mp4
###################################################################
def getMelSpecGram(fname):
    if not os.path.isfile(fname):
        print('File does not exists.')
        return 0
    dataType = fname[-3:]
    if dataType == 'mp4':
        audioName = genWave(fname)
        mels = prepareAudio(audioName)
    elif dataType == 'wav':
        mels = prepareAudio(fname)
    return mels

##################################################################
# extract the AudioSet features from the file
##################################################################
def getAudioSetFeatures(fname):
    pproc = vggish_postprocess.Postprocessor('vggish_pca_params.npz')
    mels = getMelSpecGram(fname)
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish_model.ckpt')
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)
        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                    feed_dict={features_tensor: mels})
        postprocessed_batch = pproc.postprocess(embedding_batch)

        sample = unit8_to_float32(postprocessed_batch)
    return sample

###################################################################
# convert the 8bit interger to float number between (-1, 1)
####################################################################
def unit8_to_float32(x):
    return (np.float32(x) - 128.) / 128.

####################################################################
# pad the AudioSet features into 10 segements
# repeat those which are less than 10 segments: ABB - ABBABBABBA
####################################################################
def pad2Ten(features):
    length = features.shape[0]
    if length <= 10:
        repetition = 10 // length + 1
        padFeatures = np.zeros((30, 128))
        start = 0
        end = length
        while start < 10:
            padFeatures[start:end] = features
            start = start + length
            end = end + length
        return padFeatures[0:10]
    else:
        return features[0:10]

#################################################################
# get the predictions for the input AudioSet features
# predictions: (number of seconds, 527 probabilities)
# left & right: specify how many segments are used from 
# left & right context
#################################################################
def tenSegModelPreds(features, modelName, left=3, right=0):
    model = load_model(modelName)
    length = features.shape[0]
    inputAll = np.zeros((length, 10, 128))
    for i in range(length):
        start = max(0, i - left)
        end = min(length, i + 1 + right)
        current = features[start:end]
        current = pad2Ten(features[start:end])
        inputAll[i] = current
    predictions = model.predict(inputAll)
    return predictions

##################################################################
# get the time indexes for the 'num'th second for the subtitles
##################################################################
def getStartEnd(num):
    START_TIME = datetime.datetime(100,1,1,0,0,0)
    start = START_TIME + datetime.timedelta(0,num*1+0.001)
    end = START_TIME + datetime.timedelta(0,(num+1)*1)
    return (str(start.time()).replace('.',',')[:-3], str(end.time())+",000")

###################################################################
# give the string output subtitles
# thres: probability threshold to determine if the class is present
# showSpeechMusic: if the speech and music class is shown
###################################################################
def giveThres(pred, thres=0.015, showSpeechMusic=False):
    with open("class_labels_indices.csv", "r") as fh:
        allclasses = fh.read().splitlines()
    classes2displaynames={int(i.split(',')[0]):i.split(',')[2] for i in allclasses[1:]}
    
    sort = np.argsort(pred)
    sort = sort[::-1]
    for i in range(527):
        if pred[sort[i+1]] < thres: break

    if i < 3:
        num_max = 3
    else:
        num_max = min(5, i)

    sort = sort[0:num_max]
    if 500 in sort:
        sort = [500]
    #sort = sort[0:i]
    if showSpeechMusic == True : names = [classes2displaynames[i] for i in sort]
    else: names = [classes2displaynames[i] for i in sort if i!=0 and i!=137]
    sent = '--'.join(names)
    return sent

#################################################################
# give overall descriptions
#################################################################
def giveSumRes(pred, thres=0.015, showSpeechMusic=False):
    print('Tagging for the whold clip.')
    with open("class_labels_indices.csv", "r") as fh:
        allclasses = fh.read().splitlines()
    classes2displaynames={int(i.split(',')[0]):i.split(',')[2] for i in allclasses[1:]}
    sort = np.argsort(pred)
    sort = sort[::-1]
    for i in range(527):
        if pred[sort[i+1]] < thres: break
    if i < 3:
        num_max = 3
    else:
        num_max = min(5, i)

    sort = sort[0:num_max]
    if not showSpeechMusic:
        sort = [index for index in sort if index!=0 and index!=137] 
    predSum = 0.0
    for i in range(len(sort)):
        predSum += pred[sort[i]]
    for index in sort:
        print(classes2displaynames[index])
        print('----------------' + str(round(pred[index]/predSum*100,2)))
#################################################################
# write the subtitles for 'num'th second
# sub is the string output
#################################################################
def subWrite(f, num, sub):
    f.write(str(num+1))
    f.write("\n")
    (start, end) = getStartEnd(num)
    f.write(start)
    f.write(' --> ')
    f.write(end)
    f.write("\n")
    f.write(sub)
    f.write("\n")
    f.write("\n")
#################################################################
# writing the whole subtitles
# thres: threshold probabilities
#################################################################
def genSubsThres(preds, fname, thres, ifShowMusicSpeech=False):
    srtName = fname[:-4] + '.srt'
    with open(srtName, "w") as f:
        for i in range(len(preds)):
            sub = giveThres(preds[i], thres, ifShowMusicSpeech)
            subWrite(f, i, sub)
#################################################################
# argparse true or false
#################################################################
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')  

def printHeatMap(preds):
    classes = pd.read_csv("class_labels_indices.csv")
    heatmap = go.Heatmap(z=np.transpose(preds), 
                     y=classes['display_name'],
                     colorscale='Earth')
                                  
    data = [heatmap]

    plot(data, filename= FILE_NAME[:-4] + '_LEFT_' + str(WINDOW_LEFT) + '_RIGHT_' + 
         str(WINDOW_RIGHT) + '.html')

if __name__ == '__main__':
    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--left', type=int, default=10)
    parser.add_argument('--right', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--show_music_speech', type=str2bool, nargs='?',
                        const=True, default= 'False')

    args = parser.parse_args()
    FILE_NAME = args.fname
    MODEL_NAME = args.model
    WINDOW_LEFT = args.left
    WINDOW_RIGHT = args.right
    THRESHOLD = args.threshold
    SHOW_MUSIC_SPEECH = args.show_music_speech
    preds = tenSegModelPreds(getAudioSetFeatures(FILE_NAME), 
                             MODEL_NAME, WINDOW_LEFT,
                             WINDOW_RIGHT)
    with open(FILE_NAME[:-4]+'.PROBS', 'wb') as f:
        pickle.dump(preds, f)
    printHeatMap(preds)
    pred = np.average(preds,axis=0)
    genSubsThres(preds, FILE_NAME, THRESHOLD, SHOW_MUSIC_SPEECH)
    giveSumRes(pred, THRESHOLD, SHOW_MUSIC_SPEECH)
    elapsed = time.time() - t
    print('The generation time is: ', elapsed)