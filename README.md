# A Taiko-no-Tatsujin Chart Generator

## Description
This project generate Taiko-no-Tatsujin chart file .tja from a given song file, making use of musical onset detection. Two different approaches are provided: Music Information Retrieval (MIR) with Librosa and machine learning-based method with Convolutional Neural Network (CNN)

## Requirements
- python3
- librosa
- numpy
- soundfile
- demucs
- flask
- pydub
- [ffmpeg](https://www.ffmpeg.org/download.html) (Required by pydub)
- pytorch (for CNN method)
- tqdm (for CNN method)

## Running Samples
### Install Requirements
```
$ pip install -r reqirement.txt
```
Please ensure that FFmpeg is installed in your device [(Installation Link)](https://www.ffmpeg.org/download.html). Input following command in console to check whether FFmpeg is succesffully installed.
```
$ ffmpeg -version
```

### Activate Flask API server
```
$ python app.py
```
Wait until you can see this message in the console
```
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```
Then you can see the webpage by browsing http://127.0.0.1:5000/ in your browser

### Sample Usage
Click the **_Browse File_** button or drag a folder over the file upload area to choose a audio file to upload. \

Input the starting time and finish time of the song to trim the audio.\
Recommanded audio length is around **1:30 (90s)**.\
Click the **_Trim Audio_** button to trim the audio.

Click the **_Upload and Detect BPM_** button to upload the audio to backend.

Fill in the following information for the chart:
- **Song name**: displayed name of the song
- **Difficulty**: set the average hits/sec 
- **Minimum note unit**: distance between consecutative notes
- **Detected BPM**: beats per minute (BPM) of the song.
- **Generation Method**: generate chart with CNN or Librosa.

Click the **_Generate Chart_** button to start chart generation (May take a while)

The generated chart will be automatically downloaded\
Other files are generated under `static/generated_files/` folder

### How To Play
The .tja file can be directly imported into official Taiko-no-Tatsujin game.\
You may try the charts in **unofficial** [Taiko Web websites](https://cjdgrevival.com).\
Upload the folder containing the trimmed .ogg audio and .tja chart to play

### Notice
**All audio file formats are supported as upload file in this project**\
**However, only .ogg audio formats are supposted by official Taiko-no-Tatsujin game**


## References

- [Musical Onset Detection with Convolutional Neural Networks](https://github.com/seiichiinoue/odcnn)
- [IMPROVED MUSICAL ONSET DETECTION WITH CONVOLUTIONAL NEURAL NETWORKS](http://www.ofai.at/~jan.schlueter/pubs/2014_icassp.pdf)
- [Musical Data Processing with CNN](https://qiita.com/woodyOutOfABase/items/01cc43fafe767d3edf62)
- [Automatic Note Generator for a rhythm game with Deep Learning](https://medium.datadriveninvestor.com/automatic-drummer-with-deep-learning-3e92723b5a79)
- [Demucs Music Source Separation](https://github.com/facebookresearch/demucs/tree/main)
