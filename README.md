# A Taiko-no-Tatsujin Chart Generator

## Description
This project generate Taiko-no-Tatsujin chart file .tja from a given song file (.ogg), making use of musical onset detection.

## Requirements
- python3
- librosa
- numpy
- soundfile
- demucs
- flask
- pydub
- pytorch (for CNN method)
- tqdm (for CNN method)

## Running Samples
### Install Requirements
```
$ pip install -r reqirement.txt
```

### Activate Flask API server
```
$ python app.py
```
Wait until you can see this message in the console
```
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```
Then you can see the webpage by browsing http://127.0.0.1:5000/ on your browser

### Sample Usage
Click the _Browse File_ button or drag a folder over the file upload area to choose a audio file to upload. \

Click the _Upload and Detect BPM_ button to upload the audio to backend

Fill in the following information for the chart:
- **Song name**: displayed name of the song
- **Difficulty**: set the average hits/sec 
- **Minimum note unit**: distance between consecutative notes
- **Detected BPM**: beats per minute (BPM) of the song

Click the _Generate Chart_ button to start chart generation (May take a while)

The generated chart will be automatically downloaded\
Other files are generated under `static/generated_files/`

### Notice:
**Only .ogg audio file format is supported**


## References

- [IMPROVED MUSICAL ONSET DETECTION WITH CONVOLUTIONAL NEURAL NETWORKS](http://www.ofai.at/~jan.schlueter/pubs/2014_icassp.pdf)
- [Dance Dance Convolution](https://arxiv.org/pdf/1703.06891.pdf)
- [Musical Data Processing with CNN](https://qiita.com/woodyOutOfABase/items/01cc43fafe767d3edf62)
- [Demucs Music Source Separation](https://github.com/facebookresearch/demucs/tree/main)
