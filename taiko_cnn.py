import numpy as np
from scipy.signal import argrelmax
from librosa.util import peak_pick
from librosa.onset import onset_detect
from music_processor import *
from model import *
import os
import librosa


def detection(don_inference, ka_inference, song):
    """detects notes disnotesiresultg don and ka"""
    
    don_inference = smooth(don_inference, 5)
    ka_inference = smooth(ka_inference, 5)

    don_timestamp = (peak_pick(x = don_inference, pre_max=1, post_max=2, pre_avg=4, post_avg=5, delta= 0.05,wait= 3)+7)  # 実際は7フレーム目のところの音
    ka_timestamp = (peak_pick(x = ka_inference, pre_max=1, post_max=2, pre_avg=4, post_avg=5, delta= 0.05,wait= 3)+7)
    song.don_timestamp = don_timestamp[np.where(don_inference[don_timestamp] > ka_inference[don_timestamp])]
    song.timestamp = song.don_timestamp*512/song.samplerate
    # print(len(song.timestamp))
    song.synthesize(diff='don')

    # song.ka_timestamp = song.don_timestamp
    song.ka_timestamp = ka_timestamp[np.where(ka_inference[ka_timestamp] > don_inference[ka_timestamp])]
    song.timestamp=song.ka_timestamp*512/song.samplerate
    # print(len(song.timestamp))
    song.synthesize(diff='ka')

    song.save("./generated_files_cnn/inferred_music.wav")

    return song


def create_tja(filename, song, songname, don_timestamp, ka_timestamp=None, bpm = 240):
    unit = round(60/bpm*4*100)

    print(unit)
    #song_path = os.path.join("./generated_files/", songname+'.ogg')
    if ka_timestamp is None:
        timestamp=don_timestamp*512/song.samplerate*100
        with open(filename, "w") as f:
            f.write(f'TITLE: {songname}\nSUBTITLE: --\nBPM: {bpm}\nWAVE:{songname}.ogg\nOFFSET:0\n#START\n')
            i = 0
            time = 0
            while(i < len(timestamp)):
                if time >= timestamp[i]:
                    f.write('1')
                    i += 1
                else:
                    f.write('0')
                if (time%(unit)) == (unit - 1):
                    f.write(',\n')
                time += 1
            f.write('#END')

    else:
        don_timestamp=np.rint(don_timestamp*512/song.samplerate*100).astype(np.int32)
        ka_timestamp=np.rint(ka_timestamp*512/song.samplerate*100).astype(np.int32)       
    
        with open(filename, "w") as f:
            f.write(f'TITLE: {songname}\nSUBTITLE: --\nBPM: {bpm}\nWAVE:{songname}.ogg\nOFFSET:0\n#START\n')
            #print('max', np.max((don_timestamp[-1],ka_timestamp[-1])))
            count = 0
            for time in range(0, np.max((don_timestamp[-1],ka_timestamp[-1]))):
                if (count % 100 != 0):
                    count += 1
                    f.write('0')
                else:
                    count = 0
                    if np.isin(time,don_timestamp) == True:
                        f.write('1')
                    elif np.isin(time,ka_timestamp) == True:
                        f.write('2')
                    else:
                        f.write('0')

                if (time%(unit)) == (unit - 1):
                    #print(f"{(time%(unit))} == {(unit - 1)}")
                    f.write(',\n')
            
            f.write('#END')

if __name__ == "__main__":
    
    songname = input("Song Name (Without .ogg):")

    cwd = os.getcwd()
    print(cwd)

    print("Song proccesing...")
    songpath = f"./generated_files_cnn/{songname}.ogg"

    # BPM detection using CNN
    y, sr = librosa.load(songpath, mono=True)
    
    # Detect tempo (BPM) and onset frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    
    print("Detected BPM", tempo.item())
    
    bpm = tempo.item()

    ipt = input("Please insert new BPM if detection is wrong: ")
    if ipt != '':
        bpm = float(ipt)

    song = Audio(songpath, stereo=False)
    song.feats = fft_and_melscale(song, include_zero_cross=False)
    print("Song processing done!")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = convNet()
    net = net.to(device)
    
    # Create don_inference data for don notes
    if torch.cuda.is_available():
        net.load_state_dict(torch.load('./models/don_model.pth'))
    else:
        net.load_state_dict(torch.load('./models/don_model.pth', map_location='cpu'))

    don_inference = net.infer(song.feats, device, minibatch=4192)
    don_inference = np.reshape(don_inference, (-1))

    
    # Create ka_inference data for ka notes
    if torch.cuda.is_available():
        net.load_state_dict(torch.load('./models/ka_model.pth'))
    else:
        net.load_state_dict(torch.load('./models/ka_model.pth', map_location='cpu'))

    ka_inference = net.infer(song.feats, device, minibatch=4192)
    ka_inference = np.reshape(ka_inference, (-1))

    song = detection(don_inference, ka_inference, song)
    create_tja(f"./generated_files_cnn/{songname}.tja", song, songname, song.don_timestamp, song.ka_timestamp, bpm)

