import numpy as np
from scipy.signal import argrelmax
from librosa.util import peak_pick
from librosa.onset import onset_detect
from music_processor import *
from model import *
import os
import librosa

"""
This code file is downloaded from a open-source project.
We modified this file to fit our goal of chart generation.
"""

def round_off(n):
    return int(n + 0.5) if n > 0 else int(n - 0.5)

def detection(don_inference, ka_inference, song, density):
    """detects notes disnotesiresultg don and ka"""
    
    don_inference = smooth(don_inference, 5)
    ka_inference = smooth(ka_inference, 5)
    i = 0
    delta = 0.03
    while i < 100:
        i += 1
        don_timestamp = (peak_pick(x = don_inference, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta= delta, wait=5)+7)
        ka_timestamp = (peak_pick(x = ka_inference, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta= delta, wait=5)+7)

        song.don_timestamp = don_timestamp[np.where(don_inference[don_timestamp] > ka_inference[don_timestamp])]
        song.timestamp = song.don_timestamp*512/song.samplerate
        don_num = len(song.timestamp)
        song.ka_timestamp = ka_timestamp[np.where(ka_inference[ka_timestamp] > don_inference[ka_timestamp])]
        song.timestamp=song.ka_timestamp*512/song.samplerate
        ka_num = len(song.timestamp)
        
        total_length = don_num + ka_num
        duration = song.data.shape[0] / song.samplerate
        average_density = total_length / duration
        print(f"density: {average_density}, delta = {delta}")
        if (round_off(average_density) <= density):
            break
        else:
            delta += 0.01
            
    if i == 100:
        raise Exception("Error when calculating density, time out error")

    return song


def create_tja(filename, song, songname, don_timestamp, ka_timestamp, file_name, bpm = 240, notes_unit="16th", ):


    note_unit_map = {"8th": 1/2, "16th": 1/4, "32th": 1/8, "64th": 1/16}
    interval = 60 / bpm * note_unit_map.get(notes_unit, 1/4) * 1000
    note_per_row = int(notes_unit[:-2])
    threshold = 0.25
    
    def filter(timestamps):
        starting_note = round_off((timestamps[0][0]) / interval) * interval
        snapped = []
        for timestamp, note in timestamps:
            aligned_timestamp = round_off((timestamp - starting_note) / interval) * interval
            deviation = abs((timestamp - starting_note) - aligned_timestamp)
            #print(deviation, threshold * interval)
            if deviation <= threshold * interval:
                snapped.append((aligned_timestamp+starting_note, note))
        
        return snapped
    

    don_with_labels = [(timestamp*512/song.samplerate*1000, '1') for timestamp in don_timestamp]
    ka_with_labels = [(timestamp*512/song.samplerate*1000, '2') for timestamp in ka_timestamp]

    combined_timestamp = sorted(don_with_labels + ka_with_labels, key=lambda x: x[0])
    combined_timestamp = filter(combined_timestamp)

    
    with open(filename, "w") as f:
        f.write(f'TITLE: {songname}\nSUBTITLE: --\nBPM: {bpm}\nWAVE:{file_name}\nOFFSET:{-combined_timestamp[0][0]/1000}\n#START\n')
        i = 0
        time = combined_timestamp[0][0]          
        count = 0
        while(i < len(combined_timestamp)):
            if time >= combined_timestamp[i][0]:
                f.write(combined_timestamp[i][1])
                i += 1
            else:
                f.write('0')
            count += 1
            if count % note_per_row == 0:
                f.write(',\n')
            time += interval
        f.write('#END')
    
    timestamp_dict = [{"time": timestamp, "note": note} for timestamp, note in combined_timestamp]
    
    return timestamp_dict

def process_song(songpath):
    song = Audio(songpath, stereo=False)
    song.feats = fft_and_melscale(song, include_zero_cross=False)

    return song

def generate_inference(song, density):
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

    song = detection(don_inference, ka_inference, song, density)

    return song


if __name__ == "__main__":

    songname = input("Song Name (Without .ogg):")
    #songname = "kawaii"
    cwd = os.getcwd()
    print(cwd)

    print("Song proccesing...")
    songpath = f"./generated_files_cnn/{songname}.ogg"  ##

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

