import numpy as np
from scipy.signal import argrelmax
from librosa.util import peak_pick
from librosa.onset import onset_detect
from music_processor import *
from model import *
import os
import librosa

"""
Idea taken from a open-source project.
We modified it to fit our goal of chart generation.
"""

def round_off(n):
    # Round off function (round to nearest integer, not nearest even)
    return int(n + 0.5) if n > 0 else int(n - 0.5)

def detection(don_inference, ka_inference, song, density, delta):
    
    # Smoothing the array with Hamming window of 5 frames
    don_inference = smooth(don_inference, 5)
    ka_inference = smooth(ka_inference, 5)
    
    # Pick the peaks and classify between "Don" and "Ka" based on the probability
    don_timestamp = (peak_pick(x = don_inference, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta= delta, wait=2)+7)
    ka_timestamp = (peak_pick(x = ka_inference, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta= delta, wait=2)+7)
    
    # Prevent index out of bound
    don_timestamp = don_timestamp[np.where(don_timestamp<=len(don_inference))]
    ka_timestamp = ka_timestamp[np.where(ka_timestamp<=len(ka_inference))]

    song.don_timestamp = don_timestamp[np.where(don_inference[don_timestamp] > ka_inference[don_timestamp])]
    song.timestamp = song.don_timestamp*512/song.samplerate
    song.ka_timestamp = ka_timestamp[np.where(ka_inference[ka_timestamp] > don_inference[ka_timestamp])]
    song.timestamp=song.ka_timestamp*512/song.samplerate


    return song


def create_tja(filename, song, songname, don_timestamp, ka_timestamp, file_name, bpm = 240, notes_unit="16th", ):

    # Calculate the minimum note interval from BPM
    note_unit_map = {"8th": 1/2, "16th": 1/4, "32th": 1/8, "64th": 1/16}
    interval = 60 / bpm * note_unit_map.get(notes_unit, 1/4) * 1000
    note_per_row = int(notes_unit[:-2])
    if (notes_unit != '8th'):
            threshold = 0.5
    else:
            threshold = 0.25
    
    # Filter to align the detected notes to the minimum note grid
    def filter(timestamps):
        starting_note = timestamps[0][0]
        snapped = []
        last_note = -1
        for timestamp, note in timestamps:
            aligned_timestamp = round_off((timestamp - starting_note) / interval) * interval
            deviation = abs((timestamp - starting_note) - aligned_timestamp)
            print(aligned_timestamp, timestamp)
            if round_off(aligned_timestamp) != round_off(last_note) and deviation <= threshold * interval:
                last_note = aligned_timestamp
                snapped.append((aligned_timestamp+starting_note, note))
        
        return snapped
    
    # Convert the notes sample to timestamps in ms
    don_with_labels = [(timestamp*512/song.samplerate*1000, '1') for timestamp in don_timestamp]
    ka_with_labels = [(timestamp*512/song.samplerate*1000, '2') for timestamp in ka_timestamp]

    # Combine the two kind of notes together
    combined_timestamp = sorted(don_with_labels + ka_with_labels, key=lambda x: x[0])
    combined_timestamp = filter(combined_timestamp)

    
    with open(filename, "w") as f:
        # Set the offset as the appear time of first note
        f.write(f'TITLE: {songname}\nSUBTITLE: --\nBPM: {bpm}\nWAVE:{file_name}\nOFFSET:{-combined_timestamp[0][0]/1000}\n#START\n')
        i = 0
        time = combined_timestamp[0][0]          
        count = 0

        consecutive = 0
        # Iterate until all notes are placed in the .tja file
        while(i < len(combined_timestamp)):
            if consecutive != 5 and time >= combined_timestamp[i][0]:
                f.write(combined_timestamp[i][1])
                i += 1
                consecutive += 1
            else:
                # Apply grace rule for placement 
                # Avoid delay caused by floating point rounding error
                if consecutive != 5 and abs(time - combined_timestamp[i][0]) < interval * 0.1:
                    f.write(combined_timestamp[i][1])
                    i += 1
                    consecutive += 1
                else:    
                    f.write('0')
                    consecutive = 0
            count += 1
            
            # Number of note per row is detemined by the minimum note unit
            # (16 notes per row for 16-th note)
            if count % note_per_row == 0:
                f.write(',\n')
            time += interval
        f.write('#END')
    
    # Calculate the note density for difficulty adjustment
    timestamp_dict = [{"time": timestamp, "note": note} for timestamp, note in combined_timestamp]
    total_length = len(timestamp_dict)
    duration = song.data.shape[0] / song.samplerate
    average_density = total_length / duration
    print(f"density: {average_density}, total: {total_length}, dur: {duration}")
    return timestamp_dict, average_density

def process_song(songpath):
    song = Audio(songpath, stereo=False)
    song.feats = fft_and_melscale(song, include_zero_cross=False)
    # Perform FFT and obtain the frequency domain features
    return song

def generate_inference(song, density, delta):
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
    
    # Obtain the probablity for each frame to be a Don or Ka note
    song = detection(don_inference, ka_inference, song, density, delta)

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

