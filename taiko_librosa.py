import numpy as np
from librosa.util import peak_pick
from librosa.onset import onset_detect
import librosa
import os
import soundfile as sf

def taiko_chart_generator(file_path, max_hits_per_sec):
    """
    Generate a taiko chart with note classification for TJA file.

    Args:
    - file_name (str): Path to the audio file.
    - min_note_unit (str): Minimum note unit ('8th', '16th', '32th', etc.).
    - max_hits_per_second (int): Maximum hits allowed per second.

    Returns:
    - taiko_chart (list): List of chart entries with time and note type.
    - bpm (float): Detected BPM of the audio.
    - ending (int): Time in milliseconds of the end of the chart.
    """
    # Load the audio
    y, sr = librosa.load(file_path, mono=True)
    
    # Detect tempo (BPM) and onset frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    print("Detected BPM", tempo.item())
    
    bpm = tempo.item()

    '''ipt = input("Please insert new BPM if detection is wrong: ")
    if ipt != '':
        bpm = float(ipt)'''

    # Onset detection
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=librosa.onset.onset_strength(y=y, sr=sr),
        backtrack=True,
        sr=sr,
        pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.05, wait=3
    )
    
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Calculate average hits per second
    def calculate_density(onset_times):
        # Calculate the number of hits per second
        total_hits = len(onset_times)
        total_duration = onset_times[-1]  # Duration in seconds
        avg_hits_per_sec = total_hits / total_duration
        return avg_hits_per_sec

    avg_density = calculate_density(onset_times)
    print(f"Average Hits per Second: {avg_density}")

    
    # Adjust `wait` parameter if density exceeds max hits per second
    wait = 3  # Initial value for `wait`
    while round(avg_density) > max_hits_per_sec:
        print(f"Now: {avg_density}, Density exceeded {max_hits_per_sec} hits per second. Adjusting `wait`...")
        wait += 0.1  # Increase `wait` to reduce density
        print(f"New wait value: {wait}")

        # Regenerate the onset frames with the new `wait` value
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=librosa.onset.onset_strength(y=y, sr=sr),
            backtrack=True,
            sr=sr,
            pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.05, wait=wait
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        avg_density = calculate_density(onset_times)  # Recalculate density
    
    print(f"Final Density: {avg_density} hits per second after adjustment.")

    
    # Classify onsets into 'don', 'DON', 'ka', or 'KA'
    taiko_chart = classify_don_ka(y, sr, onset_times)

    ending = onset_times[-1]

    return taiko_chart, bpm, ending, sr

def classify_don_ka(y, sr, onsets):
    # Initialize lists to store spectral features
    spectral_centroids = []

    # For each onset, extract spectral centroid (or another spectral feature)
    for onset in onsets:
        onset_samples = librosa.time_to_samples(onset, sr=sr)
        onset_window = y[onset_samples - int(0.1 * sr):onset_samples + int(0.1 * sr)]

        # Compute the spectral content (using STFT and spectral centroid)
        D = librosa.stft(onset_window)
        magnitude, _ = librosa.magphase(D)
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude).mean()
        spectral_centroids.append(spectral_centroid)

    # Compute the median spectral centroid value to serve as a threshold
    median_spectral_centroid = np.median(spectral_centroids)

    # Classify each onset as "don" or "ka" based on its relative frequency content
    taiko_chart = []
    for onset, centroid in zip(onsets, spectral_centroids):
        if centroid < median_spectral_centroid:
            note = "1"  # Low spectral centroid -> "don"
        else:
            note = "2"  # High spectral centroid -> "ka"

        taiko_chart.append({"time": onset * 1000, "note": note}) # time in ms

    return taiko_chart

def synthesize_chart(taiko_chart, output_file, song_path):
    """
    Synthesize audio with don and ka sounds based on the taiko chart.

    Args:
    - taiko_chart (list): List of chart entries with time (in seconds) and note type.
    - sr (int): Sample rate of the output audio.
    - output_file (str): Path to the output .wav file.
    - don (str): Path to the "don" sound file.
    - ka (str): Path to the "ka" sound file.
    """

    don="./don.wav"
    ka="./ka.wav"
    song, song_sr = librosa.load(song_path, mono=True)
    
    don_sound, don_sr = librosa.load(don, sr=song_sr, mono=True)  # Resample to match song_sr
    ka_sound, ka_sr = librosa.load(ka, sr=song_sr, mono=True)    # Resample to match song_sr

    don_len = len(don_sound)
    ka_len = len(ka_sound)

    # Calculate the duration of the output audio
    total_duration = int((taiko_chart[-1]["time"] / 1000) * song_sr) + max(don_len, ka_len)
    
    # Create an output buffer based on the original song's duration
    synthesized_audio = np.copy(song)


    for entry in taiko_chart:
        time = int((entry["time"] / 1000) * song_sr)  # Convert time from ms to samples
        if entry["note"] == "1":  # don
            # Ensure we don't go out of bounds of the synthesized_audio array
            end_idx = min(time + don_len, len(synthesized_audio))  # Ensure end index is within bounds
            synthesized_audio[time:end_idx] += don_sound[:end_idx - time]  # Only add the part that fits
        elif entry["note"] == "2":  # ka
            # Ensure we don't go out of bounds of the synthesized_audio array
            end_idx = min(time + ka_len, len(synthesized_audio))  # Ensure end index is within bounds
            synthesized_audio[time:end_idx] += ka_sound[:end_idx - time]  # Only add the part that fits

    # Normalize audio to avoid clipping
    synthesized_audio = np.clip(synthesized_audio, -1, 1)


    # Write to output file
    sf.write(output_file, synthesized_audio, song_sr)
    print(f"Synthesized audio saved to {output_file}")


def write_tja_file(output_file, taiko_chart, bpm, ending, song , title="Generated Song", level=5, min_note_unit = '16th'):
    """
    Write the taiko chart into a TJA file.

    Args:
    - output_file (str): Path to the output TJA file.
    - taiko_chart (list): List of chart entries with time and note type.
    - bpm (float): BPM of the chart.
    - ending (int): End time of the chart in milliseconds.
    - title (str): Title of the chart.
    - level (int): Difficulty level.
    """
    with open(output_file, "w") as f:
        current_time = 0
        chart_string = ""
        
        
        note_unit_map = {"8th": 1/2, "16th": 1/4, "32th": 1/8, "64th": 1/16}
        smallest_unit_duration = (60 / bpm) * note_unit_map["32th"] * 1000
        min_note_interval = (60 / bpm) * note_unit_map.get(min_note_unit, 1/4)
        time_increment = min_note_interval * 1000           # Set the unit to milliseconds
        print("min note interval", min_note_interval, ' time increment', time_increment)
        
        if not taiko_chart:
            raise ValueError("The taiko_chart is empty.")
        
        threshold = min_note_interval / 2

        filtered_chart = []
        """for entry in taiko_chart:
            aligned_time = round(entry["time"] / smallest_unit_duration) * smallest_unit_duration
            if (basetime == -1):
                basetime = aligned_time
            #aligned_chart.append({"time": aligned_time, "note": entry["note"]})
            #print((entry["time"] ) , time_increment)
            
            if (abs((aligned_time - basetime) % min_note_interval) < threshold) or abs((aligned_time - basetime) % min_note_interval) - min_note_interval  < threshold:
                filtered_chart.append({"time": aligned_time, "note": entry["note"]})"""
        for entry in taiko_chart:
            aligned_time = round(entry["time"] / min_note_interval) * min_note_interval
            if filtered_chart and abs(aligned_time - filtered_chart[-1]["time"]) < threshold:
                continue  # Skip duplicate notes or notes too close together
            filtered_chart.append({"time": aligned_time, "note": entry["note"]})


        f.write(f"TITLE:{title}\n")
        f.write(f"BPM:{bpm:.2f}\n")
        f.write(f"WAVE:{song}\n")
        f.write(f"OFFSET:0.0\n")
        f.write("COURSE:Oni\n")
        f.write(f"LEVEL:{level}\n")
        f.write("MEASURE:4/4\n")
        f.write("#START\n")
        

        # Generate note sequence with rests ('0') where no notes are present
        for entry in filtered_chart:
            while current_time < entry['time']:
                chart_string += "0"  # Rest
                current_time += time_increment
            chart_string += entry['note']  # Add note
            current_time += time_increment

        while current_time < ending:
            chart_string += "0"  # Fill remaining time with rests
            current_time += time_increment

        note_per_row = int(min_note_unit[:-2])

        # Split chart into rows of 16 notes
        rows = [chart_string[i:i + note_per_row] for i in range(0, len(chart_string), note_per_row)]
        f.write(",\n".join(rows))
        f.write("\n#END\n")
    return filtered_chart




"""
#cwd = os.getcwd()
path, filename = os.path.split(os.path.realpath(__file__))
print(path)
cwd = path


songname = input("Song Name (Without .ogg): ")
songname = songname + '.ogg'
songname = "viva.ogg"

songpath = os.path.join(os.path.join(cwd, 'generated_files_librosa'), songname)
print("Song path: ", songpath)

max_hits_per_sec = float(input("Max hit per second: "))
min_note_unit = input("Minimum note unit [8th/16th/32th]: ")
map_title = input("Title of the map: ")

taiko_chart, bpm, ending, sr = taiko_chart_generator(
    songpath,
    max_hits_per_sec = max_hits_per_sec
)



output_file = cwd + "/generated_files_librosa/"+ songname[:-4] + ".tja"
write_tja_file(output_file, taiko_chart, bpm, ending, title=map_title, level=1, song = songname, min_note_unit=min_note_unit)

print(f"Taiko chart written to {output_file}")"""