import numpy as np
from librosa.util import peak_pick
from librosa.onset import onset_detect
import librosa
import os
import soundfile as sf
import demucs.separate

def round_off(n):
    return int(n + 0.5) if n > 0 else int(n - 0.5)

def isolate_background(file_path):

    print("Separating the file...")
    
    input_dir = os.path.dirname(file_path)
    song_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(input_dir, f"{song_name}_demucs_output")
    os.makedirs(output_dir, exist_ok=True)

    demucs.separate.main(["--out", output_dir, "--two-stems=vocals", file_path])

    no_vovals_path = os.path.join(output_dir, "htdemucs", f"{song_name}", f"no_vocals.wav")

    if not os.path.exists(no_vovals_path):
        raise FileNotFoundError("File not found. Check the output directory.")

    background_audio, sr = librosa.load(no_vovals_path, mono=True)

    return background_audio, sr


def taiko_chart_generator(file_path, max_hits_per_sec):

    # Load the audio
    y, sr = isolate_background(file_path)

    # Detect tempo (BPM) and onset frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    print("Detected BPM", tempo.item())

    
    bpm = tempo.item()

    # Onset detection
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=librosa.onset.onset_strength(y=y, sr=sr),
        backtrack=True,
        sr=sr,
        pre_max=1, post_max=2, pre_avg=2, post_avg=2, delta=0.03, #wait=3
    )
    
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Calculate average hits per second
    def calculate_density(onset_times):
        total_hits = len(onset_times)
        total_duration = onset_times[-1]
        avg_hits_per_sec = total_hits / total_duration
        return avg_hits_per_sec

    avg_density = calculate_density(onset_times)
    print(f"Average Hits per Second: {avg_density}")

    # Adjust 'delta' parameter if density exceeds max hits per second
    delta = 0.02
    while round_off(avg_density) > max_hits_per_sec:
        print(f"Now: {avg_density}, Density exceeded {max_hits_per_sec} hits per second. Adjusting 'delta'...")
        delta += 0.005  # Increase 'delta' to reduce density
        print(f"New delta value: {delta}")

        # Regenerate the onset frames with the new 'wait' value
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=librosa.onset.onset_strength(y=y, sr=sr),
            backtrack=True,
            sr=sr,
            pre_max=1, post_max=2, pre_avg=2, post_avg=2, delta=delta, #wait=3
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        avg_density = calculate_density(onset_times)  # Recalculate density
    
    print(f"Final Density: {avg_density} hits per second after adjustment.")

    
    # Classify onsets into 'don', 'ka'
    taiko_chart = classify_don_ka(y, sr, onset_times)

    ending = onset_times[-1]

    return taiko_chart, bpm, ending, sr

def classify_don_ka(y, sr, onsets):
    spectral_centroids = []

    for onset in onsets:
        onset_samples = librosa.time_to_samples(onset, sr=sr)
        onset_window = y[onset_samples - int(0.1 * sr):onset_samples + int(0.1 * sr)]

        magnitude = np.abs(librosa.stft(onset_window))
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude).mean()
        spectral_centroids.append(spectral_centroid)

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

def synthesize_audio(taiko_chart, output_file, song_path):
    don="./don.wav"
    ka="./ka.wav"
    song, song_sr = librosa.load(song_path, mono=True)

    don_sound, don_sr = librosa.load(don, sr=song_sr, mono=True)
    ka_sound, ka_sr = librosa.load(ka, sr=song_sr, mono=True)

    don_len = len(don_sound)
    ka_len = len(ka_sound)

    synthesized_audio = np.copy(song)

    for entry in taiko_chart:
        time = int((entry["time"] / 1000) * song_sr)  
        if entry["note"] == "1":  
            end_idx = min(time + don_len, len(synthesized_audio))  
            synthesized_audio[time:end_idx] += don_sound[:end_idx - time]  
        elif entry["note"] == "2":  
            end_idx = min(time + ka_len, len(synthesized_audio))  
            synthesized_audio[time:end_idx] += ka_sound[:end_idx - time] 

    sf.write(output_file, synthesized_audio, song_sr)
    print(f"Synthesized audio saved to {output_file}")


"""def write_tja_file(output_file, taiko_chart, bpm, ending, song , title="Generated Song", level=5, min_note_unit = '16th'):
    with open(output_file, "w") as f:
        
        
        
        note_unit_map = {"8th": 1/2, "16th": 1/4, "32th": 1/8, "64th": 1/16}
        smallest_unit_duration = (60 / bpm) * note_unit_map["32th"] * 1000
        min_note_interval = (60 / bpm) * note_unit_map.get(min_note_unit, 1/4)
        time_increment = min_note_interval * 1000           # Set the unit to milliseconds
        print("min note interval", min_note_interval, ' time increment', time_increment)
        
        if not taiko_chart:
            raise ValueError("The taiko_chart is empty.")
        
        threshold = min_note_interval / 2

        filtered_chart = []
        '''for entry in taiko_chart:
            aligned_time = round(entry["time"] / smallest_unit_duration) * smallest_unit_duration
            if (basetime == -1):
                basetime = aligned_time
            #aligned_chart.append({"time": aligned_time, "note": entry["note"]})
            #print((entry["time"] ) , time_increment)
            
            if (abs((aligned_time - basetime) % min_note_interval) < threshold) or abs((aligned_time - basetime) % min_note_interval) - min_note_interval  < threshold:
                filtered_chart.append({"time": aligned_time, "note": entry["note"]})'''
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
        
        current_time = 0
        chart_string = ""
        
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

def write_tja_file(output_file, taiko_chart, bpm, ending, song , title="Generated Song", level=5, min_note_unit = '16th'):
    with open(output_file, "w") as f:
        
        note_unit_map = {"8th": 1/2, "16th": 1/4, "32th": 1/8, "64th": 1/16}
        min_note_interval = (60 / bpm) * note_unit_map.get(min_note_unit, 1/4) * 1000  # Set the unit to milliseconds
        print("min note interval", min_note_interval)
        
        if not taiko_chart:
            raise ValueError("The taiko_chart is empty.")
        
        threshold = 0.5

        filtered_chart = []
        
        def filter_chart(chart):
            if not chart:
                return []
            
            starting_time = chart[0]["time"]
            #starting_time = 0
            #starting_time = round_off((chart[0]["time"]) / min_note_interval) * min_note_interval
            snapped_chart = []
            for entry in chart:
                timestamp, note = entry["time"], entry["note"]
                
                aligned_time = round_off((timestamp-starting_time) / min_note_interval) * min_note_interval
                deviation = abs((timestamp - starting_time) - aligned_time)
                #print(timestamp, aligned_time+starting_time, deviation)
                if deviation <= threshold * min_note_interval:
                    snapped_chart.append({"time": aligned_time + starting_time, "note": note})
            return snapped_chart

        filtered_chart = filter_chart(taiko_chart)

        #filtered_chart = taiko_chart
        first_entry_time = filtered_chart[0]["time"]

        f.write(f"TITLE:{title}\n")
        f.write(f"BPM:{bpm:.2f}\n")
        f.write(f"WAVE:{song}\n")
        f.write(f"OFFSET:{-first_entry_time/1000}\n")
        f.write("COURSE:Oni\n")
        f.write(f"LEVEL:{level}\n")
        f.write("MEASURE:4/4\n")
        f.write("#START\n")
        
        current_time = first_entry_time
        chart_string = ""
        
        # Generate note sequence
        for entry in filtered_chart:
            while current_time < entry['time']:
                if abs(current_time - entry['time']) < min_note_interval * 0.1:
                    break
                chart_string += "0"  # Rest
                current_time += min_note_interval
            chart_string += entry['note']  # Add note
            current_time += min_note_interval

        while current_time < ending:
            chart_string += "0"  # Fill remaining time with rests
            current_time += min_note_interval

        note_per_row = int(min_note_unit[:-2])

        # Split chart into rows
        rows = [chart_string[i:i + note_per_row] for i in range(0, len(chart_string), note_per_row)]
        f.write(",\n".join(rows))
        f.write("\n#END\n")
    return filtered_chart



if __name__ == "__main__":
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

    print(f"Taiko chart written to {output_file}")