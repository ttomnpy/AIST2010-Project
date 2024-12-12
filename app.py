from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
from io import BytesIO
import librosa
from taiko_librosa import taiko_chart_generator, write_tja_file, synthesize_audio
from taiko_cnn import *
from pydub import AudioSegment

app = Flask(__name__)

# Set the path to 'generated_files' folder
GENERATED_FILES_FOLDER = 'static/generated_files'
os.makedirs(GENERATED_FILES_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('main.html')

@app.route('/trim', methods=['POST'])
def trim_audio():
    # Obtain the user input data from the webpage
    file = request.files['file']
    start_time = float(request.form['start_time']) * 1000 
    end_time = float(request.form['end_time']) * 1000 # in ms

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:

        # Trim the audio and save the trimmed audio in .ogg format
        audio = AudioSegment.from_file(file)
        trimmed_audio = audio[start_time:end_time]
        original_file_name = os.path.splitext(file.filename)[0]
        trimmed_file_name = f"{original_file_name}_trimmed.ogg"
        trimmed_file_path = os.path.join(GENERATED_FILES_FOLDER, trimmed_file_name)
        trimmed_audio.export(trimmed_file_path, format="ogg")

        print("trimmed:", trimmed_file_path)

        return jsonify({'trimmed_file_path': trimmed_file_path})

    except Exception as e:
        return jsonify({'error': 'Failed to trim audio', 'message': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_audio():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    print(f"Received file: {file.filename}")
    file.seek(0)

    try:
        # Read the file into memory
        file_bytes = BytesIO(file.read())

        # Detect BPM using librosa
        y, sr = librosa.load(file_bytes, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # Debug messages: Trying to understand the octave error
        first_beat_time, last_beat_time = librosa.frames_to_time((beat_frames[0], beat_frames[-1]), sr=sr)
        bpm = 60 / ((last_beat_time - first_beat_time) / (len(beat_frames) - 1))
        print("Calculated BPM", bpm)
        print(f"Detected BPM: {tempo.item()}")

    except Exception as e:
        print(e)
        return jsonify({'error': 'Failed to process the audio file', 'message': str(e)}), 500

    file_name = file.filename
    file_path = os.path.join(GENERATED_FILES_FOLDER, file_name)
    
    # Save the file for further process
    with open(file_path, 'wb') as f:
        f.write(file_bytes.getvalue())

    # Return the bpm value to display on the webpage for user reference
    return jsonify({'bpm': round_off(tempo.item()), 'file_name': file_name})

@app.route('/process', methods=['POST'])
def process():
    # Obtain the user input data from the webpage
    song_name = request.form['sname']
    difficulty = int(request.form['difficulty'])
    notes = request.form['notes']
    bpm = float(request.form['bpm'])
    file_name = request.form['file_name']
    approach = request.form['method']
    
    file_path = os.path.join(GENERATED_FILES_FOLDER, file_name)

    # Use differnet approach according to user preference
    if approach == 'librosa':
        taiko_chart, bpm_detected, ending, sr = taiko_chart_generator(file_path, max_hits_per_sec=difficulty)
        output_file = os.path.join(GENERATED_FILES_FOLDER, f"{song_name}.tja")
        synethized_file = os.path.join(GENERATED_FILES_FOLDER, f"{song_name}_syn.wav")
        
        filtered_chart = write_tja_file(output_file, taiko_chart, bpm, ending, file_name, song_name, 0, notes)
        synthesize_audio(filtered_chart, synethized_file, file_path)
    elif approach == 'cnn':
        song = process_song(file_path)

        output_file = os.path.join(GENERATED_FILES_FOLDER, f"{song_name}.tja")
        synethized_file = os.path.join(GENERATED_FILES_FOLDER, f"{song_name}_syn.wav")
        delta = 0.03
        i = 0
        while i < 100:
            song = generate_inference(song, difficulty, delta)
            filtered_chart, density = create_tja(output_file, song, song_name, song.don_timestamp, song.ka_timestamp, file_name, bpm, notes)
            if (round_off(density) <= difficulty):
                break
            else:
                delta += 0.005

        synthesize_audio(filtered_chart, synethized_file, file_path)
        if i == 100:
            raise Exception("Time out error when generating chart")

    # Return url for tja file and synthesized audio for playback in webpage and download
    return jsonify({
        'tja_file_url': url_for('static', filename=f"generated_files/{song_name}.tja"),
        'synthesized_audio_url': url_for('static', filename=f"generated_files/{song_name}_syn.wav"),
    })

if __name__ == '__main__':
    app.run(debug=True)
