from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
from io import BytesIO
import librosa
from taiko_librosa import taiko_chart_generator, write_tja_file, synthesize_chart

app = Flask(__name__)

# Set the path to 'generated_files' folder
GENERATED_FILES_FOLDER = 'static/generated_files'
os.makedirs(GENERATED_FILES_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    """Handle audio upload and detect BPM."""
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Debugging: Log the file details
    print(f"Received file: {file.filename}")
    print(f"File size: {len(file.read())} bytes")

    # After reading, reset the pointer to the start of the file
    file.seek(0)

    try:
        # Read the file into memory (no need to save it yet)
        file_bytes = BytesIO(file.read())

        # Debugging: Check if file has been read correctly
        print(f"File size after reading into BytesIO: {len(file_bytes.getvalue())} bytes")

        # Detect BPM using librosa
        y, sr = librosa.load(file_bytes, mono=True)  # Load directly from byte buffer
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        print(f"Detected BPM: {tempo.item()}")

    except Exception as e:
        # Detailed error message for debugging
        return jsonify({'error': 'Failed to process the audio file', 'message': str(e)}), 500

    # Generate the file path for saving the audio in the same directory as the generated TJA files
    file_name = file.filename
    file_path = os.path.join(GENERATED_FILES_FOLDER, file_name)
    
    # Save the file to the same folder for later use in taiko chart generation
    with open(file_path, 'wb') as f:
        f.write(file_bytes.getvalue())

    return jsonify({'bpm': tempo.item(), 'file_name': file_name})

@app.route('/process', methods=['POST'])
def process():
    """Generate the Taiko chart and serve the .tja file."""
    song_name = request.form['sname']
    difficulty = int(request.form['difficulty'])
    notes = request.form['notes']
    bpm = float(request.form['bpm'])
    file_name = request.form['file_name']
    
    # Get the file path assuming it's in the generated_files folder
    file_path = os.path.join(GENERATED_FILES_FOLDER, file_name)

    # Generate the Taiko chart
    taiko_chart, bpm_detected, ending, sr = taiko_chart_generator(file_path, max_hits_per_sec=difficulty)
    
    # Define the output file path for the .tja file
    output_file = os.path.join(GENERATED_FILES_FOLDER, f"{song_name}.tja")
    synethized_file = os.path.join(GENERATED_FILES_FOLDER, f"{song_name}_syn.wav")
    
    # Write the .tja file
    filtered_chart = write_tja_file(output_file, taiko_chart, bpm, ending, file_name, song_name, 0, notes)
    #synthesize_chart(filtered_chart, synethized_file, os.path.join(GENERATED_FILES_FOLDER, f"{song_name}_demucs_output", "htdemucs", f"{song_name}", f"no_vocals.wav"))
    synthesize_chart(filtered_chart, synethized_file, file_path)

    # Return the .tja file for download

    #return send_file(synethized_file, as_attachment=True)
    return jsonify({
        'tja_file_url': url_for('static', filename=f"generated_files/{song_name}.tja"),
        'synthesized_audio_url': url_for('static', filename=f"generated_files/{song_name}_syn.wav"),
    })

if __name__ == '__main__':
    app.run(debug=True)
