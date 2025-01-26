import librosa
import numpy as np
from dtw import accelerated_dtw
from scipy.spatial.distance import euclidean
import sounddevice as sd
import wave
import streamlit as st




# Step 1: Load Audio Files
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)  # Load audio file
    return y, sr

# Step 2: Record Audio from Microphone
def record_audio(output_file, duration=10, sr=22050):
    print("Recording... Speak into the microphone.")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    print("Recording complete.")

    # Save the recording to a file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sr)
        wf.writeframes((recording * 32767).astype(np.int16).tobytes())

# Step 3: Extract Features (Pitch, Timing)
def extract_features(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    times = librosa.times_like(pitches, sr=sr)
    return pitches, times

# Step 4: Align Lyrics with Teacher's Audio (Forced Alignment)
def align_lyrics_to_audio(lyrics, audio_features, sr):
    timestamps = []
    word_duration = len(audio_features) / len(lyrics.split())  # Approx duration per word
    for i, word in enumerate(lyrics.split()):
        start = i * word_duration / sr
        end = (i + 1) * word_duration / sr
        timestamps.append((word, start, end))
    return timestamps

# Step 5: Compare Features Between Teacher and Student
def compare_audio_features(teacher_pitches, student_pitches, teacher_times, student_times):
    # Ensure pitches are 1D by collapsing magnitudes into scalar values
    teacher_pitch_values = teacher_pitches.max(axis=0)  # Take the maximum pitch for each frame
    student_pitch_values = student_pitches.max(axis=0)  # Take the maximum pitch for each frame

    # Dynamic Time Warping to align student and teacher features
    _, _, _, path = accelerated_dtw(
        teacher_pitch_values.reshape(-1, 1),  # Reshape for DTW input
        student_pitch_values.reshape(-1, 1),
        dist=euclidean
    )

    errors = []
    t_path, s_path = path  # Unpack the alignment paths
    for t_idx, s_idx in zip(t_path, s_path):
        # Compare scalar pitch values
        if abs(teacher_pitch_values[t_idx] - student_pitch_values[s_idx]) > 0.5:  # 0.5 semitone threshold
            errors.append(student_times[s_idx])  # Log the timestamp of the error
    return errors


# Step 6: Highlight Errors in Lyrics
def highlight_errors_in_lyrics(lyrics, alignment, errors):
    highlighted_lyrics = []
    for word, start, end in alignment:
        if any(start <= e <= end for e in errors):
            highlighted_lyrics.append(f"*{word}*")  # Mark as incorrect
        else:
            highlighted_lyrics.append(word)
    return " ".join(highlighted_lyrics)

# Main Function
def analyze_performance(teacher_audio, lyrics, record_duration=10):
    # Record Student Audio
    student_audio = "student.wav"
    record_audio(student_audio, duration=record_duration)

    # Load Teacher and Student Audio
    teacher_y, teacher_sr = load_audio(teacher_audio)
    student_y, student_sr = load_audio(student_audio)

    teacher_pitches, teacher_times = extract_features(teacher_y, teacher_sr)
    student_pitches, student_times = extract_features(student_y, student_sr)

    print(teacher_pitches.shape)  # or len(teacher_pitches)
    print(student_pitches.shape)  # or len(student_pitches)

    alignment = align_lyrics_to_audio(lyrics, teacher_pitches, teacher_sr)
    errors = compare_audio_features(teacher_pitches, student_pitches, teacher_times, student_times)

    highlighted_lyrics = highlight_errors_in_lyrics(lyrics, alignment, errors)
    return highlighted_lyrics

# Example Usage
# teacher_audio = "/home/cse/Documents/ShlokaSDRM/pradakshina_mantra.wav"
# lyrics = """yāni kāni ca pāpāni janmāntarakṛtāni ca . tāni sarvāṇi naśyanti pradakṣiṇapade pade"""

# highlighted = analyze_performance(teacher_audio, lyrics, record_duration=10)
# print(highlighted)



# ... (keep your existing functions) ...

def main():
    st.title("Singing Performance Analyzer 🎤")

    # Upload teacher's audio
    teacher_audio = st.file_uploader("Upload Teacher's Audio (WAV)", type=["wav"])
    
    # Input lyrics
    lyrics = st.text_area("Paste Lyrics", height=150)
    
    # Record/upload student audio
    student_audio = st.file_uploader("Upload Student's Audio (WAV)", type=["wav"])
    
    if st.button("Analyze Performance") and teacher_audio and student_audio and lyrics:
        # Save uploaded files temporarily
        with open("teacher.wav", "wb") as f:
            f.write(teacher_audio.read())
        with open("student.wav", "wb") as f:
            f.write(student_audio.read())
        
        # Run analysis
        highlighted = analyze_performance("teacher.wav", lyrics)
        st.markdown("**Analysis Results:**")
        st.write(highlighted)

if __name__ == "__main__":
    main()