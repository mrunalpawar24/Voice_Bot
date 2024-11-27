from pyannote.audio import Pipeline
import whisper
from textblob import TextBlob
from pydub import AudioSegment
import os

# Initialize speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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"
)

# File paths
audio_file = "C:\\Users\\Lenovo\\Desktop\\mrunal.project\\record.wav"
output_file = "speaker_transcription.txt"

# Perform speaker diarization
diarization = pipeline(audio_file)

# Split audio file into speaker-specific segments
audio = AudioSegment.from_wav(audio_file)
segments_dir = "segments"
os.makedirs(segments_dir, exist_ok=True)

# Process diarization results
speaker_segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segment_path = os.path.join(segments_dir, f"{speaker}_{turn.start:.2f}_{turn.end:.2f}.wav")
    speaker_segments.append((speaker, segment_path))
    segment_audio = audio[int(turn.start * 1000):int(turn.end * 1000)]
    segment_audio.export(segment_path, format="wav")

# Load Whisper model
whisper_model = whisper.load_model("base")

# Transcribe each speaker segment
transcriptions = []
for speaker, segment_path in speaker_segments:
    transcription = whisper_model.transcribe(segment_path)
    transcriptions.append((speaker, transcription["text"]))

# Write results to file
with open(output_file, "w") as f:
    for speaker, text in transcriptions:
        f.write(f"{speaker}: {text}\n")
        print(f"{speaker}: {text}")

# Optional: Perform sentiment analysis on the combined transcription
full_text = " ".join([text for _, text in transcriptions])
sentiment = TextBlob(full_text).sentiment
print("\nSentiment Analysis:")
print(f" - Polarity: {sentiment.polarity}")
print(f" - Subjectivity: {sentiment.subjectivity}")

# Clean up temporary audio segments
import shutil
shutil.rmtree(segments_dir)

