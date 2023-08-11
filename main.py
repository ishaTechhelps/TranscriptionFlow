import os
import diarization
import audio_processing

# Initialize the speaker diarization pipeline.
token = "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"
pipeline = diarization.initialize_pipeline(token)

# Diarization on the audio file.
audio_path = "/notebooks/comedy_zoom.wav"
diarization_result = pipeline(audio_path, min_speakers=2, max_speakers=5)
speakers_intervals = diarization.get_speaker_intervals(diarization_result)

# Split and save audio for each speaker.
output_directory = "/notebooks/outputs/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for speaker, intervals in speakers_intervals.items():
    output_file_path = os.path.join(output_directory, f"{speaker}.wav")
    audio_processing.split_and_merge_audio(audio_path, intervals, output_file_path)
    print(f"Saved audio for {speaker} at {output_file_path}")
