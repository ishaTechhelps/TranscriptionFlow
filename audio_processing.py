from pydub import AudioSegment

def split_and_merge_audio(audio_path, time_segments, output_path):
    audio = AudioSegment.from_wav(audio_path)
    segments = [audio[start*1000:end*1000] for start, end in time_segments]
    merged_audio = sum(segments, AudioSegment.empty())
    merged_audio.export(output_path, format="wav")
