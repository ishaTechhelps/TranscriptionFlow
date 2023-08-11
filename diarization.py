from pyannote.audio import Pipeline

def initialize_pipeline(token):
    return Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=token)

def get_speaker_intervals(diarization):
    def annotation_to_string(diarization):
        lines = []
        for segment, _, label in diarization.itertracks(yield_label=True):
            line = f"SPEAKER file {segment.start} {segment.duration} <NA> <NA> {label} <NA> <NA>"
            lines.append(line)
        return '\n'.join(lines)

    diarization_str = annotation_to_string(diarization)

    lines = diarization_str.strip().split('\n')
    result = {}
    for line in lines:
        fields = line.split()
        if len(fields) < 7 or not fields[0].startswith("SPEAKER"):
            continue
        start_time = float(fields[2])
        duration = float(fields[3])
        end_time = start_time + duration
        speaker_name = fields[6]
        if speaker_name not in result:
            result[speaker_name] = []
        result[speaker_name].append((start_time, end_time))
    return result
