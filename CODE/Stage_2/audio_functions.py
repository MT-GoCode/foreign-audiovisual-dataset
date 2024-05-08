import wave
import webrtcvad
from pydub import AudioSegment

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield timestamp, audio[offset:offset+n]
        timestamp += duration
        offset += n

def vad_audio(audio_path, aggressiveness=1):
    # Load audio
    audio = AudioSegment.from_file(audio_path)

    # Resample if necessary
    if audio.frame_rate not in (8000, 16000, 32000, 48000):
        audio = audio.set_frame_rate(16000)  # Resample to 16000Hz

    # Convert to mono and get bytes
    audio_mono_bytes = audio.set_channels(1).raw_data

    # Initialize VAD
    vad = webrtcvad.Vad(aggressiveness)

    frames = list(frame_generator(30, audio_mono_bytes, audio.frame_rate))
    voiced_frames = []
    for timestamp, frame in frames:
        is_speech = vad.is_speech(frame, audio.frame_rate)
        if is_speech:
            voiced_frames.append(timestamp)

    # Group voiced frames and return timestamps
    timestamps = []
    if voiced_frames:
        start = voiced_frames[0]
        end = start
        for timestamp in voiced_frames[1:]:
            if timestamp - end > 0.03:  # Gap between consecutive frames is more than 30ms
                timestamps.append((start, end))
                start = timestamp
            end = timestamp
        timestamps.append((start, end))

    return timestamps

def smooth_voice_segments(timestamp_pairs, max_gap_sec):
    """
    Merge voice segments that are separated by a gap less than or equal to max_gap_sec.

    Parameters:
    - timestamp_pairs: List of (start, end) timestamps in seconds where voice activity was detected.
    - max_gap_sec: Maximum gap between segments in seconds for them to be considered contiguous.

    Returns:
    - A new list of smoothed (start, end) timestamp pairs.
    """
    if not timestamp_pairs:
        return []

    # Start with the first segment
    smoothed_segments = [timestamp_pairs[0]]

    for current_start, current_end in timestamp_pairs[1:]:
        # Get the end of the last segment in the smoothed list
        last_end = smoothed_segments[-1][1]

        # If the gap between the current segment and the last segment is less than or equal to max_gap_sec,
        # merge the current segment with the last segment by updating the end time of the last segment
        if current_start - last_end <= max_gap_sec:
            smoothed_segments[-1] = (smoothed_segments[-1][0], current_end)
        else:
            # Otherwise, add the current segment as a new entry in the smoothed list
            smoothed_segments.append((current_start, current_end))

    return smoothed_segments

def voice_activity_to_frames(time_stamps, fps):

    frame_numbers = []
    for start_time, end_time in time_stamps:
        # Calculate the starting and ending frame numbers
        start_frame = int(round(start_time * fps))
        end_frame = int(round(end_time * fps))
        frame_numbers.append((start_frame, end_frame))

    return frame_numbers