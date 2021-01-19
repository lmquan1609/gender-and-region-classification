import pyaudio
import os
import wave
import librosa
import sys

cur_dir = sys.argv[-1][:-6]

def get_audio(WAVE_OUTPUT_FILENAME,RECORD_SECONDS = 5,CHUNK = 1024,FORMAT = pyaudio.paInt16,CHANNELS = 2,RATE=22050):    
    """Record audio which duration RECORD_SECONDS save in folder audio
    Args:
        WAVE_OUTPUT_FILENAME (string): save audio with name
        RECORD_SECONDS (int): duration audio
    Returns:
        array: path file audio
    """
    WAVE_OUTPUT_FILENAME=cur_dir+'audio\\'+WAVE_OUTPUT_FILENAME+".wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return WAVE_OUTPUT_FILENAME