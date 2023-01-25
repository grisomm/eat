import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from playsound import playsound
from glob import glob
import os
import random
from pathlib import Path
from pydub import AudioSegment
import soundfile as sf
import sounddevice as sd

# crop config 
low_sound = 0.018
crop_duration = 0.5 # in second

# noise setup
#noise_ratio = 0.3 
noise_base = 'dataset/noise'
noise_source = None


def record(duration = 3):

    fs = 44100
    sd.default.samplerate = fs 
    sd.default.channels = 1

    print('# recording')
    wav = sd.rec(int(duration * fs))
    sd.wait()

    start = None

    for i, sample in enumerate(wav):
        if sample > low_sound:
            start = i
            break

    if start is not None:
        size = int(fs*crop_duration)
        if len(wav) > start + int(size/2):
            wav = wav[start:start+int(size/2)]
            sd.wait()
            return wav, fs

    print('no sound')
    return None, None
    

def init_noise():

    global noise_source
    noise_paths = glob(f'{noise_base}/*')

    noise_source = np.array([])
    for noise_path in noise_paths:
        samples, sample_rate = load_sound(noise_path)
        noise_source = np.concatenate((noise_source, samples), axis=0) 

def get_random_noise(size):

    if noise_source is None:
        init_noise()

    length = noise_source.size
    start = random.randrange(length - size)
    return noise_source[start:(start+size)]

def mix_noise(wave, noise_ratio):
    noise = get_random_noise(wave.size) 
    return wave * ( 1 - noise_ratio ) + noise * noise_ratio


def load_sound(file_path):

    ext = Path(file_path).suffix

    if ext == '.wav':
        samples, sample_rate = librosa.load(file_path, sr=None)
    elif ext == '.m4a':
        audio = AudioSegment.from_file(file_path)
        wav_file_path = file_path.replace('.m4a', '.wav')
        audio.export(wav_file_path)
        samples, sample_rate = librosa.load(wav_file_path, sr=None)
        #os.remove(file_path)
    else:
        samples = None
        sample_rate = None

    return samples, sample_rate

def w2m(wave, sample_rate):
    sgram = librosa.stft(wave)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    return mel_sgram

def crop(wave, sample_rate, limit=1000000):
    segments = list()

    i = 0
    while True:
        if i >= len(wave):
            break

        w = wave[i]
        if w > low_sound:
            duration = int(sample_rate*crop_duration) 
            segment = wave[i:i+int(duration/2)]     # use half of duration ex) 0.25 sec
            segments.append(segment)
            i += duration
        else:
            i += 1

        if len(segments) >= limit:
            break

    return segments 

    
def save2wav(samples, sample_rate, out_path):
    sf.write(out_path, samples, sample_rate)

def save2msg(samples, sample_rate, out_path): 
    mel_sgram = w2m(samples, sample_rate)
    plt.figure(figsize=(4, 4))
    plt.gca().set_axis_off()
    librosa.display.specshow(mel_sgram, sr=sample_rate)
    plt.savefig(out_path, bbox_inches='tight', pad_inches = 0)
    plt.close()


# for debug

def play(wave, sample_rate):
    temp_wav = './temp.wav'
    sf.write(temp_wav, wave, sample_rate)
    playsound(temp_wav)
    os.remove(temp_wav)

def print_wav(wave, sample_rate, title=None):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(wave, sr=sample_rate)
    if title:
        plt.title(f'{title}');
    plt.show()

def print_msg(wave, sample_rate, title=None):

    mel_sgram = w2m(wave, sample_rate)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    if title:
        plt.title(f'{title}');
    plt.show()

if __name__ == '__main__':
    wav, sr = load_sound('dataset/water_finger/one_finger.wav')
    sgs = crop(wav, sr)
    for i, sg in enumerate(sgs):
        sg = mix_noise(sg)
        #print_msg(sg, sr, i)
        print(len(sg))
        if i > 5:
            break

    wav, sr = load_sound('dataset/water_finger/two_finger.wav')
    sgs = crop(wav, sr)
    for i, sg in enumerate(sgs):
        sg = mix_noise(sg)
        #print_msg(sg, sr, i)
        print(len(sg))
        if i > 5:
            break

