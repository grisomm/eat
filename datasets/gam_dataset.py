import torch
#import torchaudio
import librosa
import torch.nn.functional as F
import os
import random
import pandas as pd
from glob import glob
from pathlib import Path


class GamDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 mode,
                 k_fold,
                 i_fold,
                 r_seed,
                 t_ratio,
                 l_step,
                 segment_length,
                 sampling_rate,
                 transforms=None,
                 fold_id=None,
                 ):

        self.sampling_rate = sampling_rate
        self.root = root
        self.segment_length = segment_length
        self.transforms = transforms
        self.mode = mode

        self.k_fold = k_fold
        self.i_fold = i_fold
        self.r_seed = r_seed
        self.t_ratio = t_ratio
        self.l_step = l_step

        random.seed(r_seed)
        self._get_labels()

        print(f'k_fold: {k_fold}, i_fold: {i_fold}, t_ratio: {t_ratio}')
        print(f'r_seed: {r_seed}, l_step: {l_step}')

        # seperate gam_id in train, val, test

        dataset = glob(f'{root}/*/*.wav')

        # get set of gam_id
        gids = set()
        for file in dataset:
            field = Path(file).stem.split('_')
            gid = field[0]
            gids.add(gid)
        gids = sorted(gids)
        random.shuffle(gids)
        
        print(gids)



        if mode == 'train':
            self.meta = glob(f'{root}/*/*.wav')
        elif mode == 'val':
            self.meta = glob(f'{root}/*/validation/*/*.wav')
        elif mode == 'test':
            self.meta = glob(f'{root}/*/test/*/*.wav')

        # load label.csv

        self.meta = sorted(self.meta)

        for x in self.meta:
            print(x)
        print(len(self.meta))

    def _get_labels(self):

        # from 5 to 75, 15 classes
        self.labels = list()
        for i in range(5, 80, 5):
            self.labels.append(f'{i:02}')

        print(f'labels: {folders}')

    def __getitem__(self, index):
        fname = self.meta[index]
        if self.mode == 'test_file':
            label_name = 'test' 
        else:
            label_name = fname.split('/')[-2]

        label = self.labels.index(label_name)
        

        #audio, sampling_rate = torchaudio.load(fname)
        audio, sampling_rate = librosa.load(fname, sr=None)
        audio = torch.from_numpy(audio)
        audio.squeeze_()
        audio = 0.95 * (audio / audio.__abs__().max()).float()

        #print(fname, sampling_rate, audio.shape[0], self.segment_length)

        assert("sampling rate of the file is not as configured in dataset, will cause slow fetch {}".format(sampling_rate != self.sampling_rate))
        if audio.shape[0] >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        if self.transforms is not None:
            from datasets.audio_augs import AudioAugs
            audio = AudioAugs(self.transforms, sampling_rate, p=0.5)(audio)

        return audio.unsqueeze(0), label

    def __len__(self):
        return len(self.meta)


if __name__ == "__main__":
    pass
