import torch
#import torchaudio
import librosa
import torch.nn.functional as F
import os
import random
import pandas as pd
from glob import glob
from pathlib import Path
import numpy as np


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

        print(f'k_fold: {k_fold}, i_fold: {i_fold}, t_ratio: {t_ratio} '\
                f'r_seed: {r_seed}, l_step: {l_step}')

        # seperate gam_id in train, val, test

        dataset = glob(f'{root}/*/*.wav')

        # get set of gam_id
        gids = set()
        for file in dataset:
            field = Path(file).stem.split('_')
            gid = field[0]
            gids.add(gid)
        gids = sorted(gids)
        #print(gids)

        random.shuffle(gids)
        #print(gids)

        # test set
        test_set_index = int(len(gids) * t_ratio)
        test_set = gids[0:test_set_index] 
        gids = gids[test_set_index:]

        # val_set
        val_set_start = int(i_fold * len(gids) / k_fold)
        val_set_end = int((i_fold + 1) * len(gids) / k_fold)
        #print(val_set_start, val_set_end)
        val_set = gids[val_set_start:val_set_end]

        # train_set
        train_set = [ x for x in gids if x not in val_set ]

        '''
        print(f'test_set: {test_set}')
        print(f'val_set: {val_set}')
        print(f'train_set: {train_set}')
        '''

        dataset_mode = { 'train': train_set, 'val': val_set, 'test': test_set }
        self.meta = list()
        for file in dataset:
            field = Path(file).stem.split('_')
            gid = field[0]
            if gid in dataset_mode[mode]:
                self.meta.append(file)

        self.meta = sorted(self.meta)
        print(f'{mode} dataset: {len(self.meta)}')

        # load label.csv
        csv_path = glob(f'{root}/*/label.csv')
        assert len(csv_path) == 1, 'one csv_path should exits'
        csv_path = csv_path[0]

        label_dict = dict()
        with open(csv_path) as f:
            lines = f.readlines()
            lines = lines[1:]
            for line in lines:
                field = line.split(',')
                label_dict[field[0]] = self._find_nearest()
                print(field[0], field[2], label_dict[field[0]])


    def _find_nearest(value):
        idx = (np.abs(self.float_labels - value)).argmin()
        return self.labels[idx]


    def _get_labels(self):

        # from 5 to 75, 15 classes in case of l_step 5
        self.labels = list()
        for i in range(5, 80, self.l_step):
            self.labels.append(f'{i:02}')

        self.labels = sorted(self.labels)
        self.float_labels = np.asarray(sorted([ float(x) for x in self.labels ]))

        print(f'labels: {self.labels}')
        print(f'float_labels: {self.float_labels}')

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
