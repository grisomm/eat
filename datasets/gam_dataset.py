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
import json


class GamDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 mode,
                 k_fold,
                 i_fold,
                 r_seed,
                 t_ratio,
                 l_step,
                 dif,
                 ran,
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
        self.dif = dif
        self.ran = ran

        random.seed(r_seed)
        self._get_labels()

        #print(f'k_fold: {k_fold}, i_fold: {i_fold}, t_ratio: {t_ratio} '\
        #        f'r_seed: {r_seed}, l_step: {l_step}, dif: {dif}, ran: {ran}')

        # seperate gam_id in train, val, test

        label_path = glob(f'{root}/*/label.json')
        assert len(label_path) == 1, 'one label json should exits'
        label_path = label_path[0]

        with open(label_path) as f:
            source_label_json = json.load(f)

        label_json = dict()
        for gid in source_label_json:
            if source_label_json[gid]['dif'] <= dif:
                
                # check range of gam_id
                gam_id = int(gid.split('-')[0])
                if ran is None:
                    label_json[gid] = source_label_json[gid]
                elif gam_id >= ran[0] and gam_id <= ran[1]: 
                    label_json[gid] = source_label_json[gid]

        # get set of gam_id
        # and label dictionary for each gid
        gam_ids = set()
        self.label_dict = dict()
        for gid in label_json: 
            field = gid.split('-')
            gam_id = field[0]
            gam_ids.add(gam_id)

            ave = label_json[gid]['ave']
            self.label_dict[gid] = self._find_nearest(float(ave))
            #print(field[0], field[2], self.label_dict[field[0]])

        gam_ids = sorted(gam_ids)
        #print(gam_ids)

        random.shuffle(gam_ids)
        #print(gam_ids)

        # test set
        test_set_index = int(len(gam_ids) * t_ratio)
        test_set = gam_ids[0:test_set_index] 
        gam_ids = gam_ids[test_set_index:]

        # val_set
        val_set_start = int(i_fold * len(gam_ids) / k_fold)
        val_set_end = int((i_fold + 1) * len(gam_ids) / k_fold)
        #print(val_set_start, val_set_end)
        val_set = gam_ids[val_set_start:val_set_end]

        # train_set
        train_set = [ x for x in gam_ids if x not in val_set ]

        #print(f'test_set: {test_set}')
        #print(f'val_set: {val_set}')
        #print(f'train_set: {train_set}')

        dataset = glob(f'{root}/*/*.wav')
        dataset_mode = { 'train': train_set, 'val': val_set, 'test': test_set }
        self.meta = list()
        for file in dataset:
            gid = Path(file).stem.split('_')[0]
            gam_id = gid.split('-')[0]

            if gam_id in dataset_mode[mode] and gid in label_json:
                self.meta.append(file)

        self.meta = sorted(self.meta)
        print(f'{mode} dataset: {len(self.meta)}')
        #print(f'gam_part list: {label_json.keys()}')




    def _find_nearest(self, value):
        idx = (np.abs(self.float_labels - value)).argmin()
        #return self.labels[idx]
        return idx


    def _get_labels(self):

        # from 5 to 75, 15 classes in case of l_step 5
        self.labels = list()
        for i in range(5, 80, self.l_step):
            self.labels.append(f'{i:02}')
        #self.labels.append(f'{self.l_step:02}')
        #self.labels.append(f'{self.l_step+10:02}')

        self.labels = sorted(self.labels)
        self.float_labels = np.asarray(sorted([ float(x) for x in self.labels ]))

        #print(f'labels: {self.labels}')
        #print(f'float_labels: {self.float_labels}')

    def __getitem__(self, index):

        fname = self.meta[index]
        field = Path(fname).stem.split('_')[0].split('-')
        gid = int(field[0])
        pid = int(field[1])
        label = self.label_dict[f'{gid:03}-{pid:03}']
        
        #print(fname, self.labels[label])

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
