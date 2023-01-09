import torch
import torchaudio
import torch.nn.functional as F
import os
import random
from datasets.audio_augs import AudioAugs
import pandas as pd
from glob import glob


class FingerDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 mode,
                 segment_length,
                 sampling_rate,
                 transforms=None,
                 fold_id=None,
                 ):
        self.sampling_rate = sampling_rate
        self.root = root
        self.segment_length = segment_length
        self.transforms = transforms
        self._get_labels()

        if mode == 'train':
            self.meta = glob(f'{root}/train/*/*.wav')
        else:
            self.meta = glob(f'{root}/validation/*/*.wav')

        self.meta = sorted(self.meta)

    def _get_labels(self):
        self.labels = [
        'one_finger',
        'two_finger']

    def __getitem__(self, index):
        meta_i = self.meta.iloc[index]
        fname = os.path.join(self.root, 'audio22_5', 'fold'+str(meta_i['fold']), meta_i['slice_file_name'])
        label = meta_i['classID']

        fname = self.meta[index]
        label_name = fname.split('/')[-2]
        label = self.labels.index(label_name)
        audio, sampling_rate = torchaudio.load(fname)
        audio.squeeze_()
        audio = 0.95 * (audio / audio.__abs__().max()).float()

        print(fname, sampling_rate, audio.shape[0], self.segment_length)

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
            audio = AudioAugs(self.transforms, sampling_rate, p=0.5)(audio)

        return audio.unsqueeze(0), label

    def __len__(self):
        return len(self.meta)


if __name__ == "__main__":
    pass
