import audio_lib as al
import inference as eat
import argparse
import os
from glob import glob
import shutil
temp_out = './test.wav'

def test_folder(model_path, folder):

    print(f'# test from {folder}')
    net, data_set= eat.run(model_path)

    dataset_name = model_path.split('/')[-1]
    dataset_path = f'../../dataset/{dataset_name}'

    labels = glob(f'{dataset_path}/*.wav')
    labels = [ x.split('/')[-1].replace('.wav', '') for x in labels]
    labels = sorted(labels)
    print(labels)

    files = glob(f'{folder}/*.wav')
    files = sorted(files)

    for i, file in enumerate(files):

        shutil.copy(file, temp_out) 
        fname = file.split('/')[-1]

        pred = eat.inference_from_file(net, data_set)
        label = pred.item()

        print(f'[{i+1}/{len(files)}, {fname}] {labels[label]}')
        os.remove(temp_out)

def test_audio(model_path):


    print('# test from mic')
    net, data_set= eat.run(model_path)

    dataset_name = model_path.split('/')[-1]
    dataset_path = f'../../dataset/{dataset_name}'

    labels = glob(f'{dataset_path}/*.wav')
    labels = [ x.split('/')[-1].replace('.wav', '') for x in labels]
    labels = sorted(labels)
    print(labels)


    while True:
        wav, sr = al.record()
        if wav is not None:
            al.save2wav(wav, int(sr/2), temp_out)

            # evaluate
            al.play(wav, int(sr/2))
            print('analyze')

            pred = eat.inference_from_file(net, data_set)
            label = pred.item()

            print(f'label: {labels[label]}')

            os.remove(temp_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='model path')
    parser.add_argument("--folder", help="folder", nargs='?', default=None)
    args = parser.parse_args()

    if args.folder is None:
        test_audio(args.model_path)
    else:
        test_folder(args.model_path, args.folder)


