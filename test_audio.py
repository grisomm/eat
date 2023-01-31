import audio_lib as al
import inference as eat
import argparse
import os
from glob import glob
import shutil
temp_out = './test.wav'

def test_folder(args, folder):

    print(f'# test from {folder}')
    args.tools = None
    args.labels = None
    net, data_set= eat.run(args)

    model_path = args.f_res

    dataset_name = model_path.split('/')[-1]
    dataset_path = f'../../dataset/{dataset_name}'

    labels = glob(f'{dataset_path}/train/*/')
    labels = [ x.split('/')[-2] for x in labels]
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

    labels = glob(f'{dataset_path}/train/*/')
    labels = [ x.split('/')[-2] for x in labels]
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
    parser.add_argument('f_res', help='model path')
    parser.add_argument("--folder", help="folder", nargs='?', default=None)
    args = parser.parse_args()

    if args.folder is None:
        test_audio(args.f_res)
    else:
        test_folder(args, args.folder)


