import audio_lib as al
import inference as eat
import argparse
import os
temp_out = './test.wav'

def test_audio(model_path):


    while True:
        wav, sr = al.record()
        if wav is not None:
            al.save2wav(wav, sr, temp_out)
            print('playing')
            al.play(wav, sr)

            # evaluate
            pred = eat.run(model_path)
            print(f'class: {pred.item()}')

            os.remove(temp_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='model path')
    args = parser.parse_args()
    test_audio(args.model_path)

