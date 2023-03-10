import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import yaml
from utils.helper_funcs import accuracy, count_parameters, mAP, measure_inference_time
import numpy as np
import torch.nn.functional as F
import librosa
#import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_res", default=None, type=Path)

    parser.add_argument('--tools', nargs='+', type=str, default=None)
    parser.add_argument('--labels', nargs='+', type=str, default=None)

    # for gam
    parser.add_argument('--k_fold', default=None, type=int)
    parser.add_argument('--i_fold', default=None, type=int)
    parser.add_argument('--t_ratio', default=None, type=float)
    parser.add_argument('--r_seed', default=None, type=int)
    parser.add_argument('--l_step', default=None, type=int)
    parser.add_argument('--l_start', default=5, type=int)
    parser.add_argument('--dif', default=10000, type=int)
    parser.add_argument('--gam_id_range', nargs='+', type=int, default=[0,100000])
    parser.add_argument('--value_range', nargs='+', type=int, default=[0,100000])

    args = parser.parse_args()
    return args


def run(args, from_file=True):
    f_res = Path(args.f_res)
    tools = args.tools
    labels = args.labels

    # for gam
    k_fold = args.k_fold
    i_fold = args.i_fold
    t_ratio = args.t_ratio
    r_seed = args.r_seed
    l_step = args.l_step
    l_start = args.l_start
    dif = args.dif
    gam_id_range = args.gam_id_range
    value_range = args.value_range

    #args = parse_args()
    #f_res = args.f_res
    #add_noise = args.add_noise
    #with (args.f_res / Path("args.yml")).open() as f:
    with (f_res / Path("args.yml")).open() as f:
        args = yaml.load(f, Loader=yaml.Loader)
    try:
        args = vars(args)
    except:
        if 'net' in args.keys():
            del args['net']
        args_orig = args
        args = {}
        for k, v in args_orig.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    args[kk] = vv
            else:
                args[k] = v
    args['f_res'] = f_res
    args['tools'] = tools
    args['labels'] = labels 

    
    # for gam
    args['k_fold'] = k_fold
    args['i_fold'] = i_fold
    args['t_ratio'] = t_ratio
    args['r_seed'] = r_seed
    args['l_step'] = l_step
    args['l_start'] = l_start
    args['dif'] = dif 
    args['gam_id_range'] = gam_id_range
    args['value_range'] = value_range

    #args['add_noise'] = add_noise
    args['add_noise'] = None 
    with open(args['f_res'] / "args.yml", "w") as f:
        yaml.dump(args, f)
    #print(args)
    #######################
    # Load PyTorch Models #
    #######################
    from modules.soundnet import SoundNetRaw as SoundNet
    ds_fac = np.prod(np.array(args['ds_factors'])) * 4
    net = SoundNet(nf=args['nf'],
                   dim_feedforward=args['dim_feedforward'],
                   clip_length=args['seq_len'] // ds_fac,
                   embed_dim=args['emb_dim'],
                   n_layers=args['n_layers'],
                   nhead=args['n_head'],
                   n_classes=args['n_classes'],
                   factors=args['ds_factors'],
                   )

    #print('***********************************************')
    #print("#params: {}M".format(count_parameters(net)/1e6))
    if torch.cuda.is_available() and device == torch.device("cuda"):
        t_b1 = measure_inference_time(net, torch.randn(1, 1, args['seq_len']))[0]
        print('inference time batch=1: {:.2f}[ms]'.format(t_b1))
        # t_b32 = measure_inference_time(net, torch.randn(32, 1, args['seq_len']))[0]
        # print('inference time batch=32: {:.2f}[ms]'.format(t_b32))
        print('***********************************************')

    if (f_res / Path("chkpnt.pt")).is_file():
        chkpnt = torch.load(f_res / "chkpnt.pt", map_location=torch.device(device))
        model = chkpnt['model_dict']
    else:
        raise ValueError

    if 'use_dp' in args.keys() and args['use_dp']:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in model.items():
            name = k.replace('module.', '')
            state_dict[name] = v
        net.load_state_dict(state_dict, strict=True)
    else:
        net.load_state_dict(model, strict=True)
    net.to(device)
    if torch.cuda.device_count() > 1:
        from utils.helper_funcs import parse_gpu_ids
        args['gpu_ids'] = [i for i in range(torch.cuda.device_count())]
        net = torch.nn.DataParallel(net, device_ids=args['gpu_ids'])
        net.to('cuda:0')
    net.eval()
    #######################
    # Create data loaders #
    #######################
    if args['dataset'] == 'esc50':
        from datasets.esc_dataset import ESCDataset as SoundDataset
        data_set = SoundDataset(args['data_path'],
                                mode='test',
                                segment_length=args['seq_len'],
                                sampling_rate=args['sampling_rate'],
                                fold_id=args['fold_id'],
                                transforms=None)

    elif args['dataset'] == 'speechcommands':
        from datasets.speechcommand_dataset import SpeechCommandsDataset as SoundDataset
        data_set = SoundDataset(args['data_path'],
                                mode='test',
                                segment_length=args['seq_len'],
                                sampling_rate=args['sampling_rate'],
                                transforms=None)

    elif args['dataset'] == 'urban8k':
        from datasets.urban8K_dataset import Urban8KDataset as SoundDataset
        data_set = SoundDataset(args['data_path'],
                                mode='test',
                                segment_length=args['seq_len'],
                                sampling_rate=args['sampling_rate'],
                                transforms=None,
                                fold_id=args['fold_id'])

    elif args['dataset'] == 'audioset':
        from datasets.audioset_dataset import AudioSetDataset as SoundDataset
        data_set = SoundDataset(
            args['data_path'],
            'test',
            data_subtype=None,
            segment_length=args['seq_len'],
            sampling_rate=args['sampling_rate'],
            transforms=None
        )

    elif args['dataset'] == 'finger':
        if from_file:
            from datasets.finger_dataset import FingerDataset as SoundDataset
            data_set = SoundDataset(
                args['data_path'],
                'test_file',
                tools = args['tools'],
                label_filters = args['labels'],
                segment_length=args['seq_len'],
                sampling_rate=args['sampling_rate'],
                transforms=None,
                fold_id=args['fold_id'])
        else:
            from datasets.finger_dataset import FingerDataset as SoundDataset
            data_set = SoundDataset(
                args['data_path'],
                'test',
                tools = args['tools'],
                label_filters = args['labels'],
                segment_length=args['seq_len'],
                sampling_rate=args['sampling_rate'],
                transforms=None,
                fold_id=args['fold_id'])
    elif args['dataset'] == 'gam':
        from datasets.gam_dataset import GamDataset as SoundDataset
        data_set = SoundDataset(
            args['data_path'],
            'test',
            #tools = args['tools'],
            #label_filters = args['labels'],
            k_fold = args['k_fold'],
            i_fold = args['i_fold'],
            r_seed = args['r_seed'],
            t_ratio = args['t_ratio'],
            l_step = args['l_step'],
            l_start = args['l_start'],
            dif = args['dif'],
            gam_id_range = args['gam_id_range'],
            value_range = args['value_range'],
            segment_length=args['seq_len'],
            sampling_rate=args['sampling_rate'],
            transforms=None,
            fold_id=args['fold_id'])
    else:
        raise ValueError

    if args['dataset'] == 'finger':
        if from_file:
            #return inference_from_file(net=net, data_set=data_set)
            return net, data_set
        else:
            return inference_single_label(net=net, data_set=data_set, args=args)

    elif args['dataset'] == 'gam':
        return inference_single_label(net=net, data_set=data_set, args=args)

    elif args['dataset'] != 'audioset':
        inference_single_label(net=net, data_set=data_set, args=args)
    elif args['dataset'] == 'audioset':
        inference_multi_label(net=net, data_set=data_set, args=args)
    else:
        raise ValueError("check args dataset")

def inference_from_file(net, data_set):

    data_loader = DataLoader(data_set,
                             batch_size=128,
                             num_workers=8,
                             pin_memory=True if torch.cuda.is_available() else False,
                             shuffle=False)

    for i, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device)
            pred = net(x)
            _, y_est = torch.max(pred, 1)
            
            #print(x.shape, pred, y_est)
            return y_est[0]

def inference_single_label(net, data_set, args):
    data_loader = DataLoader(data_set,
                             batch_size=128,
                             num_workers=8,
                             pin_memory=True if torch.cuda.is_available() else False,
                             shuffle=False)

    labels = torch.zeros(len(data_loader.dataset), dtype=torch.float32).float()
    preds = torch.zeros(len(data_loader.dataset), args['n_classes'], dtype=torch.float32).float()
    # confusion_matrix = torch.zeros(args['n_classes'], args['n_classes'], dtype=torch.int)
    confusion_matrix = torch.zeros(args['n_classes'], args['n_classes'], dtype=torch.int)
    idx_start = 0
    for i, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            pred = net(x)
            _, y_est = torch.max(pred, 1)
            idx_end = idx_start + y.shape[0]
            preds[idx_start:idx_end, :] = pred
            labels[idx_start:idx_end] = y
            for t, p in zip(y.view(-1), y_est.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            print("{}/{}".format(i, len(data_loader)))
        idx_start = idx_end
    acc_av = accuracy(preds.detach(), labels.detach(), [1, ])[0]

    res = {
        "acc": acc_av,
        "preds": preds,
        "labels": labels.view(-1),
        "confusion_matrix": confusion_matrix
    }
    torch.save(res, args['f_res'] / "res.pt")

    #print("acc:{}".format(np.round(acc_av*100)/100))
    acc = np.round(acc_av*100)/100
    print("acc:{}".format(acc))
    print("cm:{}".format(confusion_matrix.diag().sum() / len(data_loader.dataset)))
    print("test num:{}".format(len(data_loader.dataset)))
    print('***************************************')
    bad_labels = []
    for i, c in enumerate(confusion_matrix):
        i_est = c.argmax(-1)
        if i != i_est:
            print('{} {} {}-->{}'.format(i, i_est.item(), data_set.labels[i], data_set.labels[i_est]))
            bad_labels.append([i, i_est])
    print(bad_labels)

    return acc.item()


def inference_multi_label(net, data_set, args):
    from utils.helper_funcs import collate_fn
    data_loader = DataLoader(data_set,
                             batch_size=128,
                             num_workers=8,
                             pin_memory=True if torch.cuda.is_available() else False,
                             shuffle=False,
                             collate_fn=collate_fn)

    labels = torch.zeros(len(data_loader.dataset), args['n_classes'], dtype=torch.float32).float()
    preds = torch.zeros(len(data_loader.dataset), args['n_classes'], dtype=torch.float32).float()
    confusion_matrix = torch.zeros(args['n_classes'], args['n_classes'], dtype=torch.int)
    idx_start = 0
    for i, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to('cuda:0')
            y = [F.one_hot(torch.Tensor(y_i).long(), args['n_classes']).sum(dim=0).float() for y_i in y]
            y = torch.stack(y, dim=0).contiguous().to('cuda:0')
            pred = net(x)
            idx_end = idx_start + y.shape[0]
            preds[idx_start:idx_end, :] = torch.sigmoid(pred)
            labels[idx_start:idx_end, :] = y
            print("{}/{}".format(i, len(data_loader)))
        idx_start = idx_end
    mAP_av = mAP(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
    res = {
        "mAP": mAP_av,
        "preds": preds,
        "labels": labels.view(-1),
        "confusion_matrix": confusion_matrix
    }
    torch.save(res, args['f_res'] / "res.pt")
    # torch.save(net.state_dict(), "net.pt")
    print("mAP:{}".format(np.round(mAP_av*100)/100))


if __name__ == '__main__':
    args = parse_args()

    if args.tools is not None: 

        accs = dict()
        accs['total'] = run(args, False)

        for tool in args.tools:
            args.tools = [ tool ] 
            accs[tool] = run(args, False)

        print('###### results by tool #########')
        for key, value in accs.items():
            print(key, value)
        print(accs)

    elif args.labels is not None:

        accs = dict()
        accs['total'] = run(args, False)

        for label in args.labels:
            args.labels = [ label ] 
            accs[label] = run(args, False)

        print('###### results by label #########')
        for key, value in accs.items():
            print(key, value)
        print(accs)

    else:
        run(args, False)
