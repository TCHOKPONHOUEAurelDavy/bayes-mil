from __future__ import print_function

import argparse
import pdb
import os
import math

import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from torch import autograd
# autograd.set_detect_anomaly(True)

def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_ece_loss = []
    all_val_ece_loss = []
    all_test_f1 = []
    all_val_f1 = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        print('------------------use h5? {}--------------------'.format(train_dataset.use_h5))

        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc, test_ece_loss, val_ece_loss, test_f1, val_f1  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_test_ece_loss.append(test_ece_loss)
        all_val_ece_loss.append(val_ece_loss)
        all_test_f1.append(test_f1)
        all_val_f1.append(val_f1)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
                             'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc': all_val_acc,
                             'test_ece_loss': all_test_ece_loss,
                             'val_ece_loss': all_val_ece_loss,
                             'test_f1': all_test_f1,
                             'val_f1': all_val_f1})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=2e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'msa', 'pmil-V',
                                                        'pmil-C', 'pmil-N',
                                                        'bmil-A', 'bmil-F', 'bmil-vis', 'bmil-addvis', 'bmil-conjvis', 'bmil-convis', 'bmil-spvis',
                                                        'bmil-enc', 'mil_baens', 'hmil', 'smil-D'],
                    default='clam_sb', help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument(
    '--task',
    type=str,
    choices=[
        'task_1_tumor_vs_normal',
        'task_2_tumor_subtyping',
        'mnist_fourbags',
        'mnist_even_odd',
        'mnist_adjacent_pairs',
        'mnist_fourbags_plus',
    ],
)
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}


print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/c16_all_cases.csv',
                                  data_dir=os.path.join(args.data_root_dir, ''),
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_dict={'normal': 0, 'tumor': 1},
                                  patient_strat=False,
                                  ignore=[])
    if any(flag in args.model_type for flag in ('convis', 'conjvis', 'spvis')):
        dataset.load_from_h5(True)

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=4
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/train_multi.csv',
                            data_dir= os.path.join(args.data_root_dir, ''),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'negative':0, 'micro':1, 'macro':2, 'itc':3},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping

elif args.task == 'mnist_fourbags':
    if args.data_root_dir is None:
        raise ValueError('mnist_fourbags requires --data_root_dir pointing to the generated dataset directory')
    args.n_classes = 4
    csv_path = os.path.join(args.data_root_dir, 'mnist_fourbags.csv')
    dataset = Generic_MIL_Dataset(
        csv_path=csv_path,
        data_dir=os.path.join(args.data_root_dir, ''),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={'none': 0, 'mostly_eight': 1, 'mostly_nine': 2, 'both': 3},
        patient_strat=False,
        ignore=[],
        label_col='label_name',
    )
    if any(flag in args.model_type for flag in ('convis', 'conjvis', 'spvis')):
        dataset.load_from_h5(True)

elif args.task == 'mnist_even_odd':
    if args.data_root_dir is None:
        raise ValueError('mnist_even_odd requires --data_root_dir pointing to the generated dataset directory')
    args.n_classes = 2
    csv_path = os.path.join(args.data_root_dir, 'mnist_even_odd.csv')
    dataset = Generic_MIL_Dataset(
        csv_path=csv_path,
        data_dir=os.path.join(args.data_root_dir, ''),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={'odd_majority': 0, 'even_majority': 1},
        patient_strat=False,
        ignore=[],
        label_col='label_name',
    )
    if any(flag in args.model_type for flag in ('convis', 'conjvis', 'spvis')):
        dataset.load_from_h5(True)

elif args.task == 'mnist_adjacent_pairs':
    if args.data_root_dir is None:
        raise ValueError('mnist_adjacent_pairs requires --data_root_dir pointing to the generated dataset directory')
    args.n_classes = 2
    csv_path = os.path.join(args.data_root_dir, 'mnist_adjacent_pairs.csv')
    dataset = Generic_MIL_Dataset(
        csv_path=csv_path,
        data_dir=os.path.join(args.data_root_dir, ''),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={'no_adjacent_pairs': 0, 'has_adjacent_pairs': 1},
        patient_strat=False,
        ignore=[],
        label_col='label_name',
    )
    if any(flag in args.model_type for flag in ('convis', 'conjvis', 'spvis')):
        dataset.load_from_h5(True)

elif args.task == 'mnist_fourbags_plus':
    if args.data_root_dir is None:
        raise ValueError('mnist_fourbags_plus requires --data_root_dir pointing to the generated dataset directory')
    args.n_classes = 4
    csv_path = os.path.join(args.data_root_dir, 'mnist_fourbags_plus.csv')
    dataset = Generic_MIL_Dataset(
        csv_path=csv_path,
        data_dir=os.path.join(args.data_root_dir, ''),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={'none': 0, 'three_five': 1, 'one_only': 2, 'one_and_seven': 3},
        patient_strat=False,
        ignore=[],
        label_col='label_name',
    )
    if any(flag in args.model_type for flag in ('convis', 'conjvis', 'spvis')):
        dataset.load_from_h5(True)

else:
    raise NotImplementedError

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    if args.task in ['mnist_fourbags', 'mnist_even_odd', 'mnist_adjacent_pairs', 'mnist_fourbags_plus']:
        args.split_dir = os.path.join(args.data_root_dir, 'splits', args.task)
    else:
        args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    if os.path.isabs(args.split_dir):
        args.split_dir = args.split_dir
    elif args.task in ['mnist_fourbags', 'mnist_even_odd', 'mnist_adjacent_pairs', 'mnist_fourbags_plus']:
        args.split_dir = os.path.join(args.data_root_dir, args.split_dir)
    else:
        args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")
