from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *
from utils.explainability_utils import evaluate_explainability, resolve_explanation_selection

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument(
    '--model_type',
    type=str,
    choices=[
        'bmil-vis', 'bmil-addvis', 'bmil-conjvis', 'bmil-convis',
        'bmil-addenc', 'bmil-conjenc', 'bmil-conenc',
        'bmil-enc', 'bmil-spvis', 'bmil-addspvis', 'bmil-conjspvis', 'bmil-conspvis',
    ],
    default='bmil-vis',
    help='type of model (default: bmil-vis)'
)
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False,
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--run-explainability', action='store_true', default=False,
                    help='Compute interpretability metrics after bag-level evaluation')
parser.add_argument(
    '--explanation-type',
    type=str,
    default='all',
    help='Comma separated list of explanation names (learn, learn-modified, learn-plus, '
    'int-attn-coeff, int-built-in, int-computed, int-clf) or "all".',
)
parser.add_argument('--explainability-model-mode', type=str, default='validation',
                    help='Forward-pass mode passed to the explainability helper (default: validation)')
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
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    if args.task in ['mnist_fourbags', 'mnist_even_odd', 'mnist_adjacent_pairs', 'mnist_fourbags_plus']:
        if args.data_root_dir is None:
            raise ValueError('MNIST evaluation requires --data_root_dir pointing to the dataset root')
        args.splits_dir = os.path.join(args.data_root_dir, 'splits', args.task)
    else:
        args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/c16_all_cases.csv',
                            data_dir= os.path.join(args.data_root_dir, ''),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal':0, 'tumor':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False,
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'mnist_fourbags':
    if args.data_root_dir is None:
        raise ValueError('mnist_fourbags requires --data_root_dir pointing to the generated dataset directory')
    args.n_classes = 4
    csv_path = os.path.join(args.data_root_dir, 'mnist_fourbags.csv')
    dataset = Generic_MIL_Dataset(
        csv_path=csv_path,
        data_dir=os.path.join(args.data_root_dir, ''),
        shuffle=False,
        print_info=True,
        label_dict={'none': 0, 'mostly_eight': 1, 'mostly_nine': 2, 'both': 3},
        patient_strat=False,
        ignore=[],
        label_col='label_name',
    )

elif args.task == 'mnist_even_odd':
    if args.data_root_dir is None:
        raise ValueError('mnist_even_odd requires --data_root_dir pointing to the generated dataset directory')
    args.n_classes = 2
    csv_path = os.path.join(args.data_root_dir, 'mnist_even_odd.csv')
    dataset = Generic_MIL_Dataset(
        csv_path=csv_path,
        data_dir=os.path.join(args.data_root_dir, ''),
        shuffle=False,
        print_info=True,
        label_dict={'odd_majority': 0, 'even_majority': 1},
        patient_strat=False,
        ignore=[],
        label_col='label_name',
    )

elif args.task == 'mnist_adjacent_pairs':
    if args.data_root_dir is None:
        raise ValueError('mnist_adjacent_pairs requires --data_root_dir pointing to the generated dataset directory')
    args.n_classes = 2
    csv_path = os.path.join(args.data_root_dir, 'mnist_adjacent_pairs.csv')
    dataset = Generic_MIL_Dataset(
        csv_path=csv_path,
        data_dir=os.path.join(args.data_root_dir, ''),
        shuffle=False,
        print_info=True,
        label_dict={'no_adjacent_pairs': 0, 'has_adjacent_pairs': 1},
        patient_strat=False,
        ignore=[],
        label_col='label_name',
    )

elif args.task == 'mnist_fourbags_plus':
    if args.data_root_dir is None:
        raise ValueError('mnist_fourbags_plus requires --data_root_dir pointing to the generated dataset directory')
    args.n_classes = 4
    csv_path = os.path.join(args.data_root_dir, 'mnist_fourbags_plus.csv')
    dataset = Generic_MIL_Dataset(
        csv_path=csv_path,
        data_dir=os.path.join(args.data_root_dir, ''),
        shuffle=False,
        print_info=True,
        label_dict={'none': 0, 'three_five': 1, 'one_only': 2, 'one_and_seven': 3},
        patient_strat=False,
        ignore=[],
        label_col='label_name',
    )

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    all_f1 = []
    explainability_rows = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]

        # previous evaluation function
        model, patient_results, test_error, auc, f1, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        all_f1.append(f1)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

        if args.run_explainability:
            try:
                checkpoint_name = os.path.basename(ckpt_paths[ckpt_idx])
                selection = resolve_explanation_selection(args.model_type, args.explanation_type)
                instance_types = sorted(selection.instance)
                attention_types = sorted(selection.attention)
                if not instance_types and not attention_types:
                    requested_str = ", ".join(sorted(selection.requested)) if selection.requested else "(default)"
                    print(
                        "Explainability evaluation skipped for model '{model}' because the requested "
                        "types {types} are not supported by model_type {model_type}.".format(
                            model=checkpoint_name,
                            types=requested_str,
                            model_type=args.model_type,
                        )
                    )
                    continue

                summary_bits = []
                if instance_types:
                    summary_bits.append(f"instance={', '.join(instance_types)}")
                if attention_types:
                    summary_bits.append(f"attention={', '.join(attention_types)}")
                print(
                    "Explainability selection for model '{model}' ({model_type}): {summary}.".format(
                        model=checkpoint_name,
                        model_type=args.model_type,
                        summary="; ".join(summary_bits) if summary_bits else "none",
                    )
                )
                if selection.ignored:
                    print(
                        "Ignored unsupported explanation names for {model_type}: {names}.".format(
                            model_type=args.model_type,
                            names=", ".join(sorted(selection.ignored)),
                        )
                    )
                explainability_results = evaluate_explainability(
                    model,
                    split_dataset,
                    model_type=args.model_type,
                    explanation_type=args.explanation_type,
                    model_mode=args.explainability_model_mode,
                    model_identifier=checkpoint_name,
                )
                if explainability_results:
                    fold_records = []
                    for metrics in explainability_results:
                        record = metrics.to_dict()
                        record.update({'fold': folds[ckpt_idx]})
                        fold_records.append(record)
                        explainability_rows.append(record)
                        print(
                            "Explainability metrics for model '{model}' using '{expl}' explanation: "
                            "family={family}, macro_f1={macro}, bal_acc={bal}, ndcg={ndcg}, auprc2={auprc}".format(
                                model=record.get("model_identifier", "unknown"),
                                expl=record.get("explanation_type", "unknown"),
                                family=record.get("metric_family", "unknown"),
                                macro=record.get("instance_macro_f1"),
                                bal=record.get("instance_balanced_accuracy"),
                                ndcg=record.get("attention_ndcg"),
                                auprc=record.get("attention_auprc2"),
                            )
                        )
                    pd.DataFrame(fold_records).replace({None: np.nan}).to_csv(
                        os.path.join(args.save_dir, 'fold_{}_explainability.csv'.format(folds[ckpt_idx])),
                        index=False,
                    )
                else:
                    print(
                        f"Explainability evaluation produced no metrics for fold {folds[ckpt_idx]} "
                        "(no supported explanation names requested)."
                    )
            except (OSError, KeyError, ValueError, RuntimeError) as err:
                print(f"Explainability evaluation skipped for fold {folds[ckpt_idx]}: {err}")

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc, 'test_f1': all_f1})
    if explainability_rows:
        explainability_df = pd.DataFrame(explainability_rows).replace({None: np.nan})
        explainability_df.to_csv(os.path.join(args.save_dir, 'explainability_summary.csv'), index=False)
        pivot_df = (
            explainability_df
            .set_index(['fold', 'explanation_type'])
            .sort_index()
        )
        wide_df = pivot_df.unstack('explanation_type')
        wide_df.columns = [f"{col}_{name}" for col, name in wide_df.columns]
        wide_df = wide_df.reset_index().rename(columns={'fold': 'folds'})
        final_df = final_df.merge(wide_df, on='folds', how='left')
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
