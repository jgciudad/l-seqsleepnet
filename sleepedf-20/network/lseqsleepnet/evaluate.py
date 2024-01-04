import os, gc, json, argparse, pickle, time, shutil, select, sys
import numpy as np
from os.path import join as pj

from scipy.io import loadmat, savemat
from sklearn import metrics
from imblearn.metrics import specificity_score
from imblearn.metrics import sensitivity_score

# import h5py
import hdf5storage

# affective sequence length
config = dict()

config['nsubseq'] = 10
config['subseqlen'] = 10
config['seq_len'] = config['subseqlen']*config['nsubseq']
config['num_fold_testing_data'] = 2
config['aggregation'] = 'multiplication' # 'multiplication' or 'average'
config['filter_out_artifacts'] = 1
config['nclass_model'] = 3
config['out_dir'] = '/Users/tlj258/Documents/HUMMUSS_paper/outputs/l-seqsleepnet'

n_iterations = 3
datasets_list = ["kornum", "spindle"]
cohorts_list = ["a", "d"]
n_scorers_spindle = 2
nsubseq_list = [1, 2, 3, 4, 6, 8, 10, 16, 22]



def read_groundtruth(filelist):
    labels = dict()
    file_sizes = []
    with open(filelist) as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        print(l)
        items = l.split()
        file_sizes.append(int(items[1]))
        #data = h5py.File(items[0], 'r')
        data = hdf5storage.loadmat(file_name=items[0])
        label = np.array(data['label'])  # labels
        label = np.transpose(label, (1, 0))  # rearrange dimension
        label = np.squeeze(label)
        labels[i] = label
    return labels, file_sizes

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

#score = np.zeros([config['seq_len'], len(gen.data_index), config.nclass])
def aggregate_avg(score):
    fused_score = None
    for i in range(config['seq_len']):
        prob_i = np.concatenate((np.zeros((config['seq_len'] - 1, config['nclass_model'])), softmax(np.squeeze(score[i, :, :]))), axis=0)
        prob_i = np.roll(prob_i, -(config['seq_len'] - i - 1), axis=0)
        if fused_score is None:
            fused_score = prob_i
        else:
            fused_score += prob_i
    label = np.argmax(fused_score, axis=-1) + 1 # +1 as label stored in mat files start from 1
    return label

#score = np.zeros([config['seq_len'], len(gen.data_index), config.nclass])
def aggregate_mul(score):
    fused_score = None
    for i in range(config['seq_len']):
        prob_i = np.log10(softmax(np.squeeze(score[i, :, :])))
        prob_i = np.concatenate((np.ones((config['seq_len'] - 1, config['nclass_model'])), prob_i), axis=0)
        prob_i = np.roll(prob_i, -(config['seq_len'] - i - 1), axis=0)
        if fused_score is None:
            fused_score = prob_i
        else:
            fused_score += prob_i
    label = np.argmax(fused_score, axis=-1) + 1 # +1 as label stored in mat files start from 1
    return label

def aggregate_lseqsleepnet(output_file, file_sizes):
    outputs = hdf5storage.loadmat(file_name=output_file)
    outputs = outputs['score']
    # score = [len(gen.data_index), config.epoch_seq_len, config.nclass] -> need transpose
    print(outputs.shape)
    outputs = np.transpose(outputs, (1, 0, 2))
    preds = dict()
    sum_size = 0
    for i, N in enumerate(file_sizes):
        score = outputs[:, sum_size:(sum_size + N - (config['seq_len'] - 1))]
        sum_size += N - (config['seq_len'] - 1)
        preds[i] = aggregate_avg(score) if config['aggregation'] == "average" else aggregate_mul(score)
    return preds


def calculate_metrics(labels, preds):
    ret = dict()

    if config['nclass_model'] == 3:
        preds = preds[labels != 4]
        labels = labels[labels != 4]

    #labels = np.hstack(list(labels.values()))
    #preds = np.hstack(list(preds.values()))
    ret['acc'] = metrics.accuracy_score(y_true=labels, y_pred=preds)
    ret['bal_acc'] = metrics.balanced_accuracy_score(y_true=labels, y_pred=preds)
    ret['F1'] = metrics.f1_score(y_true=labels, y_pred=preds, labels=np.arange(1, config['nclass_model']+1), average=None)
    ret['mean-F1'] = np.mean(ret['F1'])
    ret['kappa'] = metrics.cohen_kappa_score(y1=labels, y2=preds, labels=np.arange(1, config['nclass_model']+1))
    ret['sensitivity'] = sensitivity_score(y_true=labels, y_pred=preds, average=None)
    ret['precision'] = metrics.precision_score(y_true=labels, y_pred=preds, average=None)
    C = metrics.confusion_matrix(y_true=labels, y_pred=preds)


    return ret

results = dict()

lines = dict()
lines['spindle'] = dict()

lines['kornum'] = {'acc': np.zeros((n_iterations, len(nsubseq_list))),
                'bal_acc': np.zeros((n_iterations, len(nsubseq_list))),
                'kappa': np.zeros((n_iterations, len(nsubseq_list))),
                'sensitivity': np.zeros((n_iterations, 3, len(nsubseq_list))),
                'precision': np.zeros((n_iterations, 3, len(nsubseq_list))), 
                'fscore': np.zeros((n_iterations, 3, len(nsubseq_list))),
                'avg_acc': np.zeros((1, len(nsubseq_list))),
                'avg_bal_acc': np.zeros((1, len(nsubseq_list))),
                'avg_kappa': np.zeros((1, len(nsubseq_list))),
                'avg_sensitivity': np.zeros((3, len(nsubseq_list))),
                'avg_precision': np.zeros((3, len(nsubseq_list))), 
                'avg_fscore': np.zeros((3, len(nsubseq_list)))
          }

lines['spindle']['cohort_A'] = {'acc': np.zeros((n_iterations, n_scorers_spindle, len(nsubseq_list))),
                'bal_acc': np.zeros((n_iterations, n_scorers_spindle, len(nsubseq_list))),
                'kappa': np.zeros((n_iterations, n_scorers_spindle, len(nsubseq_list))),
                'sensitivity': np.zeros((n_iterations, n_scorers_spindle, 3, len(nsubseq_list))),
                'precision': np.zeros((n_iterations, n_scorers_spindle, 3, len(nsubseq_list))), 
                'fscore': np.zeros((n_iterations, n_scorers_spindle, 3, len(nsubseq_list))),
                'avg_acc': np.zeros((1, len(nsubseq_list))),
                'avg_bal_acc': np.zeros((1, len(nsubseq_list))),
                'avg_kappa': np.zeros((1, len(nsubseq_list))),
                'avg_sensitivity': np.zeros((3, len(nsubseq_list))),
                'avg_precision': np.zeros((3, len(nsubseq_list))), 
                'avg_fscore': np.zeros((3, len(nsubseq_list)))
          }

lines['spindle']['cohort_D'] = {'acc': np.zeros((n_iterations, n_scorers_spindle, len(nsubseq_list))),
                'bal_acc': np.zeros((n_iterations, n_scorers_spindle, len(nsubseq_list))),
                'kappa': np.zeros((n_iterations, n_scorers_spindle, len(nsubseq_list))),
                'sensitivity': np.zeros((n_iterations, n_scorers_spindle, 3, len(nsubseq_list))),
                'precision': np.zeros((n_iterations, n_scorers_spindle, 3, len(nsubseq_list))), 
                'fscore': np.zeros((n_iterations, n_scorers_spindle, 3, len(nsubseq_list))),
                'avg_acc': np.zeros((1, len(nsubseq_list))),
                'avg_bal_acc': np.zeros((1, len(nsubseq_list))),
                'avg_kappa': np.zeros((1, len(nsubseq_list))),
                'avg_sensitivity': np.zeros((3, len(nsubseq_list))),
                'avg_precision': np.zeros((3, len(nsubseq_list))), 
                'avg_fscore': np.zeros((3, len(nsubseq_list)))
          }


for nsubseq_idx, nsubseq in enumerate(nsubseq_list):

    config['nsubseq'] = nsubseq
    config['seq_len'] = config['subseqlen']*config['nsubseq']

    for dataset in datasets_list:
        if dataset == 'kornum':
                
            data_list_file = "/Users/tlj258/Documents/Code/HUMMUSS/SleepTransformer_mice/shhs/data_preprocessing/kornum_data/file_list/local/eeg1/test_list.txt"
            label_list = []
            labels, file_sizes = read_groundtruth(data_list_file)
            label_list.extend(list(labels.values()))

            for it in range(n_iterations):
                pred_list = []
                
                output_file = pj(config['out_dir'], 'iteration' + str(it+1), str(nsubseq)+'_'+str(config['subseqlen']), 'testing', dataset, 'test_ret.mat')

                preds = aggregate_lseqsleepnet(output_file, file_sizes)
                pred_list.extend(list(preds.values()))
                results = calculate_metrics(np.hstack(label_list), np.hstack(pred_list))

                lines['kornum']['acc'][it, nsubseq_idx] = results['acc']
                lines['kornum']['bal_acc'][it, nsubseq_idx] = results['bal_acc']
                lines['kornum']['kappa'][it, nsubseq_idx] = results['kappa']
                lines['kornum']['sensitivity'][it, :, nsubseq_idx] = results['sensitivity']
                lines['kornum']['precision'][it, :, nsubseq_idx] = results['precision']
                lines['kornum']['fscore'][it, :, nsubseq_idx] = results['F1']

            lines['kornum']['avg_acc'][0, nsubseq_idx] = np.mean(lines['kornum']['acc'][:, nsubseq_idx], axis=0)
            lines['kornum']['avg_bal_acc'][0, nsubseq_idx] = np.mean(lines['kornum']['bal_acc'][:, nsubseq_idx], axis=0)
            lines['kornum']['avg_kappa'][0, nsubseq_idx] = np.mean(lines['kornum']['kappa'][:, nsubseq_idx], axis=0)
            lines['kornum']['avg_sensitivity'][:, nsubseq_idx] = np.mean(lines['kornum']['sensitivity'][:,:,nsubseq_idx], axis=0)
            lines['kornum']['avg_precision'][:, nsubseq_idx] = np.mean(lines['kornum']['precision'][:,:,nsubseq_idx], axis=0)
            lines['kornum']['avg_fscore'][:, nsubseq_idx] = np.mean(lines['kornum']['fscore'][:,:,nsubseq_idx], axis=0)


        if dataset == 'spindle':
            for cohort in cohorts_list:
                for scorer in range(n_scorers_spindle):
                        
                    data_list_file = pj('/Users/tlj258/Documents/Code/HUMMUSS/SleepTransformer_mice/shhs/data_preprocessing/spindle_data/file_list/local', 'cohort_' + cohort.upper(), 'scorer_' + str(scorer+1), 'eeg1/test_list.txt')
                    label_list = []
                    labels, file_sizes = read_groundtruth(data_list_file)
                    label_list.extend(list(labels.values()))

                    for it in range(n_iterations):
                        pred_list = []
                        
                        output_file = pj(config['out_dir'], 'iteration' + str(it+1), str(nsubseq)+'_'+str(config['subseqlen']), 'testing', dataset,  'cohort_' + cohort.upper(),'test_ret.mat')

                        preds = aggregate_lseqsleepnet(output_file, file_sizes)
                        pred_list.extend(list(preds.values()))
                        results = calculate_metrics(np.hstack(label_list), np.hstack(pred_list))

                        lines['spindle']['cohort_' + cohort.upper()]['acc'][it, scorer, nsubseq_idx] = results['acc']
                        lines['spindle']['cohort_' + cohort.upper()]['bal_acc'][it, scorer, nsubseq_idx] = results['bal_acc']
                        lines['spindle']['cohort_' + cohort.upper()]['kappa'][it, scorer, nsubseq_idx] = results['kappa']
                        lines['spindle']['cohort_' + cohort.upper()]['sensitivity'][it, scorer, :, nsubseq_idx] = results['sensitivity']
                        lines['spindle']['cohort_' + cohort.upper()]['precision'][it, scorer, :, nsubseq_idx] = results['precision']
                        lines['spindle']['cohort_' + cohort.upper()]['fscore'][it, scorer, :, nsubseq_idx] = results['F1']

                lines['spindle']['cohort_' + cohort.upper()]['avg_acc'][0, nsubseq_idx] = np.mean(lines['spindle']['cohort_' + cohort.upper()]['acc'][:, :, nsubseq_idx])
                lines['spindle']['cohort_' + cohort.upper()]['avg_bal_acc'][0, nsubseq_idx] = np.mean(lines['spindle']['cohort_' + cohort.upper()]['bal_acc'][:, :, nsubseq_idx])
                lines['spindle']['cohort_' + cohort.upper()]['avg_kappa'][0, nsubseq_idx] = np.mean(lines['spindle']['cohort_' + cohort.upper()]['kappa'][:, :, nsubseq_idx])
                lines['spindle']['cohort_' + cohort.upper()]['avg_sensitivity'][:, nsubseq_idx] = np.mean(lines['spindle']['cohort_' + cohort.upper()]['sensitivity'][:,:,:,nsubseq_idx], axis=(0, 1))
                lines['spindle']['cohort_' + cohort.upper()]['avg_precision'][:, nsubseq_idx] = np.mean(lines['spindle']['cohort_' + cohort.upper()]['precision'][:,:,:,nsubseq_idx], axis=(0, 1))
                lines['spindle']['cohort_' + cohort.upper()]['avg_fscore'][:, nsubseq_idx] = np.mean(lines['spindle']['cohort_' + cohort.upper()]['fscore'][:,:,:,nsubseq_idx], axis=(0, 1))

savemat('metric_lines.mat', lines)

# lines.
            

        # elif dataset == 'spindle':
        #     for c in cohorts_list:


# print results
# for repeat in range(config.num_repeat):
#     print(results['Repeat' + str(repeat+1)])