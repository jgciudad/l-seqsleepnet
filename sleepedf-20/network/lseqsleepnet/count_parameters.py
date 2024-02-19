import os
import numpy as np
import tensorflow as tf

import shutil, sys
# from datetime import datetime
# import h5py
# import hdf5storage

from lseqsleepnet import LSeqSleepNet
from config import Config

# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import cohen_kappa_score

# from datagenerator_wrapper import DataGeneratorWrapper
# import time

# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_test_data", "./code/HUMMUSS/SleepTransformer_mice/shhs/data_preprocessing/kornum_data/file_list/remote/eeg1/test_list.txt", "file containing the list of test EEG data")
tf.app.flags.DEFINE_string("eog_test_data", "", "file containing the list of test EOG data")
tf.app.flags.DEFINE_string("emg_test_data", "", "file containing the list of test EMG data")

tf.app.flags.DEFINE_string("out_dir", "./outputs/l-seqsleepnet/reduced_test_set/iteration1/1_3/testing/kornum/", "Output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "../../checkpoint/", "Checkpoint directory")

tf.app.flags.DEFINE_float("dropout_rnn", 0.75, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("nfilter", 32, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("attention_size", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("nhidden2", 64, "Sequence length (default: 20)")

tf.app.flags.DEFINE_integer("batch_size", 8, "Number of instances per mini-batch (default: 32)")
tf.app.flags.DEFINE_integer("nclass_data", 4, "Number of classes in the data (whether artifacts are discarded or not is controlled in nclass_model)")
tf.app.flags.DEFINE_integer("nclass_model", 3, "Number of classes for sleep stage prediction (i.e. in mice, if artifacts are discarded, then nclass_model=3)")
tf.app.flags.DEFINE_integer("artifacts_label", 3, "Categorical label of the artifact class in the data")
tf.app.flags.DEFINE_integer("ndim", 129, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("frame_seq_len", 17, "Sequence length (default: 20)")

# subsuqence length
tf.app.flags.DEFINE_integer("sub_seq_len", 3, "Sequence length (default: 32)")
# number of subsequence
tf.app.flags.DEFINE_integer("nsubseq", 1, "number of overall segments (default: 9)")

tf.app.flags.DEFINE_string("best_model_criteria", 'balanced_accuracy', "whether to save the model with best 'balanced_accuracy' or 'accuracy' (default: accuracy)")
tf.app.flags.DEFINE_string("loss_type", 'weighted_ce', "whether to use 'weighted_ce' or 'normal_ce' (default: accuracy)")

tf.app.flags.DEFINE_integer("dualrnn_blocks", 1, "Number of dual rnn blocks (default: 1)")

tf.app.flags.DEFINE_float("gpu_usage", 0.5, "Dropout keep probability (default: 0.5)")

FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
print(sys.argv[0])
flags_dict = {}
for idx, a in enumerate(sys.argv):
    if a[:2]=="--":
        flags_dict[a[2:]] = sys.argv[idx+1]

for attr in sorted(flags_dict): # python3
    print("{}={}".format(attr.upper(), flags_dict[attr]))
print("")

# path where some output are stored
out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
# path where checkpoint models are stored
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

with open(os.path.join(out_path,'test_settings.txt'), 'w') as f:
    for attr in sorted(flags_dict):  # python3
        f.write("{}={}".format(attr.upper(), flags_dict[attr]))
        f.write('\n')

config = Config()
config.dropout_rnn = FLAGS.dropout_rnn
config.sub_seq_len = FLAGS.sub_seq_len
config.nfilter = FLAGS.nfilter
config.nhidden1 = FLAGS.nhidden1
config.nhidden2 = FLAGS.nhidden2
config.attention_size = FLAGS.attention_size

config.nclass_data = FLAGS.nclass_data
config.nclass_model = FLAGS.nclass_model
config.artifacts_label = FLAGS.artifacts_label
config.ndim = FLAGS.ndim
config.frame_seq_len = FLAGS.frame_seq_len
config.best_model_criteria = FLAGS.best_model_criteria
config.loss_type = FLAGS.loss_type
config.batch_size = FLAGS.batch_size
config.l2_reg_lambda = config.l2_reg_lambda / FLAGS.batch_size # scaling by btach size because now I'm normalizing the loss by the number of elements in batch

config.nsubseq = FLAGS.nsubseq
config.dualrnn_blocks = FLAGS.dualrnn_blocks

eeg_active = (FLAGS.eeg_test_data != "")
eog_active = (FLAGS.eog_test_data != "")
emg_active = (FLAGS.emg_test_data != "")

if (not eog_active and not emg_active):
    print("eeg active")
    # test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
    #                                          num_fold=config.num_fold_testing_data,
    #                                          data_shape_2=[config.frame_seq_len, config.ndim],
    #                                          seq_len = config.sub_seq_len * config.nsubseq,
    #                                          nclasses = config.nclass_data,
    #                                          shuffle=False)
    # test_gen_wrapper.compute_eeg_normalization_params_by_signal()
    nchannel = 1

elif(eog_active and not emg_active):
    print("eeg and eog active")
    # test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
    #                                         eog_filelist=os.path.abspath(FLAGS.eog_test_data),
    #                                         num_fold=config.num_fold_testing_data,
    #                                         data_shape_2=[config.frame_seq_len, config.ndim],
    #                                         seq_len = config.sub_seq_len * config.nsubseq,
    #                                         nclasses = config.nclass_data,
    #                                         shuffle=False)
    # test_gen_wrapper.compute_eeg_normalization_params_by_signal()
    # test_gen_wrapper.compute_eog_normalization_params_by_signal()
    nchannel = 2
elif(eog_active and emg_active):
    print("eeg, eog, and emg active")
    # test_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
    #                                         eog_filelist=os.path.abspath(FLAGS.eog_test_data),
    #                                         emg_filelist=os.path.abspath(FLAGS.emg_test_data),
    #                                         num_fold=config.num_fold_testing_data,
    #                                         data_shape_2=[config.frame_seq_len, config.ndim],
    #                                         seq_len = config.sub_seq_len * config.nsubseq,
    #                                         nclasses = config.nclass_data,
    #                                         shuffle=False)
    # test_gen_wrapper.compute_eeg_normalization_params_by_signal()
    # test_gen_wrapper.compute_eog_normalization_params_by_signal()
    # test_gen_wrapper.compute_emg_normalization_params_by_signal()
    nchannel = 3

config.nchannel = nchannel

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_usage, allow_growth=False)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        net = LSeqSleepNet(config=config)

        out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        # saver = tf.train.Saver(tf.all_variables())
        # # Load the saved model
        # best_dir = os.path.join(checkpoint_path, "best_model_acc")
        # saver.restore(sess, best_dir)
        # print("Model all loaded")

        variables = list()
        variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        # #restorer = tf.train.Saver(variables)
        # print("RESTORE VARIABLES")
        # #print(variables)
        # for i, v in enumerate(variables):
        #     print(v.name[:-2])

        # vars_in_checkpoint = tf.train.list_variables(best_dir)
        # print("IN-CHECK-POINT VARIABLES")
        # #print(vars_in_checkpoint)
        # vars_in_checkpoint_names = list()
        # for i, v in enumerate(vars_in_checkpoint):
        #     print(v[0])
        #     vars_in_checkpoint_names.append(v[0])

        # var_list_to_retstore = [v for v in variables if v.name[:-2] in vars_in_checkpoint_names]
        # print("ACTUAL RESTORE VARIABLES")
        # print(var_list_to_retstore)


        # restorer = tf.train.Saver(var_list_to_retstore)
        # #restorer = tf.train.Saver(tf.all_variables())
        # # Load pretrained model
        # restorer.restore(sess, best_dir)
        # print("Model loaded")

        n_params = 0
        # for v in var_list_to_retstore:
        for v in variables:

            if 'Adam' not in v.name:
                if 'scale' not in v.name:
                    if 'mean' not in v.name:
                        if 'var' not in v.name:
                            if 'offset' not in v.name:
                                n_params = n_params + tf.size(v).eval()

        print('Number of parameters: ', len(a))
