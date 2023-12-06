import os
import numpy as np
import tensorflow as tf

import shutil, sys
from datetime import datetime
import h5py

from lseqsleepnet import LSeqSleepNet
from config import Config

from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_wrapper import DataGeneratorWrapper

import time

# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_train_data", "./code/HUMMUSS/SleepTransformer_mice/shhs/data_preprocessing/kornum_data/file_list/remote/eeg1/train_list.txt", "file containing the list of training EEG data")
tf.app.flags.DEFINE_string("eeg_eval_data", "./code/HUMMUSS/SleepTransformer_mice/shhs/data_preprocessing/kornum_data/file_list/remote/eeg1/eval_list_debug.txt", "file containing the list of evaluation EEG data")
tf.app.flags.DEFINE_string("eog_train_data", "", "file containing the list of training EOG data")
tf.app.flags.DEFINE_string("eog_eval_data", "", "file containing the list of evaluation EOG data")
tf.app.flags.DEFINE_string("emg_train_data", "", "file containing the list of training EMG data")
tf.app.flags.DEFINE_string("emg_eval_data", "", "file containing the list of evaluation EMG data")
tf.app.flags.DEFINE_string("out_dir", "./outputs/l_seq_sleepnet_train_test/", "Output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Checkpoint directory")
tf.app.flags.DEFINE_integer("training_epoch", 10, "Number of training epochs (default: 10)")

tf.app.flags.DEFINE_float("dropout_rnn", 0.9, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("nfilter", 32, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("nhidden1", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("attention_size", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("nhidden2", 64, "Sequence length (default: 20)")

tf.app.flags.DEFINE_integer("nclass_data", 4, "Number of classes in the data (whether artifacts are discarded or not is controlled in nclass_model)")
tf.app.flags.DEFINE_integer("nclass_model", 3, "Number of classes for sleep stage prediction (i.e. in mice, if artifacts are discarded, then nclass_model=3)")
tf.app.flags.DEFINE_integer("artifacts_label", 3, "Categorical label of the artifact class in the data")
tf.app.flags.DEFINE_integer("ndim", 129, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("frame_seq_len", 17, "Sequence length (default: 20)")

# subsuqence length
tf.app.flags.DEFINE_integer("sub_seq_len", 5, "Sequence length (default: 32)")
# number of subsequence
tf.app.flags.DEFINE_integer("nsubseq", 10, "number of overall segments (default: 9)")

tf.app.flags.DEFINE_boolean("early_stopping", False, "whether to apply early stopping (default: True)")
tf.app.flags.DEFINE_integer("early_stop_count", 100, "-")
tf.app.flags.DEFINE_integer("evaluate_every", 100, "-")
tf.app.flags.DEFINE_string("best_model_criteria", 'balanced_accuracy', "whether to save the model with best 'balanced_accuracy' or 'accuracy' (default: accuracy)")

# maximum number of evaluation steps to stop training. This can be used to control the number of training steps to be equivalent to SeqSleepNet
tf.app.flags.DEFINE_integer("max_eval_steps", 110, "Maximum number of evaluation steps to stop training (default: 110)")

# numbere of dualrnn encoder blocks to use (LSeqSleepNet can be deep by stacking multiple dual encoders)
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

with open(os.path.join(out_path,'training_settings.txt'), 'w') as f:
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

config.nclass = FLAGS.nclass_data
config.nclass_data = FLAGS.nclass_data
config.nclass_model = FLAGS.nclass_model
config.artifacts_label = FLAGS.artifacts_label
config.ndim = FLAGS.ndim
config.frame_seq_len = FLAGS.frame_seq_len
config.best_model_criteria = FLAGS.best_model_criteria

config.nsubseq = FLAGS.nsubseq
config.dualrnn_blocks = FLAGS.dualrnn_blocks
config.early_stop_count = FLAGS.early_stop_count
config.evaluate_every = FLAGS.evaluate_every

config.max_eval_steps = FLAGS.max_eval_steps
config.training_epoch = FLAGS.training_epoch*config.sub_seq_len*config.nsubseq

eeg_active = (FLAGS.eeg_train_data != "")
eog_active = (FLAGS.eog_train_data != "")
emg_active = (FLAGS.emg_train_data != "")

# 1 channel case
if (not eog_active and not emg_active):
    print("eeg active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             num_fold=config.num_fold_training_data, # load all data in one go
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len=config.sub_seq_len* config.nsubseq,
                                             nclasses = config.nclass_data,
                                             shuffle=True)
    valid_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_eval_data),
                                             num_fold=1, # load all data in one go
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len = config.sub_seq_len* config.nsubseq,
                                             nclasses = config.nclass_data,
                                             shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params_by_signal()
    valid_gen_wrapper.compute_eeg_normalization_params_by_signal()
    nchannel = 1

elif(eog_active and not emg_active):
    print("eeg and eog active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             eog_filelist=os.path.abspath(FLAGS.eog_train_data),
                                             num_fold=config.num_fold_training_data, # load all data in one go
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len=config.sub_seq_len* config.nsubseq,
                                             nclasses = config.nclass_data,
                                             shuffle=True)
    valid_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_eval_data),
                                             eog_filelist=os.path.abspath(FLAGS.eog_eval_data),
                                             num_fold=1, # load all data in one go
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len = config.subseq_len* config.nsubseq,
                                             nclasses = config.nclass_data,
                                             shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params_by_signal()
    train_gen_wrapper.compute_eog_normalization_params_by_signal()
    valid_gen_wrapper.compute_eeg_normalization_params_by_signal()
    valid_gen_wrapper.compute_eog_normalization_params_by_signal()
    nchannel = 2
elif(eog_active and emg_active):
    print("eeg, eog, and emg active")
    train_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_train_data),
                                             eog_filelist=os.path.abspath(FLAGS.eog_train_data),
                                             emg_filelist=os.path.abspath(FLAGS.emg_train_data),
                                             num_fold=config.num_fold_training_data, # load all data in one go
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len=config.sub_seq_len* config.nsubseq,
                                             nclasses = config.nclass_data,
                                             shuffle=True)
    valid_gen_wrapper = DataGeneratorWrapper(eeg_filelist=os.path.abspath(FLAGS.eeg_eval_data),
                                             eog_filelist=os.path.abspath(FLAGS.eog_eval_data),
                                             emg_filelist=os.path.abspath(FLAGS.emg_eval_data),
                                             num_fold=1, # load all data in one go
                                             data_shape_2=[config.frame_seq_len, config.ndim],
                                             seq_len=config.sub_seq_len* config.nsubseq,
                                             nclasses = config.nclass_data,
                                             shuffle=False)
    train_gen_wrapper.compute_eeg_normalization_params_by_signal()
    train_gen_wrapper.compute_eog_normalization_params_by_signal()
    train_gen_wrapper.compute_emg_normalization_params_by_signal()
    valid_gen_wrapper.compute_eeg_normalization_params_by_signal()
    valid_gen_wrapper.compute_eog_normalization_params_by_signal()
    valid_gen_wrapper.compute_emg_normalization_params_by_signal()
    nchannel = 3

# as there is only one fold, there is only one partition consisting all subjects,
# and next_fold should be called only once
valid_gen_wrapper.new_subject_partition() # next data fold
valid_gen_wrapper.next_fold()

config.nchannel = nchannel

# variable to keep track of best accuracy on validation set for model selection
best_acc = 0.0

# Training
# ==================================================
early_stop_count = 0

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_usage, allow_growth=True)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        net = LSeqSleepNet(config=config)

        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

        # initialize all variables
        print("Model initialized")
        sess.run(tf.initialize_all_variables())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            x_shape = x_batch.shape
            y_shape = y_batch.shape
            x = np.zeros(x_shape[:1] + (config.nsubseq, config.sub_seq_len,) + x_shape[2:])
            y = np.zeros(y_shape[:1] + (config.nsubseq, config.sub_seq_len,) + y_shape[2:])
            for s in range(config.nsubseq):
                x[:, s] = x_batch[:, s * config.sub_seq_len: (s + 1) * config.sub_seq_len]
                y[:, s] = y_batch[:, s * config.sub_seq_len: (s + 1) * config.sub_seq_len]

            frame_seq_len = np.ones(len(x_batch) * config.sub_seq_len * config.nsubseq,
                                        dtype=int) * config.frame_seq_len
            sub_seq_len = np.ones(len(x_batch) * config.nsubseq, dtype=int) * config.sub_seq_len
            inter_subseq_len = np.ones(len(x_batch) * config.sub_seq_len, dtype=int) * config.nsubseq
            feed_dict = {
                net.input_x: x,
                net.input_y: y,
                net.dropout_rnn: config.dropout_rnn,
                net.inter_subseq_len: inter_subseq_len,
                net.sub_seq_len: sub_seq_len,
                net.frame_seq_len: frame_seq_len,
                net.istraining: 1
            }
            _, step, output_loss, total_loss, acc, balanced_accuracy = sess.run(
                [train_op, global_step, net.output_loss, net.loss, net.accuracy, net.balanced_accuracy],
                feed_dict)
            
            return step, output_loss, total_loss, acc, balanced_accuracy

        def dev_step(x_batch, y_batch):
            x_shape = x_batch.shape
            y_shape = y_batch.shape
            x = np.zeros(x_shape[:1] + (config.nsubseq, config.sub_seq_len,) + x_shape[2:])
            y = np.zeros(y_shape[:1] + (config.nsubseq, config.sub_seq_len,) + y_shape[2:])
            for s in range(config.nsubseq):
                x[:, s] = x_batch[:, s * config.sub_seq_len: (s + 1) * config.sub_seq_len]
                y[:, s] = y_batch[:, s * config.sub_seq_len: (s + 1) * config.sub_seq_len]

            frame_seq_len = np.ones(len(x_batch) * config.sub_seq_len * config.nsubseq,
                                        dtype=int) * config.frame_seq_len
            sub_seq_len = np.ones(len(x_batch) * config.nsubseq, dtype=int) * config.sub_seq_len
            inter_subseq_len = np.ones(len(x_batch) * config.sub_seq_len, dtype=int) * config.nsubseq
            feed_dict = {
                net.input_x: x,
                net.input_y: y,
                net.dropout_rnn: 1.0,
                net.inter_subseq_len: inter_subseq_len,
                net.sub_seq_len: sub_seq_len,
                net.frame_seq_len: frame_seq_len,
                net.istraining: 0
            }
            output_loss, total_loss, yhat = sess.run([net.output_loss, net.loss, net.prediction], feed_dict)
            return output_loss, total_loss, yhat


        def _evaluate(gen, log_filename, config):
            # Validate the model on the entire data stored in gen variable
            output_loss =0
            total_loss = 0

            yhat = np.zeros([len(gen.data_index), config.sub_seq_len*config.nsubseq])
            # increase the batch size by this factor to better utilize the GPU
            factor = 20*4

            # test with minibatch of 10x training minibatch to speed up
            num_batch_per_epoch = np.floor(len(gen.data_index) / (factor*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x_batch, y_batch, label_batch_ = gen.next_batch(factor*config.batch_size)
                output_loss_, total_loss_, yhat_ = dev_step(x_batch, y_batch)
                output_loss += output_loss_
                total_loss += total_loss_
                for s in range(config.nsubseq):
                    yhat[(test_step - 1) * factor * config.batch_size: test_step * factor * config.batch_size,
                    s*config.sub_seq_len:(s+1)*config.sub_seq_len] = yhat_[:, s]
                test_step += 1
            if(gen.pointer < len(gen.data_index)):
                actual_len, x_batch, y_batch, label_batch_ = gen.rest_batch(factor*config.batch_size)
                output_loss_, total_loss_, yhat_ = dev_step(x_batch, y_batch)
                output_loss += output_loss_
                total_loss += total_loss_
                for s in range(config.nsubseq):
                    yhat[(test_step - 1) * factor * config.batch_size: len(gen.data_index),
                    s * config.sub_seq_len:(s + 1) * config.sub_seq_len] = yhat_[:, s]
            yhat = yhat + 1

            acc = 0
            bal_acc = 0
            with open(os.path.join(out_dir, log_filename), "a") as text_file:
                text_file.write("{:g} {:g} ".format(output_loss, total_loss))
                for n in range(config.sub_seq_len*config.nsubseq):
                    yhat_n = yhat[:,n]
                    y_n = gen.label[gen.data_index - (config.sub_seq_len*config.nsubseq - 1) + n]

                    if config.nclass_model != config.nclass_data and config.artifacts_label != None:
                        yhat_n = yhat_n[y_n != config.artifacts_label+1]
                        y_n = y_n[y_n != config.artifacts_label+1]

                    acc_n = accuracy_score(y_n, yhat_n) # due to zero-indexing
                    bal_acc_n = balanced_accuracy_score(y_n, yhat_n)

                    if n == config.sub_seq_len*config.nsubseq - 1:
                        text_file.write("{:g} {:g} \n".format(acc_n, bal_acc_n))
                    else:
                        text_file.write("{:g} {:g} ".format(acc_n, bal_acc_n))
                    acc += acc_n
                    bal_acc += bal_acc_n
            acc /= (config.sub_seq_len*config.nsubseq)
            bal_acc /= (config.sub_seq_len*config.nsubseq)

            return acc, bal_acc, yhat, output_loss, total_loss

        # test the model first
        # print("{} Start off validation".format(datetime.now()))
        # eval_acc, eval_bal_acc, eval_yhat, eval_output_loss, eval_total_loss = \
        #     _evaluate(gen=valid_gen_wrapper.gen, log_filename="eval_result_log.txt", config=config)
        # valid_gen_wrapper.gen.reset_pointer()

        start_time = time.time()
        # Loop over number of epochs
        for epoch in range(config.training_epoch):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
            train_gen_wrapper.new_subject_partition()

            for data_fold in range(config.num_fold_training_data):
                train_gen_wrapper.next_fold()

                train_batches_per_epoch = np.floor(len(train_gen_wrapper.gen.data_index) / config.batch_size / config.sub_seq_len / config.nsubseq).astype(np.uint32)

                step = 1
                while step < train_batches_per_epoch:
                    # Get a batch
                    x_batch, y_batch, label_batch = train_gen_wrapper.gen.next_batch(config.batch_size)
                    train_step_, train_output_loss_, train_total_loss_, acc_, bal_acc_ = train_step(x_batch, y_batch)
                    time_str = datetime.now().isoformat()

                    print("{}: step {}, output_loss {}, total_loss {} acc {} bal_acc {}".format(time_str, train_step_, train_output_loss_, train_total_loss_, acc_, bal_acc_))
                    with open(os.path.join(out_dir, "train_log.txt"), "a") as text_file:
                        text_file.write("{:g} {:g} {:g} {:g} {:g}\n".format(train_step_, train_output_loss_, train_total_loss_, acc_, bal_acc_))
                    step += 1

                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % config.evaluate_every == 0:
                        # Validate the model on the validation set
                        print("{} Start validation".format(datetime.now()))

                        eval_acc, eval_bal_acc, eval_yhat, eval_output_loss, eval_total_loss = \
                        _evaluate(gen=valid_gen_wrapper.gen, log_filename="eval_result_log.txt", config=config)

                        if config.best_model_criteria == 'accuracy':
                            tracked_acc = eval_acc
                        elif config.best_model_criteria == 'balanced_accuracy':
                            tracked_acc = eval_bal_acc

                        early_stop_count += 1
                        if(tracked_acc >= best_acc):
                            early_stop_count = 0 # reset
                            best_acc = tracked_acc
                            checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) +'.ckpt')
                            save_path = saver.save(sess, checkpoint_name)

                            print("Best model updated")
                            source_file = checkpoint_name
                            dest_file = os.path.join(checkpoint_path, 'best_model_acc')
                            shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                            shutil.copy(source_file + '.index', dest_file + '.index')
                            shutil.copy(source_file + '.meta', dest_file + '.meta')

                            # write current best performance to file
                            with open(os.path.join(out_dir, "current_best.txt"), "a") as text_file:
                                text_file.write("{:g}\n".format(tracked_acc))

                        valid_gen_wrapper.gen.reset_pointer()

                        if(FLAGS.early_stopping == True):
                            print('EARLY STOPPING enabled!')
                            # early stopping only after 200 evaluation steps
                            if (current_step/config.evaluate_every >= 200 and early_stop_count >= config.early_stop_count):
                                end_time = time.time()
                                with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
                                    text_file.write("{:g}\n".format((end_time - start_time)))
                                quit()
                        else:
                            if(current_step/config.evaluate_every >= config.max_eval_steps):
                                print('Maximum training step reached!')
                                quit()

            # train_gen_wrapper.gen.reset_pointer()
            # train_gen_wrapper.gen.shuffle_data()

        end_time = time.time()
        with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
            text_file.write("{:g}\n".format((end_time - start_time)))
