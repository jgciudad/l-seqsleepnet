import numpy as np
import h5py
from datagenerator_from_list_v3 import DataGenerator3

class DataGeneratorWrapper:
    def __init__(self, eeg_filelist=None, eog_filelist=None, emg_filelist=None, num_fold=1, data_shape_2=np.array([29, 128]), seq_len = 20, nclasses = 4, artifact_detection = False, artifacts_label = 3, shuffle=False):

        # Init params

        self.eeg_list_of_files = []
        self.eog_list_of_files = []
        self.emg_list_of_files = []
        self.file_sizes = []

        self.data_shape_2 = data_shape_2

        # how many folds the data is split to fold-wise loading
        self.num_fold = num_fold
        # current fold index
        self.current_fold = 0
        # subjects of the fold
        self.sub_folds = []

        self.seq_len = seq_len
        self.Ncat = nclasses
        self.artifact_detection = artifact_detection
        self.artifacts_label = artifacts_label

        self.shuffle = shuffle

        # list of files and their size
        if eeg_filelist is not None:
            self.eeg_list_of_files, self.file_sizes = self.read_file_list(eeg_filelist)
        if eog_filelist is not None:
            self.eog_list_of_files, _ = self.read_file_list(eog_filelist)
        if emg_filelist is not None:
            self.emg_list_of_files, _ = self.read_file_list(emg_filelist)

        self.eeg_meanX, self.eeg_stdX = None, None
        self.eog_meanX, self.eog_stdX = None, None
        self.emg_meanX, self.emg_stdX = None, None

        # data generator
        self.gen = None

    # read in a list of file
    def read_file_list(self, filelist):
        list_of_files = []
        file_sizes = []
        with open(filelist) as f:
            lines = f.readlines()
            for l in lines:
                print(l)
                items = l.split()
                list_of_files.append(items[0])
                file_sizes.append(int(items[1]))
        return list_of_files, file_sizes

    def compute_eeg_normalization_params(self):
        if(len(self.eeg_list_of_files) == 0):
            return
        self.eeg_meanX, self.eeg_stdX = self.load_data_compute_norm_params(self.eeg_list_of_files)

    def compute_eeg_normalization_params_by_signal(self):
        if(len(self.eeg_list_of_files) == 0):
            return
        self.eeg_meanX, self.eeg_stdX = self.load_data_compute_norm_params_by_signal(self.eeg_list_of_files)

    def compute_eog_normalization_params(self):
        if(len(self.eog_list_of_files) == 0):
            return
        self.eog_meanX, self.eog_stdX = self.load_data_compute_norm_params(self.eog_list_of_files)

    def compute_eog_normalization_params_by_signal(self):
        if(len(self.eog_list_of_files) == 0):
            return
        self.eog_meanX, self.eog_stdX = self.load_data_compute_norm_params_by_signal(self.eog_list_of_files)

    def compute_emg_normalization_params(self):
        if(len(self.eog_list_of_files) == 0):
            return
        self.emg_meanX, self.emg_stdX = self.load_data_compute_norm_params(self.emg_list_of_files)

    def compute_emg_normalization_params_by_signal(self):
        if(len(self.eog_list_of_files) == 0):
            return
        self.emg_meanX, self.emg_stdX = self.load_data_compute_norm_params_by_signal(self.emg_list_of_files)

    def set_eeg_normalization_params(self, meanX, stdX):
        self.eeg_meanX, self.eeg_stdX = meanX, stdX

    def set_eog_normalization_params(self, meanX, stdX):
        self.eog_meanX, self.eog_stdX = meanX, stdX

    def set_emg_normalization_params(self, meanX, stdX):
        self.emg_meanX, self.emg_stdX = meanX, stdX

    # read data from mat files in the list stored in the file 'filelist'
    # and compute normalization parameters on the flight
    def load_data_compute_norm_params(self, list_of_files):
        meanX = None
        meanXsquared = None
        count = 0
        print('Computing normalization parameters')
        for i in range(len(list_of_files)):
            X2 = self.read_X2_from_mat_file(list_of_files[i].strip())
            Ni = len(X2)
            X2 = np.reshape(X2,(Ni*self.data_shape_2[0], self.data_shape_2[1]))

            meanX_i = X2.mean(axis=0)
            X2_squared = np.square(X2)
            meanXsquared_i = X2_squared.mean(axis=0)
            del X2

            if meanX is None:
                meanX = meanX_i
                meanXsquared = meanXsquared_i
            else:
                meanX = (meanX*count + meanX_i*Ni)/(count + Ni)
                meanXsquared = (meanXsquared*count + meanXsquared_i*Ni)/(count + Ni)
            count += Ni
        varX = -np.multiply(meanX, meanX) + meanXsquared
        stdX = np.sqrt(varX*count/(count-1))
        return meanX, stdX

    def load_data_compute_norm_params_by_signal(self, list_of_files):
        meanX = None
        meanXsquared = None
        count = 0
        print('Computing normalization parameters')
        means = {}
        stds = {}
        for i in range(len(list_of_files)):
            X2 = self.read_X2_from_mat_file(list_of_files[i].strip())
            Ni = len(X2)
            X2 = np.reshape(X2,(Ni*self.data_shape_2[0], self.data_shape_2[1]))

            meanX_i = X2.mean(axis=0)
            stdX_i = X2.std(axis=0)

            means[list_of_files[i]] = meanX_i
            stds[list_of_files[i]] = stdX_i

        return means, stds

    # shuffle the subjects for a new partition
    def new_subject_partition(self):
        if(self.shuffle is False):
            subject = range(len(self.file_sizes))
        else:
            subject = np.random.permutation(len(self.file_sizes))

        self.sub_folds = []
        Nsub = len(self.file_sizes)
        for i in range(self.num_fold):
            fold_i = list(range((i*Nsub)//self.num_fold, ((i+1)*Nsub)//self.num_fold))
            self.sub_folds.append([subject[k] for k in fold_i])
        self.current_fold = 0


    def read_X2_from_mat_file(self,filename):
        """
        Read in X2 data from a data file in mat file HD5F file
        """
        # Load data
        print(filename)
        data = h5py.File(filename,'r')
        data.keys()
        X2 = np.array(data['X2']) # time-frequency input
        X2 = np.transpose(X2, (2, 1, 0))  # rearrange dimension
        return X2

    def is_last_fold(self):
        return (self.current_fold == self.num_fold-1)

    def next_fold(self):
        if(self.current_fold == self.num_fold):
            self.new_subject_partition()
            self.current_fold = 0

        # at lest eeg active
        print('Current fold: ')
        print(self.current_fold)
        ind = self.sub_folds[int(self.current_fold)]
        print('Current-fold subjects: ')
        print(ind)
        list_of_files = [self.eeg_list_of_files[int(i)] for i in ind]
        file_sizes = [self.file_sizes[int(i)] for i in ind]
        self.gen = DataGenerator3(list_of_files,
                                 file_sizes,
                                 data_shape_2=self.data_shape_2,
                                 seq_len=self.seq_len,
                                 Ncat=self.Ncat, 
                                 artifact_detection=self.artifact_detection,
                                 artifacts_label=self.artifacts_label)
        self.gen.normalize_by_signal(self.eeg_meanX, self.eeg_stdX)

        if(len(self.eog_list_of_files) > 0):
            list_of_files = [self.eog_list_of_files[i] for i in ind]
            eog_gen = DataGenerator3(list_of_files,
                                 file_sizes,
                                 data_shape_2=self.data_shape_2,
                                 seq_len=self.seq_len,
                                 Ncat=self.Ncat, 
                                 artifact_detection=self.artifact_detection, 
                                 artifacts_label=self.artifacts_label)
            eog_gen.normalize_by_signal(self.eog_meanX, self.eog_stdX)

        if(len(self.emg_list_of_files) > 0):
            list_of_files = [self.emg_list_of_files[i] for i in ind]
            emg_gen = DataGenerator3(list_of_files,
                                 file_sizes,
                                 data_shape_2=self.data_shape_2,
                                 seq_len=self.seq_len,
                                 Ncat=self.Ncat, 
                                 artifact_detection=self.artifact_detection, 
                                 artifacts_label=self.artifacts_label)
            emg_gen.normalize_by_signal(self.emg_meanX, self.emg_stdX)

        # 1-channel case when both eog and emg not active
        if(len(self.eog_list_of_files) == 0 and len(self.emg_list_of_files) == 0):
            self.gen.X2 = np.expand_dims(self.gen.X2, axis=-1) # expand channel dimension
            self.gen.data_shape_2 = self.gen.X2.shape[1:]
        # 2-channel input case
        elif(len(self.eog_list_of_files) > 0 and len(self.emg_list_of_files) == 0):
            self.gen.X2 = np.stack((self.gen.X2, eog_gen.X2), axis=-1) # merge and make new dimension
            self.gen.data_shape_2 = self.gen.X2.shape[1:]
        # 3-channel input case
        elif(len(self.eog_list_of_files) > 0 and len(self.emg_list_of_files) >0):
            self.gen.X2 = np.stack((self.gen.X2, eog_gen.X2, emg_gen.X2), axis=-1) # merge and make new dimension
            self.gen.data_shape_2 = self.gen.X2.shape[1:]

        if(len(self.eog_list_of_files) > 0):
            del eog_gen
        if(len(self.emg_list_of_files) > 0):
            del emg_gen

        self.current_fold += 1

        if(self.shuffle):
            self.gen.shuffle_data()

