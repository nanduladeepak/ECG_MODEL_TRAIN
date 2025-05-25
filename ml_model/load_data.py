import pandas as pd
import pandas as pd
import numpy as np
import wfdb
from tqdm import tqdm
import ast
from scipy import signal

# import cv2
# from skimage.morphology import skeletonize
# from scipy import ndimage
# import tensorflow as tf


def lowpass_scipy(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data)
    return filtered_data

def highpass_scipy(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data)
    return filtered_data



def butter_bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=500.0, order=5):
    """
    Applies a Butterworth bandpass filter to the input signal.

    Parameters:
    - signal: np.ndarray
        The ECG signal to be filtered.
    - lowcut: float
        The lower cutoff frequency of the filter in Hz.
    - highcut: float
        The upper cutoff frequency of the filter in Hz.
    - fs: float
        The sampling frequency of the signal in Hz.
    - order: int
        The order of the Butterworth filter.

    Returns:
    - filtered_signal: np.ndarray
        The bandpass-filtered ECG signal.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    filtered_signal = signal.sosfiltfilt(sos, signal)
    return filtered_signal

class DataLoader:
    def __init__(self, path= './ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/', sampling_rate=100, poles=5, upperCutoff = 15):
        self.poles = poles
        self.upperCutoff = upperCutoff
        self.path = path
        self.sampling_rate = sampling_rate
        self.rawData_df = None
        self.raw_ecg_signal = None
        self.agg_df = None
        self.x_all = None
        self.y_all = None
        self.ecg_df = None
        self.target_columns = ['NORM', 'MI', 'STTC', 'HYP', 'CD']
        self.target_sub_columns = [
                'sub_NORM',
                # MI
                'sub_IMI', 'sub_LMI', 'sub_AMI', 'sub_PMI',
                # STTC
                'sub_STTC','sub_NST_', 'sub_ISCA', 'sub_ISC_', 'sub_ISCI',
                # HYP
                'sub_LVH', 'sub_RAO/RAE', 'sub_RVH', 'sub_SEHYP', 'sub_LAO/LAE',
                # CD
                'sub_LAFB/LPFB', 'sub_IRBBB', 'sub_IVCD', 'sub__AVB',  'sub_CRBBB', 'sub_CLBBB', 'sub_ILBBB', 'sub_WPW',
            ]
        self.X_train = None
        self.X_test = None
        self.X_val = None
        
        self.X_flt_train = None
        self.X_flt_test = None
        self.X_flt_val = None
        
        self.y_train = None
        self.y_test = None
        self.y_val = None
        
        self.y_sub_train = None
        self.y_sub_test = None
        self.y_sub_val = None
        self.main()
        
        
    
    def __load_data(self):
        self.rawData_df = pd.read_csv(f'{self.path}/ptbxl_database.csv',index_col='ecg_id')
        self.rawData_df.scp_codes = self.rawData_df.scp_codes.apply(lambda x: ast.literal_eval(x))
        
    def __load_ecg_data(self):
        if self.sampling_rate == 100:
            data = [wfdb.rdsamp(self.path+f) for f in tqdm(self.rawData_df.filename_lr)]
        else:
            data = [wfdb.rdsamp(self.path+f) for f in tqdm(self.rawData_df.filename_hr)]
        self.raw_ecg_signal = np.array([signal for signal, meta in data])
        
    def __load_agg_data(self):
        # Load scp_statements.csv for diagnostic aggregation
        self.agg_df = pd.read_csv(self.path+'scp_statements.csv', index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]
        
    def _aggregate_supclass_diagnostic(self, y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    def _aggregate_subclass_diagnostic(self, y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_subclass)
        ret = list(set(tmp))
        ret = ['sub_'+r for r in ret] # to distinguish between subclass and superclass columns
        return ret
    
    def __update_cols(self, row):
        for sc in row['diagnostic_superclass']:
            row[sc] = 1
        for sc in row['diagnostic_subclass']:
            row[sc] = 1
            
        return row
    
    def __get_data_by_folds(self, update_cols):
        feature_cols = ['age', 'sex', 'height', 'weight', 'nurse', 'site', 'device',] # could add more columns as features
        folds = np.arange(1, 11)
        assert len(folds)  > 0, '# of provided folds should longer than 1'

        filt = np.isin(self.rawData_df.strat_fold.values, folds)
        self.x_all = self.raw_ecg_signal[filt]
        y_selected = self.rawData_df[filt]
        
        for sc in update_cols:
            y_selected[sc] = 0
            
        self.cols = update_cols
        
        y_selected = y_selected.apply(self.__update_cols, axis=1)
        
        self.y_all = y_selected[list(feature_cols)+list(update_cols)+['strat_fold']]
    
    def __preprocess_raw_data(self):
        # Apply diagnostic superclass
        self.rawData_df['diagnostic_superclass'] = self.rawData_df.scp_codes.apply(self._aggregate_supclass_diagnostic)
        self.rawData_df['diagnostic_superclass_len'] = self.rawData_df['diagnostic_superclass'].apply(len)
        self.rawData_df.loc[self.rawData_df.diagnostic_superclass_len > 1, 'diagnostic_superclass']
        
        # Apply diagnostic subclass
        self.rawData_df['diagnostic_subclass'] = self.rawData_df.scp_codes.apply(self._aggregate_subclass_diagnostic)
        self.rawData_df['diagnostic_subclass_len'] = self.rawData_df['diagnostic_subclass'].apply(len)
        self.rawData_df.loc[self.rawData_df.diagnostic_subclass_len > 1, 'diagnostic_subclass']
        
        all_superclass = pd.Series(np.concatenate(self.rawData_df['diagnostic_superclass'].values))
        all_subclass = pd.Series(np.concatenate(self.rawData_df['diagnostic_subclass'].values))
        superclass_cols = all_superclass.unique()
        subclass_cols = all_subclass.unique()
        update_cols = np.concatenate([superclass_cols, subclass_cols]) # add meta data columns
        self.__get_data_by_folds(update_cols)
        
    
    def __final_df(self):
        self.ecg_df = self.y_all.copy()
        self.ecg_df['ecg_heads'] = list(self.x_all.transpose(0, 2, 1))
        
    def __prepare_df_to_train(self):
        df_shuffled = self.ecg_df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df_shuffled)
        train_end = int(0.6 * n)
        test_end = train_end + int(0.2 * n)

        train_df = df_shuffled.iloc[:train_end]
        test_df = df_shuffled.iloc[train_end:test_end]
        val_df = df_shuffled.iloc[test_end:]

        # Set model input as ecg_heads and model output as hot-encoded classes ['NORM','MI','STTC','HYP','CD']
        self.X_train = np.expand_dims(np.array(train_df['ecg_heads'].tolist()), -1)
        self.X_test = np.expand_dims(np.array(test_df['ecg_heads'].tolist()), -1)
        self.X_val = np.expand_dims(np.array(val_df['ecg_heads'].tolist()), -1)

        
        self.X_flt_train = butter_bandpass_filter(np.array(train_df['ecg_heads'].tolist()), lowcut = 1, highcut = self.upperCutoff, fs = self.sampling_rate, order = self.poles)
        self.X_flt_test = butter_bandpass_filter(np.array(test_df['ecg_heads'].tolist()), lowcut = 1, highcut = self.upperCutoff, fs = self.sampling_rate, order = self.poles)
        self.X_flt_val = butter_bandpass_filter(np.array(val_df['ecg_heads'].tolist()), lowcut = 1, highcut = self.upperCutoff, fs = self.sampling_rate, order = self.poles)

        self.y_train = train_df[self.target_columns].values
        self.y_test = test_df[self.target_columns].values
        self.y_val = val_df[self.target_columns].values
        
        self.y_sub_train = train_df[self.target_sub_columns].values
        self.y_sub_test = test_df[self.target_sub_columns].values
        self.y_sub_val = val_df[self.target_sub_columns].values
    
    def main(self):
        self.__load_data()
        self.__load_agg_data()
        self.__load_ecg_data()
        self.__preprocess_raw_data()
        self.__final_df()
        self.__prepare_df_to_train()

    def get_in(self):
        return self.X_train, self.X_test, self.X_val

    def get_flt_in(self):
        return self.X_flt_train, self.X_flt_test, self.X_flt_val

    def get_class_out(self):
        return self.y_train, self.y_test, self.y_val

    def get_sub_class_out(self):
        return self.y_sub_train, self.y_sub_test, self.y_sub_val


# 'NORM',
# 'MI',
# 'STTC',
# 'HYP',
# 'CD'


# 'sub_NORM',
# 'sub_IMI',
# 'sub_STTC',
# 'sub_NST_',
# 'sub_LVH',
# 'sub_LAFB/LPFB',
# 'sub_RAO/RAE',
# 'sub_IRBBB',
# 'sub_RVH',
# 'sub_IVCD',
# 'sub_LMI',
# 'sub_AMI',
# 'sub__AVB',
# 'sub_ISCA',
# 'sub_ISC_',
# 'sub_SEHYP',
# 'sub_ISCI',
# 'sub_CRBBB',
# 'sub_CLBBB',
# 'sub_LAO/LAE',
# 'sub_ILBBB',
# 'sub_WPW',
# 'sub_PMI'