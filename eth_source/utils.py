import time
import os
from tensorflow.python.client import timeline
import numpy as np
import quaternion
import cv2
import gspread
from oauth2client.service_account import ServiceAccountCredentials

class Logger:
    def __init__(self, filename, stdout=None):
        self.stdout = stdout
        self.logfile = open(filename, 'w')

    def print(self, text):
        if self.stdout is not None:
            self.stdout.write(text)
        self.logfile.write(text)

    def close(self):
        self.logfile.close()

class GoogleSheetLogger:
    def __init__(self, sheet_name):
        self.sheet_name = sheet_name
        # use creds to create a client to interact with the Google Drive API
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('../experiment-logger-5f2ecef8f4ce.json', scope)
        client = gspread.authorize(creds)

        # Find a workbook by name.
        self.workbook = client.open("experiment_logs")
        self.sheet = self.workbook.worksheet(sheet_name)
        self.header = self.sheet.row_values(1)

    def append_row(self, values):
        if isinstance(values, list):
            self.sheet.append_row(values)
        elif isinstance(values, dict):
            values_list = []
            skip_column = 1
            len_header = 1
            for header in self.header:
                if header in values:
                    values_list.extend(values[header])
                    len_header = len(values[header])
                    skip_column = 1
                elif skip_column >= len_header:
                    values_list.append(None)
                    skip_column = 1
                    len_header = 1
                elif skip_column < len_header:
                    skip_column += 1

            self.sheet.append_row(values_list)

def get_model_dir_timestamp(prefix="", suffix="", connector="_"):
    """
    Creates a directory name based on timestamp.

    Args:
        prefix:
        suffix:
        connector: one connector character between prefix, timestamp and suffix.

    Returns:

    """
    return prefix+connector+str(int(time.time()))+connector+suffix


def create_tf_timeline(model_dir, run_metadata):
    """
    This is helpful for profiling slow Tensorflow code.

    Args:
        model_dir:
        run_metadata:

    Returns:

    """
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    timeline_file_path = os.path.join(model_dir,'timeline.json')
    with open(timeline_file_path, 'w') as f:
        f.write(ctf)

def split_data_dictionary(dictionary, split_indices, keys_frozen=[], verbose=1):
    """
    Splits the data dictionary of lists into smaller chunks. All (key,value) pairs must have the same number of
    elements. If there is an index error, then the corresponding (key, value) pair is copied directly to the new
    dictionaries.

    Args:
        dictionary (dict): data dictionary.
        split_indices (list): Each element contains a list of indices for one split. Multiple splits are supported.
        keys_frozen (list): list of keys that are to be copied directly. The remaining (key,value) pairs will be
            used in splitting. If not provided, all keys will be used.
        verbose (int): status messages.

    Returns:
        (tuple): a tuple containing chunks of new dictionaries.
    """
    # Find <key, value> pairs having an entry per sample.
    sample_level_keys = []
    dataset_level_keys = []

    num_samples = sum([len(l) for l in split_indices])
    for key, value in dictionary.items():
        if not(key in keys_frozen) and ((isinstance(value, np.ndarray) or isinstance(value, list)) and (len(value) == num_samples)):
            sample_level_keys.append(key)
        else:
            dataset_level_keys.append(key)
            print(str(key) + " is copied.")

    chunks = []
    for chunk_indices in split_indices:
        dict = {}

        for key in dataset_level_keys:
            dict[key] = dictionary[key]
        for key in sample_level_keys:
            dict[key] = [dictionary[key][i] for i in chunk_indices]
        chunks.append(dict)

    return tuple(chunks)

def aa_to_rot_matrix(data):
    """
    Converts the orientation data to represent angle axis as rotation matrices. `data` is expected in format
    (seq_length, n*3). Returns an array of shape (seq_length, n*9).
    """
    # reshape to have sensor values explicit
    data_c = np.array(data, copy=True)
    seq_length, n = data_c.shape[0], data_c.shape[1] // 3
    data_r = np.reshape(data_c, [seq_length, n, 3])

    qs = quaternion.from_rotation_vector(data_r)
    rot = np.reshape(quaternion.as_rotation_matrix(qs), [seq_length, n, 9])

    return np.reshape(rot, [seq_length, 9*n])


def rot_matrix_to_aa(data):
    """
    Converts the orientation data given in rotation matrices to angle axis representation. `data` is expected in format
    (seq_length, n*9). Returns an array of shape (seq_length, n*3).
    """
    seq_length, n_joints = data.shape[0], data.shape[1]//9
    data_r = np.reshape(data, [seq_length, n_joints, 3, 3])
    data_c = np.zeros([seq_length, n_joints, 3])
    for i in range(seq_length):
        for j in range(n_joints):
            data_c[i, j] = np.ravel(cv2.Rodrigues(data_r[i, j])[0])
    return np.reshape(data_c, [seq_length, n_joints*3])

def get_seq_len_histogram(sequence_length_array, num_bins=10, collapse_first_and_last_bins=[1, -1]):
    """
    Creates a histogram of sequence-length.
    Args:
        sequence_length_array: numpy array of sequence length for all samples.
        num_bins:
        collapse_first_and_last_bins: selects bin edges between the provided indices by discarding from the first and
            last bins.
    Returns:
        (list): bin edges.
    """
    h, bins = np.histogram(sequence_length_array, bins=num_bins)
    if collapse_first_and_last_bins is not None:
        return [int(b) for b in bins[collapse_first_and_last_bins[0]:collapse_first_and_last_bins[1]]]
    else:
        return [int(b) for b in bins]

