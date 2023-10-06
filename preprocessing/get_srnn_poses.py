"""
A simple script to extract SRNN poses as .npz file which is required by
srnn_poses_in_euler.py. The code is based on Martinez et al. (2017) and on
Aksan et al. (2021)
"""

import numpy as np
from six.moves import xrange
from common.conversions import aa2rotmat, rotmat2euler


def read_csv_as_float(filename):
    out_array = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(",")
        if len(line) > 0:
            out_array.append(np.array([np.float32(x) for x in line]))

    return np.array(out_array)


def load_data(path_to_dataset, subjects, actions):
    data = {}
    for subj in subjects:
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]

            for subact in [1, 2]:  # subactions
                print(
                    "Reading subject {0}, action {1}, subaction {2}".format(
                        subj, action, subact
                    )
                )

                filename = "{0}/S{1}/{2}_{3}.txt".format(
                    path_to_dataset, subj, action, subact
                )
                action_sequence = read_csv_as_float(filename)

                n, d = action_sequence.shape
                even_list = range(0, n, 2)

                data[(subj, action, subact, "even")] = action_sequence[even_list, :]

    return data


def find_indices_srnn(data, action):
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    subject = 5
    subaction1 = 1
    subaction2 = 2

    T1 = data[(subject, action, subaction1, "even")].shape[0]
    T2 = data[(subject, action, subaction2, "even")].shape[0]
    prefix, suffix = 50, 100

    idx = []
    idx.append(rng.randint(16, T1 - prefix - suffix))
    idx.append(rng.randint(16, T2 - prefix - suffix))
    idx.append(rng.randint(16, T1 - prefix - suffix))
    idx.append(rng.randint(16, T2 - prefix - suffix))
    idx.append(rng.randint(16, T1 - prefix - suffix))
    idx.append(rng.randint(16, T2 - prefix - suffix))
    idx.append(rng.randint(16, T1 - prefix - suffix))
    idx.append(rng.randint(16, T2 - prefix - suffix))
    return idx


def get_batch_srnn(data, action):
    actions = [
        "directions",
        "discussion",
        "eating",
        "greeting",
        "phoning",
        "posing",
        "purchases",
        "sitting",
        "sittingdown",
        "smoking",
        "takingphoto",
        "waiting",
        "walking",
        "walkingdog",
        "walkingtogether",
    ]

    if not action in actions:
        raise ValueError("Unrecognized action {0}".format(action))

    frames = {}
    frames[action] = find_indices_srnn(data, action)

    batch_size = 8  # we always evaluate 8 seeds
    subject = 5  # we always evaluate on subject 5
    source_seq_len = 50
    target_seq_len = 25
    seq_len = source_seq_len + target_seq_len
    num_joints = 33

    seeds = [(action, (i % 2) + 1, frames[action][i]) for i in range(batch_size)]

    srnn_data = {"pose": [], "euler": []}

    for i in xrange(batch_size):
        _, subsequence, idx = seeds[i]
        idx = idx + 50

        data_sel = data[(subject, action, subsequence, "even")]

        data_aa = data_sel[(idx - source_seq_len) : (idx + target_seq_len), :]

        data_aa_reshaped = np.reshape(data_aa, [seq_len, -1, 3])
        data_rotmat_reshaped = aa2rotmat(data_aa_reshaped)
        data_eul_reshaped = rotmat2euler(data_rotmat_reshaped)

        data_eul = np.reshape(data_eul_reshaped, [seq_len, num_joints * 3])

        srnn_data["pose"].append(data_aa)
        srnn_data["euler"].append(data_eul)

    return srnn_data


if __name__ == "__main__":
    actions = [
        "directions",
        "discussion",
        "eating",
        "greeting",
        "phoning",
        "posing",
        "purchases",
        "sitting",
        "sittingdown",
        "smoking",
        "takingphoto",
        "waiting",
        "walking",
        "walkingdog",
        "walkingtogether",
    ]
    subjects = [5]

    seed_name = ""
    path_to_dataset = "<path-to>/h3.6m/dataset"
    save_path = "<path-to>/h3.6m/tfrecords/martinez_euler_gt_25fps{}.npz".format(
        seed_name
    )

    # load the dataset
    data = load_data(path_to_dataset, subjects, actions)

    # get the SRNN poses for the evaluation as the format required
    # by srnn_poses_in_euler.py
    test_data = {}
    for action in actions:
        test_data[action] = get_batch_srnn(data, action)

    # save the SRNN poses to be fed to srnn_poses_in_euler.py
    srnn_poses = {}
    srnn_poses["data"] = test_data
    np.savez(save_path, **srnn_poses)

    print("SRNN poses saved at:", save_path)
    print("Done!")
