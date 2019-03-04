import os
import numpy as np
import matplotlib.pyplot as plt
from data_utils import readCSVasFloat


def plot(joints, angles_only, title):
    angles = np.linalg.norm(joints, axis=-1)
    n_joints = joints.shape[1]

    for i in range(n_joints):
        ax = plt.subplot(n_joints, 1, i + 1)

        # plot distances on the left axis
        ax.plot(angles[:, i], '-k', alpha=0.5)
        # ax.set_ylim([0, 2.0])
        # ax.set_ylabel('angle', color='k')
        # ax.tick_params('y', colors='k')

        if not angles_only:
            ax2 = ax.twinx()
            ax2.plot(joints[:, i, 0], '-r', alpha=0.5)
            ax2.plot(joints[:, i, 1], '-g', alpha=0.5)
            ax2.plot(joints[:, i, 2], '-b', alpha=0.5)
            # ax2.set_ylabel('axis', color='r')
            # ax2.tick_params('y', colors='r')

        # ax.set_title(sensor_names[i])

    plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    # how many samples to visualize
    n_samples = 10
    # only plot angles per joint
    angles_only = False

    # parse all data files
    data_files = []
    for root_dir, dirnames, fnames in sorted(os.walk('../data/h3.6m/dataset')):
        for f in fnames:
            if f.endswith('.txt'):
                data_files.append(os.path.join(root_dir, f))

    idx = np.random.randint(0, len(data_files), n_samples)

    for i in idx:
        fname = data_files[i]
        data = readCSVasFloat(fname)  # shape (seq_length, 99)

        # only visualize the joints that are not zero everywhere
        joints = np.reshape(data, [data.shape[0], -1, 3])
        data_std = np.std(data, axis=0)

        joints_to_ignore = np.where(np.all(np.reshape(data_std, [-1, 3]) < 1e-4, axis=-1))[0]
        joints_to_ignore = np.insert(joints_to_ignore, 0, 0)
        joints_to_use = [x for x in range(joints.shape[1]) if x not in joints_to_ignore]
        joints = joints[:, joints_to_use]  # shape (seq_length, n_joints, 3)

        plot(joints, angles_only, fname)

