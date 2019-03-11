import numpy as np
from matplotlib import pyplot as plt, animation as animation
from mpl_toolkits.mplot3d import Axes3D

from smpl import SMPLForwardKinematicsNP
from smpl import SMPL_MAJOR_JOINTS
from smpl import SMPL_PARENTS
from motion_metrics import get_closest_rotmat
from motion_metrics import is_valid_rotmat
from viz import save_animation


class Visualizer(object):
    """
    Helper class to visualize SMPL joint angle input.
    """
    def __init__(self, smpl_model_path):
        self.smpl_fk = SMPLForwardKinematicsNP(smpl_model_path)

    def visualize(self, seed, prediction, target):
        """
        Visualize prediction and ground truth side by side. At the moment only supports sparse pose input in rotation
        matrix format.
        Args:
            seed: A np array of shape (seed_seq_length, n_joints*dof)
            prediction: A np array of shape (target_seq_length, n_joints*dof)
            target: A np array of shape (target_seq_length, n_joints*dof)
        """
        assert seed.shape[-1] == prediction.shape[-1] == target.shape[-1] == len(SMPL_MAJOR_JOINTS) * 9
        assert prediction.shape[0] == target.shape[0]
        n_joints = len(SMPL_MAJOR_JOINTS)
        dof = 9

        # stitch seed in front of prediction and target
        pred = np.concatenate([seed, prediction], axis=0)
        targ = np.concatenate([seed, target], axis=0)

        # make sure the rotations are valid
        pred_val = np.reshape(pred, [-1, n_joints, 3, 3])
        pred = get_closest_rotmat(pred_val)
        pred = np.reshape(pred, [-1, n_joints * dof])

        # check that the targets are valid
        targ_are_valid = is_valid_rotmat(np.reshape(targ, [-1, n_joints, 3, 3]))
        assert targ_are_valid, 'target rotation matrices are not valid rotations'

        # compute positions
        pred_pos = self.smpl_fk.from_sparse(pred, return_sparse=False)  # (N, SMPL_NR_JOINTS, 3)
        targ_pos = self.smpl_fk.from_sparse(targ, return_sparse=False)  # (N, SMPL_NR_JOINTS, 3)

        # visualize
        # TODO(kamanuel) check if color switch happens at the intended place
        visualize_positions([pred_pos, targ_pos], SMPL_PARENTS, change_color_after_frame=seed.shape[0])


def visualize_positions(positions, parents, out_file=None, keep_frames=False, fps=60.0,
                        change_color_after_frame=None, overlay=False):
    """
    Visualize motion given 3D positions. Can visualize several motions side by side. If the sequence lengths don't
    match, all animations are displayed until the shortest sequence length.
    Args:
        positions: a list of np arrays in shape (seq_length, n_joints, 3) giving the 3D positions per joint and frame.
        parents: skeleton structure
        out_file: output file path if the visualization is saved as video.
        keep_frames:  boolean whether to save video frames or not.
        fps: frames per second
        change_color_after_frame: after this frame id, the color of the plot is changed
        overlay:
    """
    seq_length = np.amin([pos.shape[0] for pos in positions])
    n_joints = positions[0].shape[1]
    pos = positions

    # create figure with as many subplots as we have skeletons
    fig = plt.figure()
    n_axes = 1 if overlay else len(pos)
    axes = [fig.add_subplot(1, n_axes, i + 1, projection='3d') for i in range(n_axes)]

    # create point object for every bone in every skeleton
    all_lines = []
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    for i, joints in enumerate(pos):
        idx = 0 if overlay else i
        ax = axes[idx]
        lines_j = [
            ax.plot(joints[0:1, n,  0], joints[0:1, n, 1], joints[0:1, n, 2], '-o' + colors[i],
                    markersize=3.0)[0] for n in range(1, n_joints)]
        all_lines.append(lines_j)

    # dirty hack to get equal axes behaviour
    min_val = np.amin(pos[0], axis=(0, 1))
    max_val = np.amax(pos[0], axis=(0, 1))
    max_range = (max_val - min_val).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (max_val[0] + min_val[0])
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max_val[1] + min_val[1])
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max_val[2] + min_val[2])

    for ax in axes:
        ax.set_aspect('equal')

        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def on_move(event):
        # find which axis triggered the event
        source_ax = None
        for i in range(len(axes)):
            if event.inaxes == axes[i]:
                source_ax = i
                break

        # transfer rotation and zoom to all other axes
        if source_ax == None:
            return

        for i in range(len(axes)):
            if i != source_ax:
                axes[i].view_init(elev=axes[source_ax].elev, azim=axes[source_ax].azim)
                axes[i].set_xlim3d(axes[source_ax].get_xlim3d())
                axes[i].set_ylim3d(axes[source_ax].get_ylim3d())
                axes[i].set_zlim3d(axes[source_ax].get_zlim3d())
        fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig_text = fig.text(0.05, 0.95, '')

    def update_frame(num, positions, lines, parents, colors):
        for l in range(len(positions)):
            k = 0
            pos = positions[l]
            points_j = lines[l]
            for i in range(1, len(parents)):
                a = pos[num, i]
                b = pos[num, parents[i]]
                p = np.vstack([b, a])
                points_j[k].set_data(p[:, :2].T)
                points_j[k].set_3d_properties(p[:, 2].T)
                if change_color_after_frame and num > change_color_after_frame:
                    points_j[k].set_color(colors[l + 1])
                else:
                    points_j[k].set_color(colors[l])

                k += 1
        time_passed = '{:>.2f} seconds passed'.format(1/60.0*num)
        fig_text.set_text(time_passed)

    # create the animation object, for animation to work reference to this object must be kept
    line_ani = animation.FuncAnimation(fig, update_frame, seq_length,
                                       fargs=(pos, all_lines, parents, colors + [colors[0]]),
                                       interval=int(round(1000.0 / 25.0)), blit=False)

    if out_file is not None:
        save_animation(fig, seq_length, update_frame, [pos, all_lines, parents, colors + [colors[0]]], 0, None, out_file, keep_frames, fps)
    else:
        plt.show()
    plt.close()
