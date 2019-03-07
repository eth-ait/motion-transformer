"""Functions to visualize human poses"""
import data_utils
import numpy as np
import os
import subprocess
import shutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


class Ax3DPose(object):
    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of our representation
        self.I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        self.J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
        # Left / right indicator
        self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        self.ax = ax

        vals = np.zeros((32, 3))

        # Make connection matrix
        self.plots = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
        """
        Update the plotted 3d pose.

        Args
          channels: 96-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """

        assert channels.size == 96, "channels should have 96 entries, it has %d instead"%channels.size
        vals = np.reshape(channels, (32, -1))

        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

        r = 750
        xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])

        self.ax.set_aspect('equal')


def save_animation(fig, seq_length, update_func, update_func_args,
                   start_recording=0, end_recording=None, movie_fname=None, keep_frames=False, fps=25):
    """
    Save animation as frames to disk and may be as movie.
    :param fig: Figure where animation is displayed.
    :param seq_length: Total length of the animation.
    :param start_recording: Frame index where to start recording.
    :param end_recording: Frame index where to stop recording (defaults to `seq_length`, exclusive).
    :param update_func: Update function that is driving the animation.
    :param update_func_args: Arguments for `update_func`
    :param movie_fname: Path and name of the output movie file or `None` if no movie should be procuded.
    :param keep_frames: Whether or not to clear the dumped frames.
    :param fps: Frame rate.
    """
    tmp_path = os.path.join(movie_fname+"_tmp")
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(tmp_path)

    start_frame = start_recording
    end_frame = end_recording or seq_length

    for j in range(start_frame, end_frame):
        update_func(j, *update_func_args)
        fig.savefig(os.path.join(tmp_path, 'frame_{:0>4}.png'.format(j)))

    if movie_fname:
        #print('\nconverting to movie ...')
        # create movie and save it to destination
        if not movie_fname.endswith('.avi'):
            out_file = '{}.avi'.format(movie_fname)
        else:
            out_file = movie_fname

        command = ['ffmpeg',
                   '-start_number', str(start_frame), '-framerate', str(fps),
                   '-loglevel', 'panic',
                   '-i', os.path.join(tmp_path, 'frame_%04d.png'),
                   '-vcodec', 'mpeg4', '-b', '800k', '-y',
                   out_file]
        FNULL = open(os.devnull, 'w')
        subprocess.Popen(command, stdout=FNULL).wait()
        FNULL.close()
        print('saved to {}'.format(out_file))

    if not keep_frames:
        shutil.rmtree(tmp_path, ignore_errors=True)


def visualize_positions(positions, parents, out_file=None, keep_frames=False, fps=60.0,
                        change_color_after_frame=None, overlay=False):
    """
    Visualize motion given 3D positions. Can visualize several motions side by side. If the sequence lengths don't
    match, all animations are displayed until the shortest sequence length.
    :param positions: a list of np arrays in shape (seq_length, n_joints, 3) giving the 3D positions per joint and frame.
    :param change_color_after_frame: after this frame id, the color of the plot is changed
    :param out_file: output file path if the visualization is saved as video.
    :param keep_frames: boolean whether to save video frames or not.
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
        time_passed = '{:>.2f} seconds passed'.format(1./25.*num)
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
