"""Functions to visualize human poses"""
import numpy as np
import os
import subprocess
import shutil


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
