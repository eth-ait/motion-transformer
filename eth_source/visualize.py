import os
import subprocess
import numpy as np
import quaternion
from matplotlib import pyplot as plt, animation as animation
from matplotlib.animation import writers
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from types import SimpleNamespace

from fk import SMPLForwardKinematics
from motion_metrics import get_closest_rotmat
from motion_metrics import is_valid_rotmat
from motion_metrics import aa2rotmat
from motion_metrics import rotmat2aa
from smpl import sparse_to_full
from smpl import SMPL_MAJOR_JOINTS
from smpl import SMPL_NR_JOINTS
from smpl import SMPL_PARENTS

import multiprocessing

try:
    import sys

    sys.path.append('../external/smpl_py3')
    from smpl_webuser.serialization import load_model
except Exception:
    print("SMPL model not available.")

_prop_cycle = plt.rcParams['axes.prop_cycle']
_colors = _prop_cycle.by_key()['color']


class Visualizer(object):
    """
    Helper class to visualize motion. It supports an interactive mode as well as saving frames/videos.
    """

    def __init__(self, interactive, rep="rot_mat", is_sparse=True,
                 fk_engine=None, output_dir=None, skeleton=True, dense=True, to_video=False):
        """
        Initializer. Determines if visualizations are shown interactively or saved to disk.
        Args:
            interactive: Boolean if motion is to be shown in an interactive matplotlib window. If True, requires
              `fk_engine` and can only display skeletons as animating dense meshes is too slow. If False,
              `output_dir` must be passed. In this case, we frames (and optionally a video) are dumped to disk.
              This is slow as it uses SMPL to produce meshes and joint positions for every time instance.
            rep: Representation of the input motions, 'rot_mat', 'quat', or 'aa'.
            is_sparse: If the input motions are sparse, i.e. only using 15 SMPL joints.
            fk_engine: The forward-kinematics engine required for interactive mode.
            output_dir: Where to dump frames/videos in non-interactive mode.
            skeleton: Boolean if skeleton should be shown in non-interactive mode.
            dense: Boolean if mesh should be shown in non-interactive mode.
            to_video: Boolean if a video should be dumped to disk in non-interactive mode.
        """
        self.interactive = interactive
        self.fk_engine = fk_engine
        self.video_dir = output_dir
        self.rep = rep
        self.is_sparse = is_sparse
        self.dense = dense  # also plots the SMPL mesh, WARNING: this is very slow
        self.skeleton = skeleton
        self.to_video = to_video
        self.base_color = _colors[0]  # what color to use to display ground-truth and seed
        self.prediction_color = _colors[1]  # what color to use for predictions, use _colors[2] for non-RNN-SPL models
        assert rep in ["rot_mat", "quat", "aa"]
        if self.interactive:
            assert self.fk_engine
            self.expected_n_input_joints = len(self.fk_engine.major_joints) if is_sparse else self.fk_engine.n_joints
            self.smpl_m = None
        else:
            assert self.skeleton or self.dense, "either skeleton or mesh (or both) should be displayed"
            assert output_dir
            self.expected_n_input_joints = len(SMPL_MAJOR_JOINTS) if is_sparse else SMPL_NR_JOINTS
            self.smpl_m = load_model('../external/smpl_py3/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')

    def visualize(self, seed, prediction, target, title):
        """
        Visualize prediction and ground truth side by side. At the moment only supports sparse pose input in rotation
        matrix or quaternion format.
        Args:
            seed: A np array of shape (seed_seq_length, n_joints*dof)
            prediction: A np array of shape (target_seq_length, n_joints*dof)
            target: A np array of shape (target_seq_length, n_joints*dof)
            title: Title of the plot
        """
        if self.rep == "quat":
            self.visualize_quat(seed, prediction, target, title)
        elif self.rep == "rot_mat":
            self.visualize_rotmat(seed, prediction, target, title)
        else:
            self.visualize_aa(seed, prediction, target, title)

    def visualize_quat(self, seed, prediction, target, title):
        assert seed.shape[-1] == prediction.shape[-1] == target.shape[-1] == self.expected_n_input_joints * 4
        assert prediction.shape[0] == target.shape[0]
        dof = 4

        def _to_rotmat(x):
            b = x.shape[0]
            xq = quaternion.from_float_array(np.reshape(x, [b, -1, dof]))
            xr = quaternion.as_rotation_matrix(xq)
            return np.reshape(xr, [b, -1])

        self.visualize_rotmat(_to_rotmat(seed), _to_rotmat(prediction), _to_rotmat(target), title)

    def visualize_aa(self, seed, prediction, target, title):
        assert seed.shape[-1] == prediction.shape[-1] == target.shape[-1] == self.expected_n_input_joints * 3
        assert prediction.shape[0] == target.shape[0]
        dof = 3

        def _to_rotmat(x):
            b = x.shape[0]
            xaa = aa2rotmat(np.reshape(x, [b, -1, dof]))
            return np.reshape(xaa, [b, -1])

        self.visualize_rotmat(_to_rotmat(seed), _to_rotmat(prediction), _to_rotmat(target), title)

    def visualize_rotmat(self, seed, prediction, target, title):
        assert seed.shape[-1] == prediction.shape[-1] == target.shape[-1] == self.expected_n_input_joints * 9
        assert prediction.shape[0] == target.shape[0]
        n_joints = self.expected_n_input_joints
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

        # check that the targets are valid
        pred_are_valid = is_valid_rotmat(np.reshape(pred, [-1, n_joints, 3, 3]))
        assert pred_are_valid, 'predicted rotation matrices are not valid rotations'

        if self.interactive:
            self._visualize_interactively(pred, targ, title, seed.shape[0])
        else:
            self._visualize_offline(pred, targ, title, seed.shape[0])

    def _visualize_interactively(self, pred, targ, title, change_color_after_frame):
        """
        Visualize predicted and target joint angles in an interactive matplotlib window.
        Args:
            pred: np array of shape (N, n_joints*3*3).
            targ: np array of shape (N, n_joints*3*3).
            title: Window title.
            change_color_after_frame: After which time step to change the color of the prediction.
        """
        # compute positions ourselves as this is a bit faster
        if self.is_sparse:
            pred_pos = self.fk_engine.from_sparse(pred, return_sparse=False)  # (N, full_n_joints, 3)
            targ_pos = self.fk_engine.from_sparse(targ, return_sparse=False)  # (N, full_n_joints, 3)
        else:
            pred_pos = self.fk_engine.from_rotmat(pred)
            targ_pos = self.fk_engine.from_rotmat(targ)

        # swap axes
        pred_pos = pred_pos[..., [0, 2, 1]]
        targ_pos = targ_pos[..., [0, 2, 1]]

        animate(positions=[pred_pos, targ_pos],
                colors=[self.base_color, self.base_color],
                titles=['prediction', 'target'],
                fig_title=title,
                parents=self.fk_engine.parents,
                change_color_after_frame=(change_color_after_frame, None),
                color_after_change=self.prediction_color)

    def _visualize_offline(self, pred, targ, title, change_color_after_frame):
        """
        Dumps every frame to an image on disk. Uses SMPL model directly to compute forward-kinematics.
        Args:
            pred: np array of shape (N, n_joints*3*3).
            targ: np array of shape (N, n_joints*3*3).
            title: Window title.
            change_color_after_frame: After which time step to change the color of the prediction.
        """
        if self.is_sparse:
            pred_full = sparse_to_full(pred, SMPL_MAJOR_JOINTS, SMPL_NR_JOINTS)
            targ_full = sparse_to_full(targ, SMPL_MAJOR_JOINTS, SMPL_NR_JOINTS)
        else:
            pred_full = pred
            targ_full = targ

        # to angle-axis
        pred_full_aa = np.reshape(rotmat2aa(np.reshape(pred_full, [-1, SMPL_NR_JOINTS, 3, 3])),
                                  [-1, SMPL_NR_JOINTS * 3])
        targ_full_aa = np.reshape(rotmat2aa(np.reshape(targ_full, [-1, SMPL_NR_JOINTS, 3, 3])),
                                  [-1, SMPL_NR_JOINTS * 3])

        f_name = title.replace('/', '.')
        f_name = f_name.split('_')[0]  # reduce name otherwise stupid OSes (i.e., all of them) can't handle it
        dir_prefix = 'dense' if self.dense else 'skel'
        dir_prefix = 'dense-skel' if self.dense and self.skeleton else dir_prefix
        out_name = os.path.join(self.video_dir, dir_prefix, f_name)

        animate_offline(joint_angles=[pred_full_aa, targ_full_aa],
                        colors=[self.base_color, self.base_color],
                        titles=['prediction', 'target'],
                        fig_title=title,
                        dense=self.dense,
                        skeleton=self.skeleton,
                        change_color_after_frame=(change_color_after_frame, None),
                        color_after_change=self.prediction_color,
                        out_dir=out_name,
                        to_video=self.to_video)


def animate(positions, colors, titles, fig_title, parents, change_color_after_frame=None,
            color_after_change=None, overlay=False, fps=60):
    """
    Visualize motion given 3D positions. Can visualize several motions side by side. If the sequence lengths don't
    match, all animations are displayed until the shortest sequence length.
    Args:
        positions: a list of np arrays in shape (seq_length, n_joints, 3) giving the 3D positions per joint and frame
        colors: list of color for each entry in `positions`
        titles: list of titles for each entry in `positions`
        fig_title: title for the entire figure
        parents: skeleton structure
        fps: frames per second
        change_color_after_frame: after this frame id, the color of the plot is changed (for each entry in `positions`)
        color_after_change: what color to apply after `change_color_after_frame`
        overlay: if true, all entries in `positions` are plotted into the same subplot
    """
    seq_length = np.amin([pos.shape[0] for pos in positions])
    n_joints = positions[0].shape[1]
    pos = positions

    # create figure with as many subplots as we have skeletons
    fig = plt.figure(figsize=(16, 9))
    plt.clf()
    n_axes = 1 if overlay else len(pos)
    axes = [fig.add_subplot(1, n_axes, i + 1, projection='3d') for i in range(n_axes)]
    fig.suptitle(fig_title)

    # create point object for every bone in every skeleton
    all_lines = []
    # available_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    for i, joints in enumerate(pos):
        idx = 0 if overlay else i
        ax = axes[idx]

        lines_j = [
            ax.plot(joints[0:1, n, 0], joints[0:1, n, 1], joints[0:1, n, 2], '-o',
                    markersize=2.0, color=colors[i])[0] for n in range(1, n_joints)]
        all_lines.append(lines_j)

        ax.set_title(titles[i])

    # dirty hack to get equal axes behaviour
    min_val = np.amin(pos[0], axis=(0, 1))
    max_val = np.amax(pos[0], axis=(0, 1))
    max_range = (max_val - min_val).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (max_val[0] + min_val[0])
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max_val[1] + min_val[1])
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max_val[2] + min_val[2])

    for ax in axes:
        ax.set_aspect('equal')
        ax.axis('off')

        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        ax.view_init(elev=0, azim=-56)

    def on_move(event):
        # find which axis triggered the event
        source_ax = None
        for i in range(len(axes)):
            if event.inaxes == axes[i]:
                source_ax = i
                break

        # transfer rotation and zoom to all other axes
        if source_ax is None:
            return

        for i in range(len(axes)):
            if i != source_ax:
                axes[i].view_init(elev=axes[source_ax].elev, azim=axes[source_ax].azim)
                axes[i].set_xlim3d(axes[source_ax].get_xlim3d())
                axes[i].set_ylim3d(axes[source_ax].get_ylim3d())
                axes[i].set_zlim3d(axes[source_ax].get_zlim3d())
        fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig_text = fig.text(0.05, 0.05, '')

    def update_frame(num, positions, lines):
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
                if change_color_after_frame and change_color_after_frame[l] and num >= change_color_after_frame[l]:
                    points_j[k].set_color(color_after_change)
                else:
                    points_j[k].set_color(colors[l])
                k += 1

        time_passed = '{:>.2f} seconds passed'.format(1 / 60.0 * num)
        fig_text.set_text(time_passed)

    # create the animation object, for animation to work reference to this object must be kept
    fargs = (pos, all_lines)
    line_ani = animation.FuncAnimation(fig, update_frame, seq_length, fargs=fargs, interval=1000 / fps)
    plt.show()
    plt.close()


def save_animation(fig, seq_length, update_func, update_func_args, out_folder, image_format="png",
                   start_recording=0, end_recording=None, create_mp4=False, fps=60):
    """
    Save animation as transparent pngs to disk.
    Args:
        fig: Figure where animation is displayed.
        seq_length: Total length of the animation.
        update_func: Update function that is driving the animation.
        update_func_args: Arguments for `update_func`.
        out_folder: Where to store the frames.
        image_format: In which format to save the frames.
        start_recording: Frame index where to start recording.
        end_recording: Frame index where to stop recording (defaults to `seq_length`, exclusive).
        create_mp4: Convert frames to a movie using ffmpeg.
        fps: Input and output fps.
    """
    if create_mp4:
        assert image_format == "png"
    tmp_path = out_folder
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    start_frame = start_recording
    end_frame = end_recording or seq_length

    for j in range(start_frame, end_frame):
        update_func(j, *update_func_args)
        fig.savefig(os.path.join(tmp_path, 'frame_{:0>4}.{}'.format(j, image_format)), dip=1000)

    if create_mp4:
        save_to_movie(tmp_path, out_folder, start_frame, fps)


def save_to_movie(source_dir, out_dir, start_frame, input_fps, frame_format='frame_%04d.png', output_fps=30):
    """
    Convert single frames stored on disk into an mp4 movie ussing ffmpeg.
    Args:
        source_dir: Where the frames are stored.
        out_dir: Where to store the final video.
        start_frame: At which frame to start the video.
        input_fps: Frequency of the input.
        frame_format: Filename format of the frames, e.g. 'frame_%04d.png'
        output_fps: Desired output frequency.
    """
    counter = 0
    movie_name = os.path.join(out_dir, "vid{}.mp4".format(counter))

    while os.path.exists(movie_name):
        counter += 1
        movie_name = os.path.join(out_dir, "vid{}.mp4".format(counter))

    command = ['ffmpeg',
               '-start_number', str(start_frame),
               '-framerate', str(input_fps),  # must be this early, otherwise it is not applied
               '-r', str(output_fps),  # output is usually 30 fps
               '-loglevel', 'panic',
               '-i', os.path.join(source_dir, frame_format),
               '-c:v', 'libx264',
               '-preset', 'slow',
               '-profile:v', 'high',
               '-level:v', '4.0',
               '-pix_fmt', 'yuv420p',
               '-y',
               movie_name]
    FNULL = open(os.devnull, 'w')
    subprocess.Popen(command, stdout=FNULL).wait()
    FNULL.close()


def animate_offline(joint_angles, colors, titles, fig_title, dense=True, skeleton=True,
                    change_color_after_frame=None, color_after_change=None, overlay=False, out_dir=None, fps=60,
                    to_video=False, n_threads=8):
    """
    Visualize motion given joint angles in the SMPL model. Can visualize several motions side by side.
    If the sequence lengths don't match, all animations are displayed until the shortest sequence length.
    Args:
        joint_angles: a list of np arrays in shape (seq_length, n_joints*3) giving the joint angles in angle-axis
        colors: list of color for each entry in `joint_angles`
        titles: list of titles for each entry in `joint_angles`
        fig_title: title for the entire figure
        dense: boolean if the dense mesh should be displayed
        skeleton: boolean if the joints should be displayed
        change_color_after_frame: after this frame id, the color of the plot is changed (for each entry in
          `joint_angles`)
        color_after_change: what color to apply after `change_color_after_frame`
        out_dir: where to store frames and video
        fps: frames per second of the input sequence
        n_threads: number of threads to parallelize creation of frames
    """
    # Create output dir if necessary.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pool = multiprocessing.Pool()

    # Must split joint_angles to pass them to each worker.
    seq_length = joint_angles[0].shape[0]
    start_idxs = list(range(0, seq_length, seq_length//n_threads))
    if seq_length % n_threads != 0:
        # Make the last worker have some more work
        start_idxs = start_idxs[:-1]
    split_idxs = start_idxs[1:]
    angles_split = []
    for angles in joint_angles:
        a = np.array_split(angles, split_idxs, axis=0)
        angles_split.append(a)

    angles_final = []
    for i in range(n_threads):
        a = []
        for j in range(len(joint_angles)):
            a.append(angles_split[j][i])
        angles_final.append(a)

    remaining_args = {'colors': colors, 'change_color_after_frame': change_color_after_frame,
                      'color_after_change': color_after_change, 'dense': dense, 'skeleton': skeleton,
                      'titles': titles, 'fig_title': fig_title, 'out_dir': out_dir}
    inputs = zip(angles_final, start_idxs, [remaining_args]*n_threads)

    pool.map(_worker, inputs)
    print("All frames created.")

    if to_video:
        print("Saving to video ... ")
        save_to_movie(out_dir, out_dir, 0, fps)


def _worker(args):

    # get list of all joint angle inputs and start index for this worker
    all_angles, start_idx, remaining = args
    r = SimpleNamespace(**remaining)

    n_frames = len(all_angles[0])
    n_seq = len(all_angles)
    n_joints = SMPL_NR_JOINTS
    smpl_m = load_model('../external/smpl_py3/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')

    def to_mesh(v, f, c):
        # flip y and z
        v = v[..., [0, 2, 1]]
        mesh = Poly3DCollection(v[f], alpha=0.2, linewidths=(0.25,))
        face_color = c
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        return mesh

    def get_mesh_and_positions(fr):
        meshes = []
        positions = []
        for i in range(len(all_angles)):
            angles = all_angles[i]

            c = r.colors[i]
            if r.change_color_after_frame:
                if r.change_color_after_frame[i] and start_idx + fr >= r.change_color_after_frame[i]:
                    c = r.color_after_change
            smpl_m.pose[:] = angles[fr]

            if r.dense:
                mesh = to_mesh(smpl_m.r, smpl_m.f, c)
                meshes.append(mesh)

            if r.skeleton:
                pos = smpl_m.J_transformed.r.copy()
                pos = pos[..., [0, 2, 1]]
                positions.append(pos)

        return meshes if r.dense else None, positions if r.skeleton else None

    # create figure with as many subplots as we have skeletons
    fig = plt.figure(figsize=(16, 9))
    plt.clf()
    n_axes = n_seq
    axes = [fig.add_subplot(1, n_axes, i + 1, projection='3d') for i in range(n_axes)]
    fig.suptitle(r.fig_title)

    meshes, positions = get_mesh_and_positions(0)

    # create point object for every bone in every skeleton
    all_lines = []
    # available_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    for i in range(n_seq):
        idx = i
        ax = axes[idx]

        if meshes is not None:
            ax.add_collection3d(meshes[i])

        if positions is not None:
            joints = positions[i]
            lines_j = [
                ax.plot(joints[n:n+1, 0], joints[n:n+1, 1], joints[n:n+1, 2], '-o',
                        markersize=2.0, color=r.colors[i])[0] for n in range(1, n_joints)]
            all_lines.append(lines_j)

        ax.set_title(r.titles[i])

    # dirty hack to get equal axes behaviour
    min_val = np.array([-1.0, -1.0, -1.5])
    max_val = np.array([1.0, 0.5, 0.5])
    max_range = (max_val - min_val).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (max_val[0] + min_val[0])
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max_val[1] + min_val[1])
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max_val[2] + min_val[2])

    for ax in axes:
        ax.set_aspect('equal')
        ax.axis('off')

        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        ax.view_init(elev=0, azim=-56)

    fig_text = fig.text(0.05, 0.05, '')

    def update_frame(positions, lines, parents, colors, meshes, axes, fr):
        frame_id = start_idx + fr
        if meshes is not None:
            for l in range(len(meshes)):
                ax = axes[l]
                ax.collections.remove(ax.collections[0])
                ax.add_collection3d(meshes[l])

        if positions is not None:
            for l in range(len(positions)):
                k = 0
                pos = positions[l]
                points_j = lines[l]
                for i in range(1, len(parents)):
                    a = pos[i]
                    b = pos[parents[i]]
                    p = np.vstack([b, a])
                    points_j[k].set_data(p[:, :2].T)
                    points_j[k].set_3d_properties(p[:, 2].T)
                    c = colors[l]
                    if r.change_color_after_frame:
                        if r.change_color_after_frame[l] and frame_id >= r.change_color_after_frame[l]:
                            c = r.color_after_change
                    points_j[k].set_color(c)

                    k += 1
        time_passed = '{:>.2f} seconds passed'.format(1 / 60.0 * (start_idx + fr))
        fig_text.set_text(time_passed)

    for fr in range(n_frames):
        meshes, positions = get_mesh_and_positions(fr)
        update_frame(positions, all_lines, SMPL_PARENTS, r.colors, meshes, axes, fr)
        fig.savefig(os.path.join(r.out_dir, 'frame_{:0>4}.png'.format(start_idx+fr)), dip=1000)


def visualize_quaternet():
    experiment_id = "1553184554"
    is_longterm = True
    results_folder = "C:\\Users\\manuel\\projects\\motion-modelling\\quaternet_results\\test_results_quaternet_{}{}.npz".format(
        experiment_id,
        "_longterm" if is_longterm else "")
    d = dict(np.load(results_folder))

    selected_idxs = []
    if not is_longterm:
        selected_labels = ["ACCAD/0/Male1General",
                           "ACCAD/0/Male1Running",
                           "ACCAD/0/Male2MartialArtsStances_c3dD12",
                           "ACCAD/3/Male2General",
                           "BioMotion/0/rub0030023",
                           "BioMotion/1/rub0050003",
                           "BioMotion/2/rub0120028",
                           "BioMotion/4/rub0020002",
                           "BioMotion/4/rub0220000",
                           "BioMotion/5/rub0050000"]
    else:
        selected_labels = ["ACCAD/0/Male1Walking_c3dWalk_SB_B14"]

    for s_label in selected_labels:
        counter = 0
        for idx, label in enumerate(d['labels']):
            if label.startswith(s_label):
                counter += 1
                selected_idxs.append(idx)

        assert counter == 1

    fk_engine = SMPLForwardKinematics()
    video_dir = os.path.join("C:\\Users\\manuel\\projects\\motion-modelling\\quaternet_results\\", experiment_id)
    visualizer = Visualizer(fk_engine, video_dir, rep="quat")

    for idx in selected_idxs:
        visualizer.visualize(d['seed'][idx], d['prediction'][idx], d['target'][idx], title=d['labels'][idx])


if __name__ == '__main__':
    visualize_quaternet()
