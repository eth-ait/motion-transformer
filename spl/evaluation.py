"""
SPL: training and evaluation of neural networks with a structured prediction layer.
Copyright (C) 2019 ETH Zurich, Emre Aksan, Manuel Kaufmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import glob
import json
import argparse

import numpy as np
import tensorflow as tf

from spl.data.amass_tf import TFRecordMotionDataset
from spl.model.zero_velocity import ZeroVelocityBaseline
from spl.model.rnn import RNN
from spl.model.seq2seq import Seq2SeqModel
from spl.model.buggy_transformer import Transformer2d

from common.constants import Constants as C
from visualization.render import Visualizer
from visualization.fk import H36MForwardKinematics
from visualization.fk import SMPLForwardKinematics
from metrics.motion_metrics import MetricsEngine

try:
    from common.logger import GoogleSheetLogger

    if "GLOGGER_WORKBOOK_AMASS" not in os.environ:
        raise ImportError("GLOGGER_WORKBOOK_AMASS not found.")
    if "GDRIVE_API_KEY" not in os.environ:
        raise ImportError("GDRIVE_API_KEY not found.")
    GLOGGER_AVAILABLE = True
except ImportError:
    GLOGGER_AVAILABLE = False
    print("GLogger not available...")


def load_latest_checkpoint(session, saver, experiment_dir):
    """Restore the latest checkpoint found in `experiment_dir`."""
    ckpt = tf.train.get_checkpoint_state(experiment_dir, latest_filename="checkpoint")

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Loading model checkpoint {0}".format(ckpt_name))
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        raise (ValueError, "Checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path))


def get_model_cls(model_type):
    if model_type == C.MODEL_ZERO_VEL:
        return ZeroVelocityBaseline
    elif model_type == C.MODEL_RNN:
        return RNN
    elif model_type == C.MODEL_SEQ2SEQ:
        return Seq2SeqModel
    elif model_type == C.MODEL_TRANS2D:
        return Transformer2d
    else:
        raise Exception("Unknown model type.")


def create_and_restore_model(session, experiment_dir, data_dir, config, dynamic_test_split):
    model_cls = get_model_cls(config["model_type"])
    window_length = config["source_seq_len"] + config["target_seq_len"]

    if config["use_h36m"]:
        data_dir = os.path.join(data_dir, '../../h3.6m/tfrecords/')

    if dynamic_test_split:
        data_split = "test_dynamic"
    else:
        assert window_length <= 180, "TFRecords are hardcoded with length of 180."
        data_split = "test"

    test_data_path = os.path.join(data_dir, config["data_type"], data_split, "amass-?????-of-?????")
    meta_data_path = os.path.join(data_dir, config["data_type"], "training", "stats.npz")
    print("Loading test data from " + test_data_path)

    # Create dataset.
    with tf.name_scope("test_data"):
        test_data = TFRecordMotionDataset(data_path=test_data_path,
                                          meta_data_path=meta_data_path,
                                          batch_size=32,
                                          shuffle=False,
                                          extract_windows_of=window_length,
                                          window_type=C.DATA_WINDOW_BEGINNING,
                                          num_parallel_calls=4,
                                          normalize=not config["no_normalization"])
        test_pl = test_data.get_tf_samples()

    # Create model.
    with tf.name_scope(C.TEST):
        test_model = model_cls(
            config=config,
            data_pl=test_pl,
            mode=C.SAMPLE,
            reuse=False)
        test_model.build_graph()
        test_model.summary_routines()

    num_param = 0
    for v in tf.trainable_variables():
        num_param += np.prod(v.shape.as_list())
    print("# of parameters: " + str(num_param))

    # Restore model parameters.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1, save_relative_paths=True)
    load_latest_checkpoint(session, saver, experiment_dir)
    return test_model, test_data


def evaluate_model(session, _eval_model, _eval_iter, _metrics_engine,
                   undo_normalization_fn, _return_results=False):
    # make a full pass on the validation or test dataset and compute the metrics
    n_batches = 0
    _eval_result = dict()
    _metrics_engine.reset()
    session.run(_eval_iter.initializer)
    try:
        while True:
            # Get the predictions and ground truth values
            res = _eval_model.sampled_step(session)
            prediction, targets, seed_sequence, data_id = res
            # Unnormalize predictions if there normalization applied.
            p = undo_normalization_fn(
                {"poses": prediction}, "poses")
            t = undo_normalization_fn(
                {"poses": targets}, "poses")
            _metrics_engine.compute_and_aggregate(p["poses"], t["poses"])

            if _return_results:
                s = undo_normalization_fn(
                    {"poses": seed_sequence}, "poses")
                # Store each test sample and corresponding predictions with
                # the unique sample IDs.
                for k in range(prediction.shape[0]):
                    _eval_result[data_id[k].decode("utf-8")] = (
                        p["poses"][k],
                        t["poses"][k],
                        s["poses"][k])
            n_batches += 1
            if n_batches == 5:
                break
    except tf.errors.OutOfRangeError:
        pass
    # finalize the computation of the metrics
    final_metrics = _metrics_engine.get_final_metrics()
    return final_metrics, _eval_result


def evaluate(session, test_model, test_data, args, eval_dir, use_h36m):
    test_iter = test_data.get_iterator()

    # Create metrics engine including summaries
    pck_thresholds = C.METRIC_PCK_THRESHS
    if use_h36m:
        fk_engine = H36MForwardKinematics()
        target_lengths = [x for x in C.METRIC_TARGET_LENGTHS_H36M if x <= test_model.target_seq_len]
    else:
        fk_engine = SMPLForwardKinematics()
        target_lengths = [x for x in C.METRIC_TARGET_LENGTHS_AMASS if x <= test_model.target_seq_len]

    representation = C.QUATERNION if test_model.use_quat else C.ANGLE_AXIS if test_model.use_aa else C.ROT_MATRIX
    metrics_engine = MetricsEngine(fk_engine,
                                   target_lengths,
                                   force_valid_rot=True,
                                   pck_threshs=pck_thresholds,
                                   rep=representation)
    # create the necessary summary placeholders and ops
    metrics_engine.create_summaries()
    # reset computation of metrics
    metrics_engine.reset()

    # Google logging.
    if args.glog_entry and GLOGGER_AVAILABLE:
        exp_id = os.path.split(eval_dir)[-1].split("-")[0] + ".test"
        glogger_workbook = os.environ["GLOGGER_WORKBOOK_AMASS"]
        gdrive_key = os.environ["GDRIVE_API_KEY"]
        model_name = '-'.join(os.path.split(eval_dir)[-1].split('-')[1:])
        static_values = dict()
        static_values["Model ID"] = exp_id
        static_values["Model Name"] = model_name

        credentials = tf.gfile.Open(gdrive_key, "r")
        glogger = GoogleSheetLogger(
            credentials,
            glogger_workbook,
            sheet_names=["until_{}".format(24)],
            model_identifier=exp_id,
            static_values=static_values)

    print("Evaluating test set...")
    undo_norm_fn = test_data.unnormalize_zero_mean_unit_variance_channel
    test_metrics, eval_result = evaluate_model(session, test_model,
                                               test_iter,
                                               metrics_engine,
                                               undo_norm_fn,
                                               _return_results=True)

    print(metrics_engine.get_summary_string_all(test_metrics, target_lengths,
                                                pck_thresholds))

    # If there is a new checkpoint, log the result.
    if args.glog_entry and GLOGGER_AVAILABLE:
        for t in metrics_engine.target_lengths:
            eval_ = metrics_engine.get_metrics_until(test_metrics,
                                                     t,
                                                     pck_thresholds,
                                                     prefix="test ")
            glogger.update_or_append_row(eval_,
                                         "until_{}".format(t))

    if args.visualize:
        data_representation = "quat" if test_model.use_quat else "aa" if test_model.use_aa else "rotmat"
        # visualize some random samples stored in `eval_result` which is a
        # dict id -> (prediction, seed, target)
        if not args.to_video:
            visualizer = Visualizer(interactive=True, fk_engine=fk_engine,
                                    rep=data_representation)
        else:
            visualizer = Visualizer(interactive=False, fk_engine=fk_engine,
                                    rep=data_representation,
                                    output_dir=eval_dir,
                                    skeleton=not args.no_skel,
                                    dense=not args.no_mesh,
                                    to_video=args.to_video)

        n_samples_viz = 30
        # Get random indices or just all of them.
        rng = np.random.RandomState(4313)
        idxs = rng.randint(0, len(eval_result), size=n_samples_viz)
        # idxs = list(range(n_samples_viz))

        # Select some indices for faster visualization or just all of them.
        # selected_idxs = [4]  # [12, 13, 14, 27, 29]  # [5, 6, 7, 19]  # [0, 1, 2, 5, 6, 7, 9, 19, 24, 27]
        # sample_keys = [list(sorted(eval_result.keys()))[i] for ii, i in enumerate(idxs) if ii in selected_idxs]
        sample_keys = [list(sorted(eval_result.keys()))[i] for i in idxs]

        # Find an entry by name
        ''' 
        #sample_keys = ['CMU/0/CMU/120_120_18']
        interesting_keys = ['CMU/0/CMU/106_106_34',
                            'BioMotion/0/BioMotion/rub0220001_treadmill_fast_dynamics',
                            'Transition/0/Transition/mazen_c3dairkick_walkbackwards',
                            'CMU/0/CMU/01_01_06']
        sample_keys = ["CMU/0/CMU/120_120_18",
                       "ACCAD/0/ACCAD/Male1Running_c3dRun_SB_C27_SB__SB2__SB_crouch_SB_to_SB_run_dynamics",
                       "CMU/26/CMU/86_86_03"]
        '''
        # idxs = [i for i in range(64)]
        # sample_keys = [list(sorted(eval_result.keys()))[i] for i in idxs]

        sample_keys = ["ACCAD/0/ACCAD/Male1Walking_c3dWalk_SB_B14_SB__SB2__SB_Walk_SB_turn_SB_right_SB_135_dynamics",
                       "ACCAD/0/ACCAD/Male2MartialArtsStances_c3dD10_SB__SB2__SB_victory_SB_2_dynamics",
                       "BioMotion/0/BioMotion/rub0010000_treadmill_norm_dynamics",
                       "BioMotion/0/BioMotion/rub0080021_catching_and_throwing_dynamics",
                       "BioMotion/0/BioMotion/rub0320027_circle_walk_dynamics"]
        
        print("Visualizing samples...")
        for i, k in enumerate(sample_keys):
            prediction, target, seed = eval_result[k]
            len_diff = prediction.shape[0] - target.shape[0]
            if len_diff > 0:
                target = np.concatenate([target, np.tile(target[-1:], (len_diff, 1))], axis=0)
            visualizer.visualize_results(seed, prediction, target,
                                         title=k + "_i{}".format(i))


if __name__ == '__main__':
    # If you would like to quantitatively evaluate a model, then
    # --dynamic_test_split shouldn't be passed. In this case, the model will be
    # evaluated on 180 frame windows extracted from the entire test split.
    # You can still visualize samples. However, number of predicted frames
    # will be less than or equal to 60. If you intend to evaluate/visualize
    # longer predictions, then you should pass --dynamic_test_split which
    # enables using original full-length test sequences. Hence,
    # --seq_length_out can be much longer than 60.

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', required=True, default=None, type=str,
                        help="Experiment ID (experiment timestamp) or "
                             "comma-separated list of ids.")
    parser.add_argument('--eval_dir', required=False, default=None, type=str,
                        help="Main visualization directory. First, a folder "
                             "with the experiment name is created inside. "
                             "If not passed, then save_dir is used.")
    parser.add_argument('--save_dir', required=False, default=None, type=str,
                        help="Path to experiments. If not passed, "
                             "then AMASS_EXPERIMENTS environment variable is "
                             "used.")
    parser.add_argument('--data_dir', required=False, default=None, type=str,
                        help="Path to data. If not passed, "
                             "then AMASS_DATA environment variable is used.")

    parser.add_argument('--seq_length_in', required=False, type=int,
                        help="Seed sequence length")
    parser.add_argument('--seq_length_out', required=False, type=int,
                        help="Target sequence length")
    parser.add_argument('--batch_size', required=False, default=64, type=int,
                        help="Batch size")

    parser.add_argument('--visualize', required=False, action="store_true",
                        help="Visualize ground-truth and predictions "
                             "side-by-side by using human skeleton.")
    parser.add_argument('--no_skel', required=False, action="store_true",
                        help="Dont show skeleton in offline visualization.")
    parser.add_argument('--no_mesh', required=False, action="store_true",
                        help="Dont show mesh in offline visualization")
    parser.add_argument('--to_video', required=False, action="store_true",
                        help="Save the model predictions to mp4 videos in the "
                             "experiments folder.")
    parser.add_argument('--dynamic_test_split', required=False,
                        action="store_true",
                        help="Test samples are extracted on-the-fly.")
    parser.add_argument('--glog_entry', required=False,
                        action="store_true",
                        help="Create a Google sheet entry if available.")

    _args = parser.parse_args()
    if ',' in _args.model_id:
        model_ids = _args.model_id.split(',')
    else:
        model_ids = [_args.model_id]

    # Set experiment directory.
    _save_dir = _args.save_dir if _args.save_dir else os.environ["AMASS_EXPERIMENTS"]
    # Set data paths.
    _data_dir = _args.data_dir if _args.data_dir else os.environ["AMASS_DATA"]

    # Run evaluation for each model id.
    for model_id in model_ids:
        try:
            _experiment_dir = glob.glob(os.path.join(_save_dir, model_id + "-*"), recursive=False)[0]
        except IndexError:
            print("Model " + str(model_id) + " is not found in " + str(_save_dir))
            continue

        try:
            tf.reset_default_graph()
            _config = json.load(open(os.path.abspath(os.path.join(_experiment_dir, 'config.json')), 'r'))
            _config["experiment_dir"] = _experiment_dir

            if _args.seq_length_out is not None and _config["target_seq_len"] != _args.seq_length_out:
                print("!!! Prediction length for training and sampling is different !!!")
                _config["target_seq_len"] = _args.seq_length_out

            if _args.seq_length_in is not None and _config["source_seq_len"] != _args.seq_length_in:
                print("!!! Seed sequence length for training and sampling is different !!!")
                _config["source_seq_len"] = _args.seq_length_in

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                exp_name = os.path.split(_experiment_dir)[-1]
                _eval_dir = _experiment_dir if _args.eval_dir is None else os.path.join(_args.eval_dir, exp_name)
                if not os.path.exists(_eval_dir):
                    os.mkdir(_eval_dir)
                _test_model, _test_data = create_and_restore_model(sess, _experiment_dir, _data_dir, _config,
                                                                   _args.dynamic_test_split)
                print("Evaluating Model " + str(model_id))
                evaluate(sess, _test_model, _test_data, _args, _eval_dir, _config["use_h36m"])

        except Exception as e:
            print("Something went wrong when evaluating model {}".format(model_id))
            raise Exception(e)
