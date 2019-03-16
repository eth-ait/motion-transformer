import os
import glob
import json
import argparse

import numpy as np
import tensorflow as tf

import amass_models as models
from amass_tf_data import TFRecordMotionDataset
from logger import GoogleSheetLogger
from constants import Constants as C
from motion_metrics import MetricsEngine
from visualize import Visualizer


def create_and_restore_model(session, experiment_dir, config, args):
    # Create dataset.
    windows_length = args.seq_length_in + args.seq_length_out
    rep = "quat" if config.get('use_quat', False) else "aa" if config.get('use_aa') else "rotmat"

    data_path = os.environ["AMASS_DATA"]
    if config.get('use_h36m_only', False):
        data_path = os.path.join(data_path, '../per_db/h36m')

    if args.dynamic_test_split:
        config['target_seq_len'] = args.seq_length_out
        test_data_path = os.path.join(data_path, rep, "test_dynamic", "amass-?????-of-?????")
    else:
        test_data_path = os.path.join(data_path, rep, "test", "amass-?????-of-?????")
        assert windows_length == 160, "TFRecords are hardcoded with length of 160."
        windows_length = 0  # set to 0 so that dataset class works as intended

    meta_data_path = os.path.join(data_path, rep, "training", "stats.npz")

    data_normalization = not (args.no_normalization or config.get("no_normalization", False))
    with tf.name_scope("test_data"):
        test_data = TFRecordMotionDataset(data_path=test_data_path,
                                          meta_data_path=meta_data_path,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          extract_windows_of=windows_length,
                                          num_parallel_calls=16,
                                          normalize=data_normalization)
        test_pl = test_data.get_tf_samples()

    if config['model_type'] == "seq2seq":
        model_cls = models.Seq2SeqModel
    elif config['model_type'] == "simple_baseline":
        model_cls = models.ASimpleYetEffectiveBaseline
    elif config['model_type'] == "stcn":
        model_cls = models.STCN
    elif config['model_type'] == "wavenet":
        model_cls = models.Wavenet
    elif config['model_type'] == "seq2seq_feedback":
        raise NotImplementedError()
    elif config['model_type'] == "structured_stcn":
        raise NotImplementedError()
    elif config['model_type'] == "rnn":
        model_cls = models.RNN
    elif config['model_type'] == "vrnn":
        model_cls = models.VRNN
    else:
        raise Exception("Unknown model type.")

    # Create model.
    with tf.name_scope(C.TEST):
        test_model = model_cls(
            config=config,
            data_pl=test_pl,
            mode=C.SAMPLE,
            reuse=False,
            dtype=tf.float32)
        test_model.build_graph()
        test_model.summary_routines()

    num_param = 0
    for v in tf.trainable_variables():
        num_param += np.prod(v.shape.as_list())
    print("# of parameters: " + str(num_param))
    # assert config["num_parameters"] == num_param, "# of parameters doesn't match."

    # Restore model parameters.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1, save_relative_paths=True)

    """Restore the latest checkpoint found in `experiment_dir`."""
    ckpt = tf.train.get_checkpoint_state(experiment_dir, latest_filename="checkpoint")

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Loading model checkpoint {0}".format(ckpt_name))
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        raise ValueError("could not load checkpoint")

    return test_model, test_data


def evaluate(experiment_dir, config, args):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        test_model, test_data = create_and_restore_model(sess, experiment_dir, config, args)
        test_iter = test_data.get_iterator()

        # Create metrics engine including summaries
        # in milliseconds: 83.3, 166.7, 316.7, 400, 566.7, 1000]
        target_lengths = [x for x in C.METRIC_TARGET_LENGTHS if x <= test_model.target_seq_len]
        pck_threshs = C.METRIC_PCK_THRESHS
        metrics_engine = MetricsEngine("../external/smpl_py3/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl",
                                       target_lengths,
                                       force_valid_rot=True,
                                       pck_threshs=pck_threshs,
                                       rep="quat" if test_model.use_quat else "aa" if test_model.use_aa else "rot_mat")
        # create the necessary summary placeholders and ops
        metrics_engine.create_summaries()
        # reset computation of metrics
        metrics_engine.reset()

        # create logger
        if args.glog_entry:
            g_logger = GoogleSheetLogger(credential_file=C.LOGGER_MANU,
                                         workbook_name="motion_modelling_experiments")
            glog_data = {'Model ID'  : [os.path.split(experiment_dir)[-1].split('-')[0]],
                         'Model Name': ['-'.join(os.path.split(experiment_dir)[-1].split('-')[1:])],
                         'Comment'   : [""]}

        def evaluate_model(_eval_model, _eval_iter, _metrics_engine):
            # make a full pass on the validation or test dataset and compute the metrics
            eval_result = dict()
            _metrics_engine.reset()
            sess.run(_eval_iter.initializer)
            try:
                while True:
                    # TODO(kamanuel) should we compute the validation loss here as well, if so how?
                    # get the predictions and ground truth values
                    prediction, targets, seed_sequence, data_id = _eval_model.sampled_step(sess)
                    # unnormalize - if normalization is not configured, these calls do nothing
                    p = test_data.unnormalize_zero_mean_unit_variance_channel({"poses": prediction}, "poses")
                    t = test_data.unnormalize_zero_mean_unit_variance_channel({"poses": targets}, "poses")
                    s = test_data.unnormalize_zero_mean_unit_variance_channel({"poses": seed_sequence}, "poses")
                    _metrics_engine.compute_and_aggregate(p["poses"], t["poses"])

                    # Store each test sample and corresponding predictions with the unique sample IDs.
                    for i in range(prediction.shape[0]):
                        eval_result[data_id[i].decode("utf-8")] = (p["poses"][i], t["poses"][i], s["poses"][i])
            except tf.errors.OutOfRangeError:
                pass
            finally:
                # finalize the computation of the metrics
                final_metrics = _metrics_engine.get_final_metrics()
            return final_metrics, eval_result

        print("Evaluating test set ...")
        test_metrics, eval_result = evaluate_model(test_model, test_iter, metrics_engine)
        print("Test \t {}".format(metrics_engine.get_summary_string(test_metrics)))

        # gather the metrics
        if args.glog_entry:
            for t in metrics_engine.target_lengths:
                glog_test_metrics = metrics_engine.get_summary_glogger(test_metrics, is_validation=False, until=t)
                glog_data["Comment"] = ["until_{}".format(t)]
                glog_data = {**glog_data, **glog_test_metrics}
                g_logger.append_row(glog_data, sheet_name="until_{}".format(t))

        if args.visualize:
            # visualize some random samples stored in `eval_result` which is a dict id -> (prediction, seed, target)
            video_dir = experiment_dir if args.visualize_save else None
            visualizer = Visualizer("../external/smpl_py3/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl", video_dir,
                                    rep="quat" if test_model.use_quat else "aa" if test_model.use_aa else "rot_mat")
            n_samples_viz = 20
            rng = np.random.RandomState(42)
            idxs = rng.randint(0, len(eval_result), size=n_samples_viz)
            sample_keys = [list(sorted(eval_result.keys()))[i] for i in idxs]
            for k in sample_keys:
                visualizer.visualize(eval_result[k][2], eval_result[k][0], eval_result[k][1], title=k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', required=False, default=os.path.normpath("../experiments_amass/"), type=str, help='Model save directory.')
    parser.add_argument('--model_id', required=True, default=None, type=str, help='Experiment ID (experiment timestamp).')
    parser.add_argument('--seq_length_in', required=False, default=100, type=int, help='Seed sequence length')
    parser.add_argument('--seq_length_out', required=False, default=60, type=int, help='Target sequence length')
    parser.add_argument('--batch_size', required=False, default=64, type=int, help='Batch size')
    parser.add_argument('--no_normalization', required=False, action="store_true", help='If set, do not use zero-mean unit-variance normalization.')
    parser.add_argument('--glog_entry', required=False, action="store_true", help='Write to the Google sheet.')
    parser.add_argument('--visualize', required=False, action="store_true", help='Visualize model predictions.')
    parser.add_argument('--visualize_save', required=False, action="store_true", help='Save the model predictions to mp4 videos in the experiments folder.')
    parser.add_argument('--dynamic_test_split', required=False, action="store_true", help="Test samples are extracted on-the-fly.")

    args = parser.parse_args()
    try:
        experiment_dir = glob.glob(os.path.join(args.save_dir, args.model_id + "-*"), recursive=False)[0]
    except IndexError:
        raise Exception("Model " + str(args.model_id) + " is not found in " + str(args.save_dir))

    config = json.load(open(os.path.abspath(os.path.join(experiment_dir, 'config.json')), 'r'))
    evaluate(experiment_dir, config, args)

