"""
A script to analyze whether the model is copying seed frames to form its predictions.
"""
import numpy as np
import matplotlib.pyplot as plt

from visualization.render import Visualizer
from visualization.fk import SMPLForwardKinematics


def get_min_dists_to_seed(preds, seeds):
    min_idxs = []
    min_vals = []
    seed_len = seeds.shape[0]
    for i in range(preds.shape[0]):
        # Find the closest frame in the seed.
        pred = preds[i:i + 1].repeat(seed_len, axis=0)
        dist = np.linalg.norm(pred - seeds, axis=1)
        min_idx = np.argmin(dist)
        min_idxs.append(min_idx)
        min_vals.append(dist[min_idx])
    return min_idxs, min_vals


def analyze_single_sequence(sequences, seq_key):
    seq = sequences[seq_key]
    preds, targs, seeds = seq[0], seq[1], seq[2]

    # Create "fake" predictions by repeating the seed sequence.
    seed_len = seeds.shape[0]
    pred_len = preds.shape[0]
    n_copies = pred_len // seed_len + 1
    fake_preds = []
    for i in range(n_copies):
        if i % 2 == 0:
            # Take the reverse of the seed sequence
            fake_preds.append(seeds[::-1].copy())
        else:
            fake_preds.append(seeds)
    fake_preds = np.concatenate(fake_preds, axis=0)[:pred_len]

    # Visualize for debugging.
    fk_engine = SMPLForwardKinematics()
    visualizer = Visualizer(interactive=True, fk_engine=fk_engine,
                            rep="rotmat")
    visualizer.visualize_results(seeds, preds, targs,
                                 title=seq_key)

    _, min_vals_pred = get_min_dists_to_seed(preds, seeds)
    _, min_vals_gt = get_min_dists_to_seed(targs, seeds)
    _, min_vals_fake = get_min_dists_to_seed(fake_preds, seeds)

    plt.figure()
    plt.plot(min_vals_pred, label='Predictions')
    plt.plot(min_vals_gt, label='Ground-Truth')
    plt.plot(min_vals_fake, label='Zero-Velocity')
    plt.grid()
    plt.legend()
    plt.show()


def analyze_all_sequences(sequences):
    min_vals_pred_all = []
    min_vals_gt_all = []
    for k in sequences:
        seq = sequences[k]
        preds, targs, seeds = seq[0], seq[1], seq[2]
        if targs.shape[0] < 600:
            continue
        _, min_vals_pred = get_min_dists_to_seed(preds, seeds)
        _, min_vals_gt = get_min_dists_to_seed(targs, seeds)
        min_vals_pred_all.append(np.array(min_vals_pred))
        min_vals_gt_all.append(np.array(min_vals_gt))

    min_vals_pred_all = np.row_stack(min_vals_pred_all)
    min_vals_gt_all = np.row_stack(min_vals_gt_all)
    pred_mean = np.mean(min_vals_pred_all, axis=0)
    pred_std = np.std(min_vals_pred_all, axis=0)
    pred_confidence = 1.96 * pred_std / np.sqrt(min_vals_pred_all.shape[0])
    gt_mean = np.mean(min_vals_gt_all, axis=0)
    gt_std = np.mean(min_vals_gt_all, axis=0)
    gt_confidence = 1.96 * gt_std / np.sqrt(min_vals_gt_all.shape[0])

    x = list(range(600))
    fig, ax = plt.subplots()
    ax.plot(x, pred_mean, label='Predictions')
    ax.fill_between(x, (pred_mean - pred_confidence), (pred_mean + pred_confidence), color='C0', alpha=.2)

    ax.plot(x, gt_mean, label='Ground-Truth')
    ax.fill_between(x, (gt_mean - gt_confidence), (gt_mean + gt_confidence), color='C1', alpha=.2)
    ax.grid()
    ax.legend(loc='upper left')
    ax.set_xlabel('Time (Frame Number)')
    ax.set_ylabel('Minimum L2 Distance to Seed')
    plt.show()


def main():
    # Load pre-recorded sequences.
    sequences = np.load("C:\\Users\\manuel\\projects\\motion-modelling\\experiments_amass\\1573450146-"
                        "transformer2d-plain-amass_rotmat-b32-in120_out24-t8-s8-l8-dm128-df256-w120-1237"
                        "\\eval_samples_preds_periodic.npy", allow_pickle=True).tolist()

    # Select a jog sequence.
    seq_key = 'BioMotion/0/BioMotion/rub0640003_treadmill_jog_dynamics'
    # analyze_single_sequence(sequences, seq_key)
    analyze_all_sequences(sequences)


if __name__ == '__main__':
    main()
