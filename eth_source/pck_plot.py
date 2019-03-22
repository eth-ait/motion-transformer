import numpy as np
import matplotlib.pyplot as plt


threshs = [2, 5, 10, 15, 20, 30]

####### 10 timesteps (167 ms)
baseline_10 = [0.641023218631744, 0.823953449726105, 0.923106193542481, 0.958253502845764, 0.974659323692322, 0.988700866699219]

seq2seq_sampl_plain_10 = [0.580065786838532, 0.815743625164032, 0.931742787361145, 0.967593491077423, 0.981954574584961, 0.991881966590881]
seq2seq_sampl_spl_10 = [0.670647621154785, 0.857880234718323, 0.94590961933136, 0.972898960113525, 0.98384028673172, 0.991802394390106]

rnn_plain_10 = [0.641044616699219, 0.823951900005341, 0.923090934753418, 0.958236575126648, 0.974659323692322, 0.988700866699219]
rnn_spl_10 = [0.749398171901703, 0.913310408592224, 0.973004639148712, 0.987706661224365, 0.992871582508087, 0.996148943901062]

seq2seq_sup_plain_10 = [0.551673829555512, 0.782731413841248, 0.915994465351105, 0.960582613945007, 0.978377342224121, 0.991140186786652]
seq2seq_sup_spl_10 = [0.730941295623779, 0.894826054573059, 0.963730156421661, 0.982736110687256, 0.989960968494415, 0.995017230510712]

seq2seq_sup_drop_plain_10 = [0.542454659938812, 0.778429329395294, 0.916311085224152, 0.961280167102814, 0.979178726673126, 0.991681575775147]
seq2seq_sup_drop_spl_10 = [0.738443076610565, 0.901194393634796, 0.966484665870667, 0.983872413635254, 0.99044132232666, 0.994904041290283]

quaternet_plain_10 = [0.701463937759399, 0.8839031457901, 0.960431098937988, 0.981192588806152, 0.989211440086365, 0.99468719959259]
quaternet_spl_10 = [0.713340878486633, 0.89456570148468, 0.96587073802948, 0.984222114086151, 0.990750908851624, 0.99517810344696]


####### 24 timesteps (400 ms)
baseline_24 = [0.511265993118286, 0.689975440502167, 0.826081991195679, 0.888994693756104, 0.923690378665924, 0.958943843841553]

seq2seq_sampl_plain_24 = [0.421194314956665, 0.646687567234039, 0.823334753513336, 0.899748265743256, 0.937800407409668, 0.971210956573486]
seq2seq_sampl_spl_24 = [0.534137547016144, 0.729571640491486, 0.86307954788208, 0.919420778751373, 0.948152542114258, 0.974073946475983]

rnn_plain_24 = [0.511211216449738, 0.690009117126465, 0.826096057891846, 0.888981342315674, 0.92367821931839, 0.958943843841553]
rnn_spl_24 = [0.582135379314423, 0.776890099048615, 0.897408306598663, 0.943779766559601, 0.965346872806549, 0.983696520328522]

seq2seq_sup_plain_24 = [0.381357163190842, 0.566383123397827, 0.736491501331329, 0.831326544284821, 0.886641919612885, 0.943602025508881]
seq2seq_sup_spl_24 = [0.545399010181427, 0.724721491336823, 0.850888907909393, 0.907362759113312, 0.937463939189911, 0.967289924621582]

seq2seq_sup_drop_plain_24 = [0.379742383956909, 0.56725549697876, 0.743692457675934, 0.838185966014862, 0.893201768398285, 0.948339283466339]
seq2seq_sup_drop_spl_24 = [0.556157171726227, 0.73935604095459, 0.8657585978508, 0.920761585235596, 0.949163913726807, 0.974808216094971]

quaternet_plain_24 = [0.536811530590057, 0.737488031387329, 0.871729135513306, 0.926731586456299, 0.95427942276001, 0.977985858917236]
quaternet_spl_24 = [0.550190269947052, 0.757963716983795, 0.888717472553253, 0.938954174518585, 0.962609946727753, 0.98206490278244]


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

rnn_spl_color = colors[1]
base_color = colors[0]
zero_vel_color = 'k'

to_visualize = [[(baseline_24, "Zero-Velocity", '-', zero_vel_color),
                 (seq2seq_sampl_plain_24, "Seq2seq-sampling", '-', base_color),
                 (seq2seq_sampl_spl_24, "Seq2seq-sampling-SPL", '--', base_color),
                 (rnn_spl_24, "RNN-SPL", '--', rnn_spl_color)],
                [(baseline_24, "Zero-Velocity", '-', zero_vel_color),
                 (seq2seq_sup_plain_24, "Seq2seq", '-', base_color),
                 (seq2seq_sup_spl_24, "Seq2seq-SPL", '--', base_color),
                 (rnn_spl_24, "RNN-SPL", '--', rnn_spl_color)],
                [(baseline_24, "Zero-Velocity", '-', zero_vel_color),
                 (seq2seq_sup_drop_plain_24, "Seq2seq-dropout", '-', base_color),
                 (seq2seq_sup_drop_spl_24, "Seq2seq-dropout-SPL", '--', base_color),
                 (rnn_spl_24, "RNN-SPL", '--', rnn_spl_color)],
                [(baseline_24, "Zero-Velocity", '-', zero_vel_color),
                 (quaternet_plain_24, "QuaterNet", '-', base_color),
                 (quaternet_spl_24, "QuaterNet-SPL", '--', base_color),
                 (rnn_spl_24, "RNN-SPL", '--', rnn_spl_color)]
                ]

titles = ['Seq2seq-sampling', 'Seq2seq', 'Seq2seq-dropout', 'QuaterNet']


fig, axes = plt.subplots(2, 2, sharey=True)
legend_fontsize = 14
axes_fontsize = 12

for p in range(2):
    for k in range(2):
        ax = axes[p, k]
        idx = k*2 + p
        pcks = to_visualize[idx]

        for pck, label, linestyle, color in pcks:
            ax.plot(threshs, np.array(pck)*100, linestyle, label=label, color=color)

        if p == 1:
            ax.set_xlabel('threshold', fontsize=axes_fontsize)
        if k == 0:
            ax.set_ylabel('% of correct keypoints', fontsize=axes_fontsize)
        ax.legend(fontsize=legend_fontsize, loc='lower right')
        ax.grid(alpha=0.5)
        ax.set_title(titles[idx], fontsize=legend_fontsize)

# fig.suptitle("PCK Curves for 400 ms prediction horizon")

plt.show()


