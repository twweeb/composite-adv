import numpy as np
import matplotlib.pyplot as plt
from math import pi


class Radar(object):

    def __init__(self, fig, titles, labels, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.95, 0.95]

        self.n = len(titles)
        self.angles = [a if a <= 360. else a - 360. for a in np.arange(90, 90 + 360, 360.0 / self.n)]
        self.axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i)
                     for i in range(self.n)]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=titles, fontsize=12, weight="bold", color="black")

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
            self.ax.yaxis.grid(False)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.set_rgrids(range(1, 7), labels=label, angle=angle, fontsize=12)
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(0, 6)
            ax.xaxis.grid(True, color='black', linestyle='-')
            pos = ax.get_rlabel_position()
            ax.set_rlabel_position(pos + 7)

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)

    def fill(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.fill(angle, values, *args, **kw)


def draw_radar_chart(values, legend, show_image=False):
    fig = plt.figure()

    attack_dict = ["Hue", "Saturate ", "Rotate ", "Bright", "Contrast       ", "L-infty"]

    labels = [['0.16', '0.33π', '0.5π', '0.66π', '0.83π', ''], ['10%', '20%', '30%', '40%', '50%', ''],
              ['5°', '10°', '15°', '20°', '25°', ''], ['0.05', '0.1', '0.15', '0.2', '0.25', ''],
              ['10%', '20%', '30%', '40%', '50%', ''], ['1/255', '2/255', '3/255', '4/255', '5/255', '']]

    radar = Radar(fig, attack_dict, labels)
    tuned_value = [values[0]/pi*6, values[1] * 12, values[2]/5, values[3] * 20, values[4] * 12, values[5] * 255]
    radar.plot(tuned_value, "-", linewidth=1, color="g", alpha=.5, label=legend)
    radar.fill(tuned_value, color="g", alpha=.6)

    # radar.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=4)
    fig.tight_layout(pad=2)
    fig = plt.gcf()
    fig.set_size_inches(7.2, 9, forward=True)
    if show_image:
        plt.show()
    fig.savefig('./radar.png')
    return fig

# if __name__ == '__main__':
#     attack_order = (0,1,2)
#     adv_val = [adv_hue, adv_sat, adv_rot/30, adv_bright, adv_contrast, linf_eps]
#     radar_value = [adv_val[i].item() if i in attack_order else 0 for i in range(6)]
#     draw_radar_chart(values=radar_value, legend=show_enabled_attack(attack_order))
