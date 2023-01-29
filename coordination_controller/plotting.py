import os
import sys
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def box_config(label, x_axis_list, result_list, color_list, f,load_model, note=""):
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    # fig, ax = plt.subplots(figsize=(3.0, 4.0))
    all_data = [f(result, label) for result in
                result_list]  # [ppo1_result[label], ppo2_result[label], ppo3_result[label], random_result[label]]

    bp = ax.boxplot(all_data, showfliers=False, patch_artist=True)
    # bp = ax.boxplot(all_data, widths=0.5, showfliers=False, patch_artist=True)
    for j, box in enumerate(bp['boxes']):
        box.set(linewidth=2)
        box.set(facecolor=color_list[j])
    for whisker in bp['whiskers']:
        whisker.set(linewidth=2)
    for cap in bp['caps']:
        cap.set(linewidth=2)
    for median in bp['medians']:
        median.set(linewidth=2)

    plt.setp(ax, xticks=[y + 1 for y in range(len(all_data))],
             xticklabels=x_axis_list)
    plt.title(label)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(which='both', width=2, labelsize=16)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    fig.tight_layout()
    fig.savefig(load_model + "/" + label + note + ".png", transparent=True)
    plt.close(fig)

def plot_observable_states(model_dir, load_data=False):
    num_eps = 10
    num_arms = 4
    if load_data:
        reach_flag_data = {}
        for algo_dir in ["frames_ppo", "frames_simplified"]:
            for contact in [0, 1]:
                reach_flag_list = []
                for sample in range(num_eps):
                    for arm in range(num_arms):
                        case_dir = (
                            "contact%d_eps*_online0_low_Ce0.100_Re0.030_Cr5.000_Rr0.010/"
                            % contact
                        )
                        cases = glob.glob(
                            os.path.join(
                                model_dir,
                                algo_dir,
                                f"ep{sample}",
                                "activation",
                                f"arm{arm}",
                                case_dir,
                            )
                        )

                        for case in cases:
                            reach_flag_list.append(
                                bool(np.load(os.path.join(case, "reach_flag.npy")))
                            )
                reach_flag_data[algo_dir + "_contact%d" % contact] = reach_flag_list
        np.savez(os.path.join(model_dir, "obstacle_reaching_stats"), **reach_flag_data)
    else:
        data = np.load(os.path.join(model_dir, "obstacle_reaching_stats.npz"))
        reaching_percentage = lambda x: sum(x) / num_eps / 20 * 100
        result_list = [
            reaching_percentage(data["frames_simplified_contact0"]),
            reaching_percentage(data["frames_simplified_contact1"]),
            reaching_percentage(data["frames_ppo_contact0"]),
            reaching_percentage(data["frames_ppo_contact1"]),
        ]
        x_axis_list = ["ppo+feedback", "ppo", "heuristic+feedback", "heuristic"]
        color_list = ["mediumpurple", "yellowgreen", "cornflowerblue", "red"]

        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        ax.bar(x_axis_list, result_list, color=color_list, width=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both", width=2, labelsize=16)
        ax.tick_params(which="major", length=7)
        ax.tick_params(which="minor", length=4)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2)
        fig.tight_layout()
        fig.savefig(
            os.path.join(model_dir, "obstacle_reaching_stats.png"), transparent=True
        )


def plot_box(load_model, num_targets, num_arms, food_weights, **kwargs):
    ppo_result = np.load(
        os.path.join(load_model, "test_score_random0_greedy0_simplified0.npz")
    )
    random_result = np.load(
        os.path.join(load_model, "test_score_random1_greedy0_simplified0.npz")
    )
    simplified_result = np.load(
        os.path.join(load_model, "test_score_random0_greedy0_simplified1.npz")
    )
    if num_arms == 2:
        greedy_result = np.load(
            os.path.join(load_model, "test_score_random0_greedy1_simplified0.npz")
        )
        result_list = [ppo_result, simplified_result, greedy_result, random_result]
        x_axis_list = ["ppo", "Q", "greedy", "random"]
        color_list = ["mediumpurple", "yellowgreen", "cornflowerblue", "red"]
    else:
        # result_list =[ppo_result,  simplified_result,random_result]
        # x_axis_list = ["ppo", "roomba","random"]
        # color_list = ['mediumpurple','yellowgreen', 'red']
        result_list = [ppo_result, simplified_result]
        x_axis_list = ["ppo", "heuristic"]
        color_list = ["mediumpurple", "yellowgreen"]

    # missed_index = np.where(ppo_result['targets_eaten'] < num_targets)[0]
    #
    # # remove the index without success
    # greater = np.where(greedy_result["energy"] > ppo_result["energy"])[0]
    # video_index = np.setdiff1d(greater, missed_index)
    # print("cases that ppo has better results /n", video_index)
    #
    # target_range = np.load(load_model + "target%d_test_across_model.npy" % num_targets)
    #
    # check_targets = 0  # bool(int(input("check target positions in each case? (0 -> False / 1 -> True)")))
    # if check_targets:
    #     import matplotlib.pyplot as plt
    #     for i in video_index:
    #         print("case %d" % i)
    #         marker_size = np.linspace(150, 10, 100)
    #         arm_x = np.linspace(0.0, 1.0, 100)
    #         arm_y = np.zeros_like(arm_x)
    #         for k in range(100):
    #             plt.scatter(arm_x[k], arm_y[k], s=marker_size[k], color='k', alpha=0.2)
    #         plt.plot(target_range[i, :, 0], target_range[i, :, 1], 'bx', markersize=8, label="target%d" % i)
    #         plt.gca().set_aspect('equal')
    #         plt.gca().set_xlim([-1.0, 10.5])
    #         plt.gca().set_ylim([-1.5, 1.5])
    #         plt.savefig(load_model + "/ppo_better_%d.png" % i)
    #         plt.close("all")

    # for g in video_index:
    #     os.system(
    #         "ffmpeg -r 1 -i " + load_model + "/frames_ppo/%03d_%%03d.png -b:v 90M -c:v libx264 -pix_fmt yuv420p -f mov -y " % (
    #                     g + 1) + load_model + "/sample%d-0.0.mov" % (g + 1))
    # for g in video_index:
    #     os.system(
    #         "ffmpeg -r 1 -i " + load_model + "/frames_greedy/%03d_%%03d.png -b:v 90M -c:v libx264 -pix_fmt yuv420p -f mov -y " % (
    #                     g + 1) + load_model + "/sample%d-0.0.mov" % (g + 1))

    label_list = ["step", "energy", "modified_energy"]

    for i, label in enumerate(label_list):
        f = lambda x, y: x[y] / x["targets_eaten"]
        box_config(
            label,
            x_axis_list,
            result_list,
            color_list,
            f,
            load_model=load_model,
            note="",
        )

    label_list = ["arm_crawl_step", "arm_targets_eaten", "arm_energy"]
    for label in label_list:
        for i in range(num_arms):
            f = lambda x, y: x[y][:, i] / x["targets_eaten"]
            box_config(
                label,
                x_axis_list,
                result_list,
                color_list,
                f,
                load_model=load_model,
                note="%d" % i,
            )

    label_list = ["targets_eaten"]
    for label in label_list:
        f = lambda x, y: x["targets_eaten"]
        box_config(
            label,
            x_axis_list,
            result_list,
            color_list,
            f,
            load_model=load_model,
            note="%d" % i,
        )

    label_list = ["plus_food_energy"]
    for label in label_list:
        f = lambda x, y: food_weights * x["targets_eaten"] - x["modified_energy"]
        box_config(
            label,
            x_axis_list,
            result_list,
            color_list,
            f,
            load_model=load_model,
            note="%d" % i,
        )

    label_list = ["arm_food_energy"]
    for label in label_list:
        for i in range(num_arms):
            f = (
                lambda x, y: x["arm_targets_eaten"][:, i]
                - x["arm_energy"][:, i] / 300.0
            )
            box_config(
                label,
                x_axis_list,
                result_list,
                color_list,
                f,
                load_model=load_model,
                note="%d" % i,
            )
