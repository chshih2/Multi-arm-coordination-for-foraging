import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, NullFormatter
import sys
from scipy.spatial import distance

from comm.algorithms.workspace import (
    generate_targets_inside_workspace,
    read_polygon_workspace,
    plot_polygon_workspace,
    check_point_in_polygon,
)
from lower_continuous_muscles.learn_deformation import (
    tf_energy_difference,
    tf_target_config,
    np_target_config,
    tf_distribute_activation,
    tf_activation_to_muscle_force,
    get_rod_parameter,
    make_rod,
    make_basis,
    test_rod,
    generate_straight,
)
import tensorflow as tf

from utility.geometry import cartesian2polar
from upper_controller.fixed_targets_env import FixedTargetsEnv
from upper_controller.greedy import GreedyPath


#  LEGACY goes with coord_ppo.py

@tf.function
def tf_muscle_activation_energy(activation):
    energy = tf.linalg.trace(tf.matmul(activation, tf.transpose(activation)))
    return energy


class CoordinationEnv:
    def __init__(
        self,
        num_targets,
        obs_num_targets,
        ground_range,
        penalty_coeff,
        bonus_coeff,
        termination_step,
        state_type,
        simulation_model,
        simulation_dir,
        render_plot=False,
        online=True,
        render_dir="frames/",
    ):
        self.num_targets = num_targets
        self.obs_num_targets = obs_num_targets
        self.render_dir = render_dir
        self.crawl_amount = np.array(
            [0.2775787711143494, 0.0, 0.0]
        )  # when transverse muscle are all activated
        self.arm_length = np.array([1.0, 0.0, 0.0])
        self.penalty_coeff = penalty_coeff
        self.bonus_coeff = bonus_coeff
        self.render_plot = render_plot
        self.ep = 0
        (
            self.hull_pts,
            self.polygon,
            self.view_pts,
            self.view_polygon,
        ) = read_polygon_workspace(simulation_dir, view=True)
        self.ground_range = ground_range
        self.reachable_targets = np.zeros(self.num_targets, dtype=int)
        self.observable_targets = np.zeros(self.num_targets, dtype=int)

        self.termination_step = termination_step

        if state_type == "loc_map_energy":
            self.state_fn = lambda: np.concatenate(
                [self.polar_targets.flatten(), self.reached_map, [self.energy]]
            )
        elif state_type == "loc_map":
            self.state_fn = lambda: np.concatenate(
                [self.polar_targets.flatten(), self.reached_map]
            )
        elif state_type == "loc":
            self.state_fn = lambda: np.concatenate([self.polar_targets.flatten()])
        elif state_type == "loc_energy":
            self.state_fn = lambda: np.concatenate(
                [self.polar_targets.flatten(), [self.energy]]
            )
        elif state_type == "loc_avail":
            self.state_fn = lambda: np.concatenate(
                [self.polar_targets.flatten(), self.state_reachable_list]
            )

        self.online = online
        if self.online:
            # self.nn = tf.keras.models.load_model(simulation_model, compile=False)

            reward_type = "normalize_plain"
            state_type = "d_tip_at_f"
            num_components = 11
            num_muscles = 3
            ground_range = 0.0
            online = True

            rod = FixedTargetsEnv(
                num_targets=obs_num_targets,
                num_components=num_components,
                num_muscles=num_muscles,
                ground_range=ground_range,
                simulation_dir=simulation_dir,
                simulation_model=os.path.join(simulation_dir , "0704_model.h5"),
                reward_type=reward_type,
                state_type=state_type,
                target_file=None,
                optimal_file=None,
                permutation_file=None,
                render_plot=False,
                online=online,
            )
            self.agent = GreedyPath(rod, test_sample_file=None)

    def cal_state(self):
        self.update_reachable_region()
        self.tip = self.pos[:2]
        self.observe_index = np.where(self.observable_targets == 1)[0]
        self.state = np.zeros(self.obs_num_targets * 2)
        obs_list = np.intersect1d(self.observe_index, self.unreached_target_index)
        if len(obs_list) > 0:
            self.make_polar_targets(obs_list=self.target_range[obs_list], tip=self.tip)
            self.polar_targets = self.polar_targets[self.polar_targets[:, 0].argsort()]
            self.polar_targets = self.polar_targets[: self.obs_num_targets]
            self.state[: len(self.polar_targets) * 2] = self.polar_targets.flatten()

    def crawl(self):
        self.base += self.crawl_amount
        self.pos += self.crawl_amount

    def step(self, current_action):
        self.action = current_action
        if current_action == 0:  # reach
            self.energy = self.collect_with_greedy()
            n_target_collected = self.collect_targets_in_workspace()
            self.reward = (
                (n_target_collected - self.energy / 300.0) / 2.0 + self.bonus_coeff
                if n_target_collected > 0
                else -self.penalty_coeff
            )
        elif current_action == 1:  # crawl
            self.crawl()
            self.energy = 99.0
            self.reward = (-self.energy / 300.0) / 2.0 + self.bonus_coeff
        self.total_energy += self.energy
        self.counter += 1
        if self.render_plot:
            self.render(current_action)

        # if current_action == 0:  # reach+crawl
        #     self.crawl()
        #     self.total_energy+=99.0

        self.cal_state()
        done = self.check_done()

        return self.state, self.reward, done

    def cal_energy_for_redundant_step(self):
        target = generate_targets_inside_workspace(self.polygon)
        target = target[:2]
        curr_activation = generate_straight(eager=False)[0]
        _, energy_cost, _ = self.agent.env.cal_activation_and_energy(
            target, curr_activation
        )
        return energy_cost

    def check_missing_target(self):
        # end episode if there are missed targets
        return self.base[0] > (self.end_mark + 0.5) and len(self.new_reached_list) == 0

    def check_done(self):
        good_done = self.targets_eaten == self.num_targets
        miss_done = self.check_missing_target()
        break_done = self.counter > self.termination_step
        done = good_done or break_done or miss_done
        return done

    def collect_with_greedy(self):
        total_energy = 0.0
        self.log_targets = []
        if len(self.new_reached_list) > 0:
            greedy_actions = list(
                np.setdiff1d(range(self.num_targets), self.new_reached_list)
            )
            self.log_activations = []
            self.agent.env.activation = generate_straight(eager=False)[0]
            remain_list = list(self.new_reached_list)

            for _ in range(len(self.new_reached_list)):
                (
                    potential_activation,
                    energy_cost_map,
                ) = self.agent.env.cal_energy_cost_map(
                    self.agent.env.activation, remain_list
                )
                energy_cost_map[greedy_actions] = np.inf
                action = np.argmin(energy_cost_map)
                greedy_actions.append(action)
                diff_activation = np_target_config(
                    potential_activation[action][np.newaxis, :]
                ) - np_target_config(self.agent.env.activation[np.newaxis, :])
                clip_activation = np.maximum(
                    diff_activation, np.zeros_like(diff_activation)
                )
                self.agent.env.activation = potential_activation[action]
                remain_list = np.setdiff1d(remain_list, action)
                # total_energy += energy_cost_map[action] #use energy difference (du^2)
                apply_activation = np_target_config(
                    potential_activation[action][np.newaxis, :]
                )
                # abs_energy = np.trace(np.matmul(apply_activation, np.transpose(apply_activation)))
                abs_energy = np.trace(
                    np.matmul(clip_activation, np.transpose(clip_activation))
                )
                self.log_activations.append(apply_activation)
                self.log_targets.append(self.agent.env.target_range[action])
                total_energy += abs_energy  # use energy (u^2)
            # total_energy += self.agent.env.cal_return_energy(self.agent.env.activation).numpy() #use energy difference (du^2)
        return total_energy

    def plot_straight_arm(self):
        # marker_size = np.linspace(150, 10, 100)
        arm_x = np.linspace(self.base[0], self.pos[0], 100)
        arm_y = np.zeros_like(arm_x)
        for k in range(100):
            self.ax.scatter(
                arm_x[k],
                arm_y[k],
                s=self.marker_size[k],
                color="mediumpurple",
                alpha=0.3,
            )
        (plot,) = self.ax.plot(arm_x, arm_y, "mediumpurple", linewidth=2)
        return plot

    def plot_targets(self):
        target_list = self.target_range
        for i in range(self.num_targets):
            plot_reachable, plot_unreached = None, None
            if self.reachable_targets[i] == 1:
                # plt.plot(target_list[i, 0], target_list[i, 1], 'ro', markersize=15, linewidth=14,
                #          alpha=0.2)  # , label="reachable")
                plot_reachable = self.ax.scatter(
                    target_list[i, 0],
                    target_list[i, 1],
                    marker="o",
                    color="r",
                    label="target",
                    s=300,
                    linewidths=7,
                    alpha=0.2,
                )
            if self.reached_map[i] == 0:
                # plt.plot(target_list[i, 0], target_list[i, 1], 'bx', markersize=8, label="target%d" % i)
                plot_unreached = self.ax.scatter(
                    target_list[i, 0],
                    target_list[i, 1],
                    marker="x",
                    color="b",
                    label="target",
                    s=300,
                    linewidths=7,
                )
        return plot_reachable, plot_unreached

    def save_plot(self, current_action, energy, legend_list):
        # plt.title("time step %d action %d, eaten %d, crawl step %d,\n energy %.2f, total energy %.2f, reward %.2f" % (
        #     self.counter, current_action, self.targets_eaten, self.crawl_step, energy/300.0, self.total_energy/300.0, self.reward))
        #
        # plt.gca().set_aspect('equal')
        # plt.gca().set_xlim([-1.0, 10.5])
        # plt.gca().set_ylim([-1.5, 1.5])
        # plt.savefig(self.render_dir + "/%03d_%03d" % (self.ep, self.render_counter))
        # # plt.show()
        # plt.close('all')

        # self.ax.xaxis.set_minor_locator(AutoMinorLocator())
        # self.ax.yaxis.set_minor_locator(AutoMinorLocator())
        label_list = [
            "workspace",
            "view range",
            "arm",
            "reachable targets",
            "unreached targets",
        ]
        if legend_list[3] == None:
            legend_list.pop(3)
            label_list.pop(3)
        if legend_list[-1] == None:
            legend_list.pop(-1)
            label_list.pop(-1)
        # plt.legend(legend_list,
        #            label_list, fontsize=28, bbox_to_anchor=(1, 1), loc="upper left")
        self.ax.set_xlim(-1.0, 10.1)
        self.ax.set_ylim(-0.1, 1.2)
        self.ax.xaxis.set_ticks(np.arange(0.0, 10.1, 1.0))
        self.ax.yaxis.set_ticks(np.arange(0.0, 1.1, 0.5))
        plt.gca().set_aspect("equal", adjustable="box")

        # self.ax.tick_params(which='both', width=2, labelsize=28)
        # self.ax.tick_params(which='major', length=7)
        # self.ax.tick_params(which='minor', length=4)
        self.ax.yaxis.set_major_formatter(NullFormatter())
        self.ax.xaxis.set_major_formatter(NullFormatter())
        for axis in ["top", "bottom", "left", "right"]:
            self.ax.spines[axis].set_linewidth(2)

        # if current_action == 0:
        #     action_label = 'reach'
        # else:
        #     action_label = 'crawl'
        #     energy=99.0
        # plt.title(r"$t = %d$" % self.counter + ",\naction: %s,\nreached %d targets,\n" % (
        #     action_label, self.targets_eaten) + r'$E_{%d}=%.4f$'%(self.counter,energy / 300.0) + "\n" + r"total energy $E = %.4f$" % (self.total_energy / 300.0) + "\n",
        #           fontsize=28)

        left, width = 0.25, 0.65
        bottom, height = 0.25, 0.65
        right = left + width
        top = bottom + height
        # self.ax.text(right, top, r'$E_%d=%.4f$'%(self.counter,energy / 300.0),
        #         horizontalalignment='right',
        #         verticalalignment='top',
        #         transform=self.ax.transAxes,
        #         fontsize=28)

        self.fig.tight_layout()
        self.fig.savefig(
            self.render_dir + "/%03d_%03d" % (self.ep, self.render_counter),
            transparent=True,
        )
        plt.close(self.fig)
        plt.close("all")
        self.fig, self.ax = plt.subplots(figsize=(16, 6))

    def render(self, current_action):
        if current_action == 1:
            self.crawl_step += 1
        if current_action == -1 or current_action == 1:
            self.render_counter += 1
            plot1 = plot_polygon_workspace(
                self.hull_pts, x_shift=self.base[0], ax=self.ax
            )
            plot2 = plot_polygon_workspace(
                self.view_pts, x_shift=self.base[0], ax=self.ax
            )
            plot3 = self.plot_straight_arm()
            plot_reachable, plot_unreached = self.plot_targets()
            legend_list = [plot1, plot2, plot3, plot_reachable, plot_unreached]
            self.save_plot(current_action, 0.0, legend_list)
        elif current_action == 0:
            cur_state, done, rewards = self.agent.env.reset(), False, 0
            if len(self.new_reached_list) > 0:
                greedy_actions = list(
                    np.setdiff1d(range(self.num_targets), self.new_reached_list)
                )
                for _ in range(self.num_targets - len(greedy_actions)):
                    pre_activation = self.agent.env.activation.copy()
                    self.render_counter += 1
                    plot1 = plot_polygon_workspace(
                        self.hull_pts, x_shift=self.base[0], ax=self.ax
                    )
                    plot2 = plot_polygon_workspace(
                        self.view_pts, x_shift=self.base[0], ax=self.ax
                    )
                    greedy_state = self.agent.env.energy_cost_map.copy()
                    greedy_state[greedy_actions] = np.inf
                    action = np.argmin(greedy_state)
                    greedy_actions.append(action)
                    _, _, _ = self.agent.env.step(action)
                    arm = self.agent.env.cal_position()
                    diff_activation = np_target_config(
                        self.agent.env.activation[np.newaxis, :]
                    ) - np_target_config(pre_activation[np.newaxis, :])
                    clip_activation = np.maximum(
                        diff_activation, np.zeros_like(diff_activation)
                    )
                    # apply_activation = np_target_config(self.agent.env.activation[np.newaxis, :])
                    # energy = np.trace(np.matmul(apply_activation, np.transpose(apply_activation)))
                    energy = np.trace(
                        np.matmul(clip_activation, np.transpose(clip_activation))
                    )
                    # marker_size = np.linspace(150, 10, 100)
                    (plot3,) = self.ax.plot(
                        arm[0, :] / 0.2 + self.base[0],
                        arm[1, :] / 0.2,
                        "mediumpurple",
                        linewidth=2,
                    )
                    for k in range(100):
                        plt.scatter(
                            arm[0, k] / 0.2 + self.base[0],
                            arm[1, k] / 0.2,
                            s=self.marker_size[k],
                            color="mediumpurple",
                            alpha=0.3,
                        )

                    target_list = self.target_range
                    i = action
                    self.ax.scatter(
                        target_list[i, 0],
                        target_list[i, 1],
                        marker="x",
                        color="b",
                        label="target",
                        s=300,
                        linewidths=7,
                    )

                    # plt.plot(target_list[i, 0], target_list[i, 1], 'bx', markersize=12, label="target%d" % i)

                    for i in np.setdiff1d(self.new_reached_list, greedy_actions):
                        # plt.plot(target_list[i, 0], target_list[i, 1], 'bx', markersize=12, label="target%d" % i)
                        self.ax.scatter(
                            target_list[i, 0],
                            target_list[i, 1],
                            marker="x",
                            color="b",
                            label="target",
                            s=300,
                            linewidths=7,
                        )
                    plot_reachable, plot_unreached = self.plot_targets()
                    legend_list = [plot1, plot2, plot3, plot_reachable, plot_unreached]
                    self.save_plot(current_action, energy, legend_list)

                self.render_counter += 1
                plot1 = plot_polygon_workspace(
                    self.hull_pts, x_shift=self.base[0], ax=self.ax
                )
                plot2 = plot_polygon_workspace(
                    self.view_pts, x_shift=self.base[0], ax=self.ax
                )
                plot3 = self.plot_straight_arm()
                plot_reachable, plot_unreached = self.plot_targets()
                # return_energy = self.agent.env.cal_return_energy(self.agent.env.activation) #use energy difference (du^2)
                return_energy = 0  # use energy (u^2)
                self.total_energy += return_energy
                legend_list = [plot1, plot2, plot3, plot_reachable, plot_unreached]
                self.save_plot(
                    current_action, energy=return_energy, legend_list=legend_list
                )
            else:
                self.render_counter += 1
                plot1 = plot_polygon_workspace(self.hull_pts, x_shift=self.base[0])
                plot2 = plot_polygon_workspace(self.view_pts, x_shift=self.base[0])
                plot3 = self.plot_straight_arm()
                plot_reachable, plot_unreached = self.plot_targets()
                legend_list = [plot1, plot2, plot3, plot_reachable, plot_unreached]
                self.save_plot(current_action, energy=0.0, legend_list=legend_list)

    def make_polar_targets(self, obs_list, tip=[1.0, 0.0]):
        polar_targets = []
        tip_x, tip_y = tip
        tip_x -= 1.0

        for target in obs_list:  # self.target_range:
            (
                x,
                y,
                _,
            ) = target  # - self.base (no need to minus the base because tip and target are all absolute)
            pol_coor = cartesian2polar(x - tip_x, y - tip_y)
            polar_targets.append(pol_coor)
        self.polar_targets = np.array(polar_targets)
        self.polar_targets[:, 1] /= np.pi

    def reset(self):
        self.counter = 0
        self.render_counter = 0
        self.total_cost = 0.0
        self.crawl_step = 0
        self.reward = 0.0
        self.action = -1
        self.base = np.array([0.0, 0.0, 0.0])
        self.pos = np.array([1.0, 0.0, 0.0])

        self.agent.env.target_range = self.target_range - self.base
        self.prev_reached_list = []
        self.total_energy = 0.0
        self.end_mark = np.max(self.target_range[:, 0])
        self.targets_eaten = 0
        self.reached_map = np.zeros(self.num_targets, dtype=int)
        self.cal_state()
        self.ep += 1
        self.reach_flag = False

        if self.render_plot:
            self.marker_size = np.linspace(200, 5, 100)
            self.fig, self.ax = plt.subplots(figsize=(16, 6))
            self.render(-1)
        return self.state

    def update_reachable_region(self):
        for i, target in enumerate(self.target_range):
            target_relative = target.copy()
            target_relative -= self.base
            target_relative[1] = abs(target_relative[1])
            inside_workspace = check_point_in_polygon(target_relative, self.polygon)
            if inside_workspace:
                self.reachable_targets[i] = 1
                self.observable_targets[i] = 1
            else:
                self.reachable_targets[i] = -1
                self.observable_targets[i] = (
                    1
                    if check_point_in_polygon(target_relative, self.view_polygon)
                    else -1
                )
        self.agent.env.target_range = self.target_range - self.base
        self.agent.env.map_target = self.num_targets
        self.agent.env.num_targets = self.num_targets
        self.reachable_index = np.where(self.reachable_targets == 1)[0]
        self.unreached_target_index = np.where(self.reached_map == 0)[0]
        self.new_reached_list = np.intersect1d(
            self.reachable_index, self.unreached_target_index
        )
        self.state_reachable_list = np.zeros(self.num_targets)
        self.state_reachable_list[self.new_reached_list] = 1

    def collect_targets_in_workspace(self):
        n_target_in_range = len(self.new_reached_list)
        self.targets_eaten += n_target_in_range
        self.reached_map[self.new_reached_list] = 1
        return n_target_in_range

    def generate_n_targets(self, from_n_map):
        if from_n_map > 0:
            random_draw = np.random.randint(from_n_map)
            self.target_range = self.target_sample[random_draw]
            if self.reward_type == "01sanity" or self.num_targets == 2:
                self.optimal_action = self.optimal_action_sample[random_draw]
            elif self.reward_type == "mean_std_sanity":
                self.energy_mean = self.permutation_mean[random_draw] / (
                    self.num_targets + 1
                )
                self.energy_std = self.permutation_std[random_draw]
        else:
            self.target_range = []
            for i in range(self.num_targets):
                flag = True
                while flag:
                    target = generate_targets_inside_workspace(self.polygon)
                    target[0] += np.random.rand() * self.ground_range
                    flag = check_point_in_polygon(np.abs(target), self.polygon)
                self.target_range.append(target)
            self.target_range = np.array(self.target_range)

    # def cal_total_energy(self, actions):
    #     activation = generate_straight()[0]  # np.zeros(self.num_components * self.num_muscles)
    #
    #     total_energy = 0
    #     activation_record = []
    #     energy_record = []
    #     for index in actions:
    #         target = self.target_range[index][:2]
    #         activation, energy, _ = self.cal_activation_and_energy(target, activation)
    #         total_energy += energy
    #         activation_record.append(activation)
    #         energy_record.append(energy.numpy())
    #     energy = self.cal_return_energy(activation)
    #     total_energy += energy
    #     energy_record.append(energy.numpy())
    #     return total_energy, activation_record, energy_record
