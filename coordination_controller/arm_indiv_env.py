import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import math

from actuations.workspace import (
    read_polygon_workspace,
    plot_polygon_workspace,
    check_point_in_polygon,
    cartesian2polar,
    y_rotation
)
from actuations.nnes_fn import (
    np_target_config,
    generate_straight,
)

from coordination_controller.fixed_targets_env import FixedTargetsEnv
from coordination_controller.greedy import GreedyPath


class SingleArmEnv:
    def __init__(
            self,
            angle,
            num_targets,
            obs_num_targets,
            simulation_dir,
            bonus_coeff,
            reward_type,
            use_arena,
            use_obstacle,
            flip_crawl_direction,
            food_weights,
    ):

        self.arm_length = np.array([1.0, 0.0, 0.0])

        if flip_crawl_direction:
            self.flip = -1
            self.crawl_amount = np.array(
                [0.2, 0.0, 0.0]
            )  # when LM1 and LM2 are all activated
            self.crawl_energy = 99.0 * 2
        else:
            self.flip = 1
            self.crawl_amount = np.array(
                [0.2775787711143494, 0.0, 0.0]
            )  # when transverse muscle are all activated
            self.crawl_energy = 99.0
        self.crawl_amount = self.flip * y_rotation(self.crawl_amount, angle)

        self.arm_length = y_rotation(self.arm_length, angle)
        self.angle = angle
        self.num_targets = num_targets
        self.obs_num_targets = obs_num_targets
        (
            self.hull_pts,
            self.polygon,
            self.view_pts,
            self.view_polygon,
        ) = read_polygon_workspace(simulation_dir, view=True)
        self.bonus_coeff = bonus_coeff
        num_components = 11
        num_muscles = 3
        ground_range = 0.0
        online = True
        self.use_arena = use_arena
        self.use_obstacle = use_obstacle
        rod = FixedTargetsEnv(
            num_targets=obs_num_targets,
            num_components=num_components,
            num_muscles=num_muscles,
            ground_range=ground_range,
            simulation_dir=simulation_dir,
            simulation_model=os.path.join(simulation_dir, "pretrained_model.h5"),
            reward_type="normalize_plain",
            state_type="d_tip_at_f",
            target_file=None,
            optimal_file=None,
            permutation_file=None,
            render_plot=False,
            online=online,
        )

        self.agent = GreedyPath(rod, test_sample_file=None)

        if reward_type == "no-shift":
            self.cal_reach_reward = (
                lambda: food_weights * self.n_target_collected - self.energy / 300.0
            )
            self.cal_crawl_reward = lambda: -self.energy / 300.0
        elif reward_type == "shift":
            self.cal_reach_reward = (
                lambda: (food_weights * self.n_target_collected - self.energy / 300.0)
                        / 2.0
                        + self.bonus_coeff
            )
            self.cal_crawl_reward = (
                lambda: (-self.energy / 300.0) / 2.0 + self.bonus_coeff
            )
        else:
            raise NotImplementedError("not implemented reward function")

    def arm_reset(self, target_range, obstacle_pts):
        self.n_target_collected = 0
        self.target_range = target_range
        self.obstacle_pts = obstacle_pts
        self.base = np.array([0.0, 0.0, 0.0])
        self.pos = self.base + self.arm_length  # np.array([1.0, 0.0, 0.0])
        self.reachable_targets = np.zeros(self.num_targets, dtype=int)
        self.observable_targets = np.zeros(self.num_targets, dtype=int)
        self.agent.env.target_range = self.target_range - self.base
        self.targets_eaten = 0
        self.reached_map = np.zeros(self.num_targets, dtype=int)
        self.end_mark = np.max(self.target_range[:, 0])
        self.render_counter = 0
        self.log_state = {
            "log_targets": [],
            "log_activations": [],
            "log_BCs": [],
            "log_marks": [],
            "log_head": [],
            "log_pos": [],
        }
        self.wall = 0  # if self.angle >= -90 and self.angle <= 0 else 1
        self.energy = 0.0
        self.reward = 0.0
        self.log_activations = []

    def crawl(self, head):
        self.base = head
        self.n_target_collected = 0
        #     base_if_crawl = head + self.crawl_amount
        #     self.base = base_if_crawl
        self.energy = self.crawl_energy
        #     self.wall = 0
        #     if self.use_arena:
        #         if base_if_crawl[0]>self.boundary[0] or base_if_crawl[2]>self.boundary[1] or base_if_crawl[0]<0.0 or base_if_crawl[2]<0.0:
        #             self.base=head
        #             self.energy = 0.0
        #             self.wall=1
        self.pos = self.base + self.arm_length

        self.reward = self.cal_crawl_reward()

    #
    #     return self.base

    def update_reachable_region(self):

        # QUESTION
        if self.use_obstacle:
            # rel_rot_obstacle = y_rotation(self.ob)
            obstacle = np.array(self.obstacle_pts)[:, :, :4]
            obstacle = np.swapaxes(obstacle, 1, 2)
            obs_on_the_way = []
            all_rel_rot_obstacle = []
            for i, obs in enumerate(obstacle):
                rel_obs = np.insert(obs, 1, np.zeros(4), axis=1) - self.base
                rel_rot_obstacle = y_rotation(rel_obs, -self.angle)
                all_rel_rot_obstacle.append(rel_rot_obstacle)
                if (
                        max(rel_rot_obstacle[:, 2]) > 0
                        and min(rel_rot_obstacle[:, 2]) < 0
                        and min(rel_rot_obstacle[:, 0]) > 0
                ):
                    obs_on_the_way.append(i)
        self.rel_rot_targets = y_rotation(self.target_range - self.base, -self.angle)
        for i, target in enumerate(self.rel_rot_targets):
            inside_workspace = False
            inside_view = False
            if math.isclose(target[2], 0.0, abs_tol=1e-4):
                block_flag = 0
                if self.use_obstacle:
                    for j in obs_on_the_way:
                        if target[0] > min(all_rel_rot_obstacle[j][:, 0]):
                            block_flag += 1
                if block_flag == 0:
                    target[1] = abs(target[1])
                    inside_workspace = check_point_in_polygon(target, self.polygon)
                    if not inside_workspace:
                        inside_view = check_point_in_polygon(target, self.view_polygon)
            if inside_workspace:
                self.reachable_targets[i] = 1
                self.observable_targets[i] = 1
            else:
                self.reachable_targets[i] = -1
                self.observable_targets[i] = 1 if inside_view else -1
        self.agent.env.target_range = (
            self.rel_rot_targets
        )  # self.target_range - self.base
        self.agent.env.map_target = self.num_targets
        self.agent.env.num_targets = self.num_targets
        self.reachable_index = np.where(np.int64(self.reachable_targets) == 1)[0]
        self.cal_reachable_list()

    def update_reached_map(self, new_reached_map):
        self.reached_map = new_reached_map
        self.cal_reachable_list()

    def cal_reachable_list(self):
        self.unreached_target_index = np.where(np.int64(self.reached_map) == 0)[0]
        self.new_reached_list = np.intersect1d(
            self.reachable_index, self.unreached_target_index
        )
        # self.state_reachable_list = np.zeros(self.num_targets)
        # self.state_reachable_list[self.new_reached_list] = 1

    def cal_state(self):
        self.update_reachable_region()
        self.tip = self.pos[:2]
        observe_index = np.where(np.int64(self.observable_targets) == 1)[0]
        self.state = np.zeros(self.obs_num_targets * 2)
        self.obs_list = np.intersect1d(observe_index, self.unreached_target_index)
        if len(self.obs_list) > 0:
            self.make_polar_targets(
                obs_list=self.rel_rot_targets[self.obs_list], tip=self.tip
            )
            self.polar_targets = self.polar_targets[self.polar_targets[:, 0].argsort()]
            self.polar_targets = self.polar_targets[: self.obs_num_targets]
            self.state[: len(self.polar_targets) * 2] = self.polar_targets.flatten()
        self.prev_reached_list = self.new_reached_list
        return self.state

    def update_head(self, head):
        self.base = head
        self.pos = head + self.arm_length

    def reach(self):
        self.energy = self.collect_with_greedy()
        self.n_target_collected = self.collect_targets_in_workspace()
        self.reward = self.cal_reach_reward()
        return self.n_target_collected

    def collect_with_greedy(self):
        total_energy = 0.0
        self.log_targets = []
        if len(self.new_reached_list) > 0:
            greedy_actions = list(
                np.setdiff1d(range(self.num_targets), self.new_reached_list)
            )
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
                clip_activation = diff_activation
                clip_activation = np.maximum(
                    diff_activation, np.zeros_like(diff_activation)
                )
                # clip_activation=np_target_config(
                #     potential_activation[action][np.newaxis, :]
                # )
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
                self.log_targets.append(self.target_range[action])
                total_energy += abs_energy  # use energy (u^2)
            # total_energy += self.agent.env.cal_return_energy(self.agent.env.activation).numpy() #use energy difference (du^2)
        return total_energy

    # def miss_done(self):
    #     return self.base[0] > self.end_mark and len(self.new_reached_list) == 0

    def collect_targets_in_workspace(self):
        n_target_in_range = len(self.new_reached_list)
        self.targets_eaten += n_target_in_range
        self.reached_map[self.new_reached_list] = 1
        return n_target_in_range

    def make_polar_targets(self, obs_list, tip=[1.0, 0.0]):
        polar_targets = []
        # tip_x, tip_y = tip
        # tip_x -= 1.0
        tip = y_rotation(self.base, -self.angle)
        tip_x, tip_y, _ = tip

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

    def plot_straight_arm(self, x_shift=None):
        # marker_size = np.linspace(150, 10, 100)
        if x_shift == None:
            arm_x = np.linspace(self.base[0], self.pos[0], 100)
        else:
            arm_x = np.linspace(x_shift, x_shift + self.arm_length[0], 100)
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

    def record_movement(self, target, BC, mark, activation, head, arm):
        self.log_state["log_targets"].append(target)  # .copy() + env.base.copy())
        self.log_state["log_BCs"].append(BC)
        self.log_state["log_marks"].append(mark)
        self.log_state["log_activations"].append(activation)
        self.log_state["log_head"].append(head)
        self.log_state["log_pos"].append(arm)

    def generate_rest_frame(self, head, action, arm_index):
        plot1 = plot_polygon_workspace(self.hull_pts, x_shift=head[0], ax=self.ax)
        plot2 = plot_polygon_workspace(self.view_pts, x_shift=head[0], ax=self.ax)
        plot3 = self.plot_straight_arm(x_shift=head[0])
        plot_reachable, plot_unreached = self.plot_targets()
        legend_list = [plot1, plot2, plot3, plot_reachable, plot_unreached]
        self.save_plot(action, 0.0, legend_list, arm_index=arm_index)

    def arm_render(self, current_action, arm_index, head):
        if current_action == 1:
            self.crawl_step += 1
        if current_action == -1:
            self.render_counter += 1
            # self.generate_rest_frame(head, current_action, arm_index)
            self.record_movement(
                target=self.pos.copy(),
                BC=0,
                mark=0,
                activation=np.zeros((3, 99)),
                head=head,
                arm=self.rest_arm,
            )
            return head
        elif current_action == 1:
            self.render_counter += 1
            # self.generate_rest_frame(head, current_action, arm_index)
            head_record = head + self.crawl_amount
            self.record_movement(
                target=self.pos.copy(),
                BC=0,
                mark=0,
                activation=self.crawl_activation,
                head=head_record,
                arm=self.rest_arm,
            )
            return head_record
        elif current_action == 0:
            cur_state, done, rewards = self.agent.env.reset(), False, 0
            if self.n_target_collected > 0:
                greedy_actions = list(
                    np.setdiff1d(range(self.num_targets), self.prev_reached_list)
                )
                for _ in range(self.num_targets - len(greedy_actions)):
                    # pre_activation = self.agent.env.activation.copy()
                    self.render_counter += 1
                    # plot1 = plot_polygon_workspace(self.hull_pts, x_shift=head[0], ax=self.ax)
                    # plot2 = plot_polygon_workspace(self.view_pts, x_shift=head[0], ax=self.ax)
                    greedy_state = self.agent.env.energy_cost_map.copy()
                    greedy_state[greedy_actions] = np.inf
                    action = np.argmin(greedy_state)
                    greedy_actions.append(action)
                    _, _, _ = self.agent.env.step(action)
                    arm = self.agent.env.cal_position()
                    # diff_activation = np_target_config(
                    #     self.agent.env.activation[np.newaxis, :]) - np_target_config(pre_activation[np.newaxis, :])
                    # clip_activation = np.maximum(diff_activation, np.zeros_like(diff_activation))

                    # energy = np.trace(np.matmul(clip_activation, np.transpose(clip_activation)))
                    #
                    # plot3, = self.ax.plot(arm[0, :] / 0.2 + head[0], arm[1, :] / 0.2, 'mediumpurple', linewidth=2)
                    # for k in range(100):
                    #     self.ax.scatter(arm[0, k] / 0.2 + head[0], arm[1, k] / 0.2, s=self.marker_size[k],
                    #                     color='mediumpurple',
                    #                     alpha=0.3)

                    target_list = self.target_range
                    # self.ax.scatter(target_list[action, 0], target_list[action, 1], marker='x', color="b",
                    #                 label='target', s=300,
                    #                 linewidths=7)

                    # for i in np.setdiff1d(self.new_reached_list, greedy_actions):
                    #     self.ax.scatter(target_list[i, 0], target_list[i, 1], marker='x', color="b",
                    #                     label='target', s=300,
                    #                     linewidths=7)
                    # plot_reachable, plot_unreached = self.plot_targets()
                    # legend_list = [plot1, plot2, plot3, plot_reachable, plot_unreached]
                    # self.save_plot(current_action, energy, legend_list, arm_index=arm_index)

                    self.record_movement(
                        target=target_list[action],
                        BC=0,
                        mark=1,
                        activation=np_target_config(
                            self.agent.env.activation[np.newaxis, :]
                        ),
                        head=head,
                        arm=arm / 0.2,
                    )

                self.render_counter += 1
                # plot1 = plot_polygon_workspace(self.hull_pts, x_shift=head[0], ax=self.ax)
                # plot2 = plot_polygon_workspace(self.view_pts, x_shift=head[0], ax=self.ax)
                # plot3 = self.plot_straight_arm(x_shift=head[0])
                # plot_reachable, plot_unreached = self.plot_targets()
                # # return_energy = self.agent.env.cal_return_energy(self.agent.env.activation) #use energy difference (du^2)
                # return_energy = 0  # use energy (u^2)
                # # self.total_energy += return_energy
                # legend_list = [plot1, plot2, plot3, plot_reachable, plot_unreached]
                # self.save_plot(current_action, energy=return_energy, legend_list=legend_list, arm_index=arm_index)
                self.record_movement(
                    target=head + self.arm_length,
                    BC=0,
                    mark=0,
                    activation=np.zeros((3, 99)),
                    head=head,
                    arm=self.rest_arm,
                )
            else:
                self.render_counter += 1
                self.record_movement(
                    target=np.zeros(3),
                    BC=0,
                    mark=0,
                    activation=np.zeros((3, 99)),
                    head=head,
                    arm=self.rest_arm,
                )
                # plot1 = plot_polygon_workspace(self.hull_pts, x_shift=self.base[0])
                # plot2 = plot_polygon_workspace(self.view_pts, x_shift=self.base[0])
                # plot3 = self.plot_straight_arm()
                # plot_reachable, plot_unreached = self.plot_targets()
                # legend_list = [plot1, plot2, plot3, plot_reachable, plot_unreached]
                # self.save_plot(current_action, energy=0.0, legend_list=legend_list, arm_index=arm_index)
            return head

    def save_plot(self, current_action, energy, legend_list, arm_index):
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
            self.render_dir
            + "/arm%d_%03d_%03d" % (arm_index, self.ep, self.render_counter),
            transparent=True,
        )
        plt.close(self.fig)
        plt.close("all")
        self.fig, self.ax = plt.subplots(figsize=(16, 6))

    def set_fig_ax(self, ep, render_dir):
        self.marker_size = np.linspace(200, 5, 100)
        self.fig, self.ax = plt.subplots(figsize=(16, 6))
        self.render_dir = render_dir
        self.ep = ep
        self.crawl_step = 0
        self.log_arm_pos = []
        self.log_arm_activation = []
        self.rest_arm = np.zeros((3, 101))
        self.rest_arm[0] = np.linspace(0, 1, 101)
        self.crawl_activation = np.zeros((3, 99))
        self.crawl_activation[2, :] = 1.0
