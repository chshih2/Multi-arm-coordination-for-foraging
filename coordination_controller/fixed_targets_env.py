import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.spatial import distance

from actuations.workspace import (
    generate_targets_inside_workspace,
    read_polygon_workspace,
    cartesian2polar
)
from actuations.nnes_fn import (
    tf_energy_difference,
    tf_target_config,
    np_target_config,
    tf_distribute_activation,
    tf_activation_to_muscle_force,
    get_rod_parameter,
    make_rod,
    test_rod,
    generate_straight,
)
import tensorflow as tf



class FixedTargetsEnv:
    def __init__(
        self,
        num_targets,
        num_components,
        num_muscles,
        ground_range,
        simulation_model,
        simulation_dir,
        target_file,
        reward_type,
        state_type,
        optimal_file=None,
        permutation_file=None,
        render_plot=False,
        online=True,
    ):
        self.num_targets = num_targets
        self.num_components = num_components
        self.energy_weight = 1.0
        self.kappa_factor = 50
        self.sigma_factor = 10
        self.section_node = 20

        rod, env, self.max_force_list, self.muscle_radius_ratio_list = make_rod()
        self.rod_list = get_rod_parameter(rod)
        (
            self.shear_matrix,
            self.bend_matrix,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.rod_list

        if (
            target_file is not None
        ):  # use targets in training/testing set instead of generating on the fly
            self.target_sample = np.load(target_file)
        if (
            optimal_file is not None
        ):  # use optimal actions to guide learning (sanity check)
            optimal_sample = np.load(optimal_file)
            self.optimal_action_sample = optimal_sample["optimal_action"]
            self.optimal_energy_history = optimal_sample["optimal_energy_history"]
        if (
            permutation_file is not None
        ):  # calculate mean and std to guide learning (sanity check)
            permutation = np.load(permutation_file)
            self.permutation_mean = np.mean(permutation, axis=1)
            self.permutation_std = np.std(permutation, axis=1)

        self.num_muscles = num_muscles
        self.crawl_amount = np.array([0.08000185387782066, 0.0, 0.0])
        self.fetch_amount = np.array([0.1, 0.0, 0.0])  # mouth position relative to base
        self.arm_length = np.array([1.0, 0.0, 0.0])
        self.render_plot = render_plot
        self.ep = 0
        self.hull_pts, self.polygon = read_polygon_workspace(simulation_dir)
        self.ground_range = ground_range

        self.online = online
        if self.online:
            self.nn = tf.keras.models.load_model(simulation_model, compile=False)

        self.reward_type = reward_type
        if reward_type == "plain" or reward_type == "01sanity":
            self.reward_fn = lambda x: -x
        elif reward_type == "normalize_plain":
            self.reward_fn = lambda x: -x / 50.0 + 1.0
        elif reward_type == "mean_std_sanity":
            self.reward_fn = lambda x: -(x - self.energy_mean) / self.energy_std
        elif reward_type == "distance_factor":
            self.reward_fn = lambda x: -x / self.distance_factor / 5.0 + 1.0
        elif reward_type == "cubic":
            self.reward_fn = lambda x: -(x ** (1 / 2)) / 6.0 + 1.0

        if reward_type == "01sanity":
            self.do_return = False
        else:
            self.do_return = True

        self.state_type = state_type
        if (
            state_type == "d_at_f" or state_type == "d_rt_f"
        ):  # deformation-absolutetarget-flag (10+num_target*2+num_target)//# deformation-relativetarget-flag (10+num_target*2+num_target)
            self.state_fn = lambda: np.concatenate(
                [
                    self.curr_configuration,
                    self.polar_targets.flatten(),
                    self.reached_map,
                ]
            )  # / self.num_targets])
        elif (
            state_type == "d_tip_rt_f" or state_type == "d_tip_at_f"
        ):  # deformation-relativetarget-flag (10+2+num_target*2+num_target):
            self.state_fn = lambda: np.concatenate(
                [
                    self.curr_configuration,
                    self.tip,
                    self.polar_targets.flatten(),
                    self.reached_map,
                ]
            )  # / self.num_targets])
        elif (
            state_type == "d_tip_rt_e_f"
        ):  # deformation-relativetarget-flag (10+2+num_target*2+num_target):
            self.state_fn = lambda: np.concatenate(
                [
                    self.curr_configuration,
                    self.tip,
                    self.polar_targets.flatten(),
                    self.energy_cost_map,
                    self.reached_map,
                ]
            )  # / self.num_targets])
        elif (
            state_type == "tip_rt_f" or state_type == "tip_at_f"
        ):  # deformation-relativetarget-flag (2+num_target*2+num_target):
            self.state_fn = lambda: np.concatenate(
                [self.tip, self.polar_targets.flatten(), self.reached_map]
            )  # / self.num_targets])
        elif state_type == "attention":
            self.state_fn = lambda: np.concatenate(
                [
                    self.polar_targets.flatten(),
                    self.potential_activation.flatten(),
                    self.energy_cost_map,
                ]
            )

    def cal_return_energy(self, curr_activation):
        activation = np_target_config(generate_straight(eager=False))
        init_activation_tf = np_target_config(
            np.float32(curr_activation[np.newaxis, :])
        )
        _, diff_energy = tf_energy_difference(init_activation_tf, activation)
        return diff_energy

    def cal_activation_and_energy(self, target, init_activation):
        potential_state = np.concatenate([target, init_activation])
        # pred=self.nn.predict(potential_state[np.newaxis,:])
        pred = self.nn(potential_state[np.newaxis, :], training=False)
        potential_activation = pred[0]
        activation = np_target_config(pred)
        init_activation_tf = tf_target_config(
            tf.cast(init_activation[tf.newaxis, :], tf.float32)
        )

        _, diff_energy = tf_energy_difference(init_activation_tf, activation)
        return potential_activation, diff_energy, init_activation_tf

    def cal_energy_cost_map(self, curr_activation, remaining_targets):
        energy_cost_map = np.zeros(
            self.num_targets
        )  # self.state = np.float32(self.reachable_targets)
        potential_activation = np.zeros(
            (self.num_targets, self.num_components * self.num_muscles)
        )
        for index in remaining_targets:
            target = self.target_range[index][:2]
            (
                potential_activation[index, :],
                energy_cost_map[index],
                _,
            ) = self.cal_activation_and_energy(target, curr_activation)
        return potential_activation, energy_cost_map

    def cal_state(self):
        self.energy_cost_map = np.zeros(
            self.num_targets
        )  # self.state = np.float32(self.reachable_targets)
        self.potential_activation = np.zeros(
            (self.num_targets, self.num_components * self.num_muscles)
        )
        unreached_target_index = np.where(self.reached_map == 0)[0]

        for index in unreached_target_index:
            target = self.target_range[index][:2]
            (
                self.potential_activation[index, :],
                self.energy_cost_map[index],
                init_activation,
            ) = self.cal_activation_and_energy(target, self.activation)

        if self.state_type != "attention":

            init_activation = tf_target_config(
                np.float32(self.activation[np.newaxis, :])
            )

            distributed_activation = tf_distribute_activation(init_activation)
            sigma, kappa, _, _ = tf_activation_to_muscle_force(
                self.rod_list,
                distributed_activation,
                self.max_force_list,
                self.muscle_radius_ratio_list,
            )

            # _, _, _, _, _, _, arm_position, curvature, strain, _, _, _ = test_rod(
            #     tf.zeros_like(init_activation), init_activation, self.rod_list, self.max_force_list,
            #     self.muscle_radius_ratio_list)

            if self.state_type != "tip_rt_f" and self.state_type != "tip_at_f":
                self.curr_configuration = np.array(
                    [
                        np.mean(kappa[0, : self.section_node]) / self.kappa_factor,
                        np.mean(kappa[0, self.section_node : 2 * self.section_node])
                        / self.kappa_factor,
                        np.mean(kappa[0, 2 * self.section_node : 3 * self.section_node])
                        / self.kappa_factor,
                        np.mean(kappa[0, 3 * self.section_node : 4 * self.section_node])
                        / self.kappa_factor,
                        np.mean(kappa[0, 4 * self.section_node :]) / self.kappa_factor,
                        np.mean(sigma[2, : self.section_node + 1]) * self.sigma_factor,
                        np.mean(
                            sigma[2, self.section_node + 1 : 2 * self.section_node + 1]
                        )
                        * self.sigma_factor,
                        np.mean(
                            sigma[
                                2, 2 * self.section_node + 1 : 3 * self.section_node + 1
                            ]
                        )
                        * self.sigma_factor,
                        np.mean(
                            sigma[
                                2, 3 * self.section_node + 1 : 4 * self.section_node + 1
                            ]
                        )
                        * self.sigma_factor,
                        np.mean(sigma[2, 4 * self.section_node + 1 :])
                        * self.sigma_factor,
                    ]
                )

        self.tip = self.pos[:2]

        if (
            self.state_type != "d_at_f"
            and self.state_type != "d_tip_at_f"
            and self.state_type != "attention"
        ):
            self.make_polar_targets(tip=self.tip)

        self.state = self.state_fn()

    def cal_position(self):
        init_activation = tf_target_config(np.float32(self.activation[np.newaxis, :]))

        distributed_activation = tf_distribute_activation(init_activation)
        sigma, kappa, _, _ = tf_activation_to_muscle_force(
            self.rod_list,
            distributed_activation,
            self.max_force_list,
            self.muscle_radius_ratio_list,
        )

        _, _, _, _, _, _, arm_position, _, _, _, _, _ = test_rod(
            tf.zeros_like(init_activation),
            init_activation,
            self.rod_list,
            self.max_force_list,
            self.muscle_radius_ratio_list,
        )
        return arm_position

    def shift_in_x(self, amount):
        self.base += amount
        self.pos += amount

    def crawl(self):
        self.shift_in_x(
            self.crawl_amount
        )  # 0.1 matches the x_shift in generate_targets
        self.cal_state()

    def step(self, current_action):

        self.check_target_region(current_action)
        self.energy = self.energy_cost_map[current_action] if self.reach_flag else 100.0

        if self.render_plot:
            self.render(current_action)

        if self.do_return:
            self.reward = self.reward_fn(self.energy)
        else:
            self.reward = float(self.optimal_action[self.counter] == current_action)

        self.counter += 1
        done = self.check_done()
        if done and self.do_return:
            self.return_energy = self.cal_return_energy(self.activation).numpy()
            self.reward += self.reward_fn(self.return_energy)

        self.cal_state()

        return self.state, self.reward, done

    def check_missing_target(self):
        index_not_reached = np.where(self.reached_map == 0)[0]
        targets_not_reached = np.array(self.target_range)[index_not_reached]
        paseed_flag = len(np.where(targets_not_reached[:, 0] < self.base[0])[0]) > 0
        return paseed_flag

    def check_done(self):
        return self.targets_eaten == self.map_target

    def render(self, current_action):
        target_list = np.array(self.target_range)
        for i in range(self.num_targets):
            if self.reached_map[i] == 0:
                plt.plot(
                    target_list[i, 0], target_list[i, 1], "x", label="target%d" % i
                )
                plt.plot(
                    target_list[i, 0], target_list[i, 1], "ro", linewidth=14, alpha=0.2
                )
        v_origin = self.base[:2]
        plot_sector(v_origin)

        deg = 3
        x = [self.base[0], self.pos[0]]
        y = [self.base[1], self.pos[1]]
        plt.plot(x, y, "--", linewidth=6, alpha=0.3)
        # x_fit=np.linspace(x[0],x[1],5)
        # params = np.polynomial.legendre.legfit(x, y, deg)
        # y_fit = np.polynomial.legendre.legval(x_fit, params)
        # plt.plot(x_fit,y_fit)
        # plt.title("time step %d action %d, eatan %d, in hand %d" % (self.counter,current_action, self.targets_eaten,self.targets_in_hand))
        plt.title(
            "time step %d action %d, eatan %d"
            % (self.counter, current_action, self.targets_eaten)
        )

        plt.gca().set_aspect("equal")
        plt.gca().set_xlim([-0.5, 2.5])
        plt.gca().set_ylim([-0.2, 1.2])
        plt.savefig("frames/%03d_%03d" % (self.ep, self.counter))
        plt.close("all")

    def make_polar_targets(self, tip=[1.0, 0.0]):
        self.polar_targets = np.zeros((self.num_targets, 2))
        tip_x, tip_y = tip

        for i in range(self.map_target):
            x, y, _ = self.target_range[i]  # target
            pol_coor = cartesian2polar(x - tip_x, y - tip_y)
            pol_coor[1] /= np.pi
            self.polar_targets[i] = pol_coor

    def reset(self):
        self.counter = 0
        self.total_cost = 0.0

        self.make_polar_targets()

        self.activation = generate_straight(eager=False)[0]
        self.pos = np.array([1.0, 0.0, 0.0])
        self.action_list = []
        # self.targets_in_hand = 0
        self.targets_eaten = 0
        self.reached_map = np.zeros(self.num_targets, dtype=int)
        self.reached_map[self.map_target : self.num_targets] = 1
        self.cal_state()
        self.ep += 1
        self.reach_flag = False
        if self.reward_type == "distance_factor":
            dist_matrix = distance.cdist(self.target_range, self.target_range)
            self.distance_factor = np.sum(dist_matrix) / 2
        return self.state

    def check_target_region(self, current_action):

        self.reach_flag = False
        if (
            self.reached_map[current_action] == 0
        ):  # self.reachable_targets[current_action] == 1 and
            self.reach_flag = True
            self.activation = self.potential_activation[current_action]
            self.pos = self.target_range[current_action]
            self.targets_eaten += 1
            self.reached_map[current_action] = 1  # self.counter + 1

        return

    def read_targets(self, index, map_target):
        self.map_target = map_target
        self.target_range = self.target_sample[index]
        if self.map_target < self.num_targets:
            self.no_target = np.zeros((self.num_targets - self.map_target, 1))
            self.target_range = np.concatenate(
                [self.target_range, self.no_target], axis=0
            )

    def generate_n_targets(self, from_n_map):
        if from_n_map > 0:
            self.map_target = self.num_targets
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
            self.target_range = np.zeros((self.num_targets, 3))
            if self.state_type == "attention":
                self.map_target = self.num_targets
            else:
                self.map_target = np.random.randint(self.num_targets) + 1

            for i in range(self.map_target):
                self.target_range[i] = generate_targets_inside_workspace(self.polygon)
            self.make_polar_targets(tip=[0.0, 0.0])
            self.target_range = self.target_range[
                self.polar_targets[:, 1].argsort()[::-1]
            ]

    def plot_targets(self):
        for i in range(self.num_targets):
            plt.plot(
                self.target_range[i][0], self.target_range[i][1], "x", label="%d" % i
            )
        plt.legend()
        plt.savefig("test_target.png")
        plt.close("all")

    def cal_total_energy(self, actions):
        activation = generate_straight(eager=False)[
            0
        ]  # np.zeros(self.num_components * self.num_muscles)

        total_energy = 0
        activation_record = []
        energy_record = []
        for index in actions:
            target = self.target_range[index][:2]
            activation, energy, _ = self.cal_activation_and_energy(target, activation)
            total_energy += energy
            activation_record.append(activation)
            energy_record.append(energy.numpy())
        energy = self.cal_return_energy(activation)
        total_energy += energy
        energy_record.append(energy.numpy())
        return total_energy, activation_record, energy_record
