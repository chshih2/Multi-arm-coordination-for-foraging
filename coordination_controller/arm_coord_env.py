import os
import math

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from shapely.geometry import Polygon
from actuations.workspace import (
    generate_targets_inside_workspace,
    read_polygon_workspace,
    plot_polygon_workspace,
    check_point_in_polygon,
    check_point_in_polygon_with_boundary,
    y_rotation
)

from coordination_controller.arm_indiv_env import SingleArmEnv
from actuations.nnes_fn import generate_straight


@tf.function
def tf_muscle_activation_energy(activation):
    energy = tf.linalg.trace(tf.matmul(activation, tf.transpose(activation)))
    return energy


class CoordinationEnv:
    def __init__(
            self,
            num_targets,
            num_arms,
            obs_num_targets,
            ground_range,
            penalty_coeff,
            bonus_coeff,
            termination_step,
            state_type,
            reward_type,
            target_type,
            use_arena,
            arena_type,
            use_obstacle,
            flip_crawl_direction,
            food_weights,
            simulation_dir,
            multi_agent=True,
            render_plot=False,
            online=True,
            render_dir="frames/",
            **kwargs
    ):
        self.num_targets = num_targets
        self.obs_num_targets = obs_num_targets
        self.render_dir = render_dir
        self.render_plot = render_plot
        self.ep = 0
        (
            self.hull_pts,
            self.polygon,
            self.view_pts,
            self.view_polygon,
        ) = read_polygon_workspace(simulation_dir, view=True)
        self.ground_range = ground_range
        self.multi_agent = multi_agent

        self.termination_step = termination_step
        self.penalty_coeff = penalty_coeff
        self.online = online

        self.arm_list = []
        if num_arms == 2:
            angle_list = [0.0, -90.0]
        else:
            angle_list = np.linspace(0, -360, num_arms + 1)[:num_arms]
        for i in range(num_arms):
            self.arm_list.append(
                SingleArmEnv(
                    angle_list[i],
                    num_targets,
                    obs_num_targets,
                    simulation_dir,
                    bonus_coeff,
                    reward_type,
                    use_arena,
                    use_obstacle,
                    flip_crawl_direction,
                    food_weights,
                )
            )
        self.num_arms = num_arms
        self.use_arena = use_arena
        self.use_obstacle = use_obstacle
        self.obstacle = []
        self.obstacle_pts_list = []

        if state_type == "target-action-flag":
            self.state_fn = lambda i: np.concatenate(
                [self.state_list[i], self.action, self.flag_list]
            )
        elif state_type == "target-flag":
            self.state_fn = lambda i: np.concatenate(
                [self.state_list[i], self.flag_list]
            )
        elif state_type == "pos-target-action-flag":
            self.state_fn = lambda i: np.concatenate(
                [
                    [self.head[0], self.head[2]],
                    self.state_list[i],
                    self.action,
                    self.flag_list,
                ]
            )
        elif state_type == "target-wall-action-flag":
            self.state_fn = lambda i: np.concatenate(
                [self.state_list[i], self.wall_list, self.action, self.flag_list]
            )
        elif state_type == "pos-target-wall-action-flag":
            self.state_fn = lambda i: np.concatenate(
                [
                    [self.head[0], self.head[2]],
                    self.state_list[i],
                    self.wall_list,
                    self.action,
                    self.flag_list,
                ]
            )
        elif state_type == "dir-target-wall-action-flag":

            def generate_nearest_target(i):
                rel_target = self.target_range - self.head
                dis_target = (
                        rel_target[:, 0] ** 2
                        + rel_target[:, 2] ** 2
                        + 100 * self.reached_map
                )
                self.closest_target_index = np.argmin(dis_target)
                closest_target_dir = np.sign(rel_target[self.closest_target_index])
                return np.concatenate(
                    [
                        [closest_target_dir[0], closest_target_dir[2]],
                        self.state_list[i],
                        self.wall_list,
                        self.action,
                        self.flag_list,
                    ]
                )

            self.state_fn = generate_nearest_target
        elif state_type == "pos-near-target-wall-action-flag":

            def generate_nearest_target(i):
                rel_target = self.target_range - self.head
                dis_target = (
                        rel_target[:, 0] ** 2
                        + rel_target[:, 2] ** 2
                        + 100 * self.reached_map
                )
                self.closest_target_index = np.argmin(dis_target)
                closest_target_rel = rel_target[self.closest_target_index]
                return np.concatenate(
                    [
                        [self.head[0], self.head[2]],
                        [closest_target_rel[0], closest_target_rel[2]],
                        self.state_list[i],
                        self.wall_list,
                        self.action,
                        self.flag_list,
                    ]
                )

            self.state_fn = generate_nearest_target
        elif state_type == "pos-near-target-wall-action-flag-index":
            self.closest_target_index = np.zeros(num_arms, dtype=int)

            def generate_nearest_target(i):
                rel_target = self.target_range - self.head
                dis_target = (
                        rel_target[:, 0] ** 2
                        + rel_target[:, 2] ** 2
                        + 100 * self.reached_map
                )
                self.closest_target_index = np.argmin(dis_target)
                closest_target_rel = rel_target[self.closest_target_index]
                return np.concatenate(
                    [
                        [self.head[0], self.head[2]],
                        [closest_target_rel[0], closest_target_rel[2]],
                        self.state_list[i],
                        self.wall_list,
                        self.action,
                        self.flag_list,
                        [i],
                    ]
                )

            self.state_fn = generate_nearest_target
        elif state_type == "central":
            self.view_max_x = np.max(self.view_pts, axis=1)[0]

            def generate_nearest_target(i):
                rel_target = self.target_range - self.head  # (# of target ,3)
                dis_target = np.linalg.norm(rel_target, axis=1) + 100 * self.reached_map
                closest_target_rel = np.zeros((self.obs_num_targets, 3))
                sorted_dis_index = dis_target.argsort()
                d = 0
                while dis_target[sorted_dis_index[d]] < self.view_max_x:
                    d += 1
                self.closest_target_index = sorted_dis_index[:min(self.obs_num_targets, d)]
                closest_target_rel[:len(self.closest_target_index)] = rel_target[self.closest_target_index, :]
                return closest_target_rel  # .reshape((1, -1))

            self.state_fn = generate_nearest_target
        elif state_type == "pos-relnear-target-wall-action-flag-index":
            self.closest_target_index = np.zeros(num_arms, dtype=int)

            def generate_nearest_target(i):
                rel_target = self.target_range - self.head
                dis_target = (
                        rel_target[:, 0] ** 2
                        + rel_target[:, 2] ** 2
                        + 100 * self.reached_map
                )
                self.closest_target_index[i] = np.argmin(dis_target)
                closest_target_rel = rel_target[self.closest_target_index[i]]
                closest_target_rel = y_rotation(
                    closest_target_rel, -self.arm_list[i].angle
                )
                return np.concatenate(
                    [
                        [self.head[0], self.head[2]],
                        [closest_target_rel[0], closest_target_rel[2]],
                        self.state_list[i],
                        self.wall_list,
                        self.action,
                        self.flag_list,
                        [i],
                    ]
                )

            self.state_fn = generate_nearest_target
        elif state_type == "relnear-target-wall-action-flag-index":
            self.closest_target_index = np.zeros(num_arms, dtype=int)

            def generate_nearest_target(i):
                rel_target = self.target_range - self.head
                dis_target = (
                        rel_target[:, 0] ** 2
                        + rel_target[:, 2] ** 2
                        + 100 * self.reached_map
                )
                self.closest_target_index[i] = np.argmin(dis_target)
                closest_target_rel = rel_target[self.closest_target_index[i]]
                closest_target_rel = y_rotation(
                    closest_target_rel, -self.arm_list[i].angle
                )
                return np.concatenate(
                    [
                        [closest_target_rel[0], closest_target_rel[2]],
                        self.state_list[i],
                        self.wall_list,
                        self.action,
                        self.flag_list,
                        [i],
                    ]
                )

            self.state_fn = generate_nearest_target
        elif state_type == "relnear-target-action-flag-index":  # 2
            self.closest_target_index = np.zeros(num_arms, dtype=int)

            def generate_nearest_target(i):
                rel_target = self.target_range - self.head
                dis_target = (
                        rel_target[:, 0] ** 2
                        + rel_target[:, 2] ** 2
                        + 100 * self.reached_map
                )
                self.closest_target_index[i] = np.argmin(dis_target)
                closest_target_rel = rel_target[self.closest_target_index[i]]
                closest_target_rel = y_rotation(
                    closest_target_rel, -self.arm_list[i].angle
                )
                return np.concatenate(
                    [
                        [closest_target_rel[0], closest_target_rel[2]],
                        self.state_list[i],
                        self.action,
                        self.flag_list,
                        [i],
                    ]
                )

            self.state_fn = generate_nearest_target
        elif state_type == "relnear-target-action-index":
            self.closest_target_index = np.zeros(num_arms, dtype=int)

            def generate_nearest_target(i):
                rel_target = self.target_range - self.head
                dis_target = (
                        rel_target[:, 0] ** 2
                        + rel_target[:, 2] ** 2
                        + 100 * self.reached_map
                )
                self.closest_target_index[i] = np.argmin(dis_target)
                closest_target_rel = rel_target[self.closest_target_index[i]]
                closest_target_rel = y_rotation(
                    closest_target_rel, -self.arm_list[i].angle
                )
                return np.concatenate(
                    [
                        [closest_target_rel[0], closest_target_rel[2]],
                        self.state_list[i],
                        self.action,
                        [i],
                    ]
                )

            self.state_fn = generate_nearest_target
        elif state_type == "relnear-target-index":  # 2
            self.closest_target_index = np.zeros(num_arms, dtype=int)

            def generate_nearest_target(i):
                rel_target = self.target_range - self.head
                dis_target = (
                        rel_target[:, 0] ** 2
                        + rel_target[:, 2] ** 2
                        + 100 * self.reached_map
                )
                self.closest_target_index[i] = np.argmin(dis_target)
                closest_target_rel = rel_target[self.closest_target_index[i]]
                closest_target_rel = y_rotation(
                    closest_target_rel, -self.arm_list[i].angle
                )
                return np.concatenate(
                    [
                        [closest_target_rel[0], closest_target_rel[2]],
                        self.state_list[i],
                        [i],
                    ]
                )

            self.state_fn = generate_nearest_target
        elif state_type == "pos-relnear-target-index":  # 2
            self.closest_target_index = np.zeros(num_arms, dtype=int)

            def generate_nearest_target(i):
                rel_target = self.target_range - self.head
                dis_target = (
                        rel_target[:, 0] ** 2
                        + rel_target[:, 2] ** 2
                        + 100 * self.reached_map
                )
                self.closest_target_index[i] = np.argmin(dis_target)
                closest_target_rel = rel_target[self.closest_target_index[i]]
                closest_target_rel = y_rotation(
                    closest_target_rel, -self.arm_list[i].angle
                )
                return np.concatenate(
                    [
                        [self.head[0], self.head[2]],
                        [closest_target_rel[0], closest_target_rel[2]],
                        self.state_list[i],
                        [i],
                    ]
                )

            self.state_fn = generate_nearest_target
        else:
            print("not implemented state function")
            exit()
        self.arena_type = arena_type
        self.center_flag = 0.0
        if arena_type == "diagonal":
            self.arena_pts = np.array(
                [
                    [0.0, 2.0, 6.0, 6.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 6.0, 6.0, 2.0, 0.0],
                ]
            )
        elif arena_type == "square":
            self.arena_pts = np.array([[0.0, 6.0, 6.0, 0.0], [0.0, 0.0, 6.0, 6.0]])
        elif arena_type == "mid-square":
            self.arena_pts = np.array([[3.0, 3.0, -3.0, -3.0], [3.0, -3.0, -3.0, 3.0]])
            self.center_flag = 1.0
        else:
            print("not implemented arena type")
        self.arena = Polygon(Polygon(tuple(zip(self.arena_pts[0], self.arena_pts[1]))))

        if num_arms == 1:
            self.generate_n_targets = self.generate_n_targets_1arm
            self.use_arena = False
        elif target_type == "constrain":
            self.generate_n_targets = self.generate_n_targets_constrain
        elif target_type == "random":
            self.generate_n_targets = self.generate_n_targets_random
        elif target_type == "grid":
            if arena_type == "mid-square":
                self.generate_n_targets = self.generate_n_targets_gridmid
            else:
                self.generate_n_targets = self.generate_n_targets_grid
        elif target_type == "obstacle":
            self.generate_n_targets = self.generate_n_targets_obstacle
        else:
            print("not implemented target generator")
            exit()

    def step(self, current_action):
        if not self.multi_agent:
            current_action = [int(x) for x in format(current_action[0], '#08b')[-self.num_arms:]]

        for arm in self.arm_list:
            arm.reward = 0.0
        self.action = current_action
        self.reach_arm_list = np.where(np.isclose(current_action, 0))[0]
        self.crawl_arm_list = np.setdiff1d(range(self.num_arms), self.reach_arm_list)
        total_get_targets = 0
        for i in self.reach_arm_list:
            total_get_targets += self.arm_list[i].reach()
            self.arm_list[i].prev_reached_list = self.arm_list[i].new_reached_list
            for j in np.setdiff1d(range(self.num_arms), [i]):
                self.reached_map = self.arm_list[i].reached_map
                self.arm_list[j].update_reached_map(self.reached_map)
        no_target_flag = total_get_targets == 0
        if len(self.reach_arm_list) == self.num_arms and no_target_flag:
            for i in range(self.num_arms):
                self.arm_list[i].reward = -self.penalty_coeff
        total_crawl = np.sum(
            [
                self.arm_list[arm_index].crawl_amount
                for arm_index in self.crawl_arm_list
            ],
            axis=0,
        )
        if math.isclose(np.sum(abs(total_crawl)), 0.0, abs_tol=1e-5):
            self.do_crawl = False
            for arm_index in self.crawl_arm_list:
                self.arm_list[arm_index].reward = -self.penalty_coeff
            if no_target_flag:
                for arm_index in self.reach_arm_list:
                    self.arm_list[arm_index].reward = -self.penalty_coeff
        else:
            self.sum_crawl(self.crawl_arm_list, total_crawl)

        self.reward = np.mean([arm.reward for arm in self.arm_list])
        self.energy = np.sum([arm.energy for arm in self.arm_list])
        self.targets_eaten = np.sum([arm.targets_eaten for arm in self.arm_list])
        self.total_energy += self.energy
        self.counter += 1
        if self.render_plot:
            self.render(current_action)

        self.state = self.update_state()

        if self.render_plot:
            self.check_targets()

        done = self.check_done()  # or self.arm1.miss_done()  or self.arm2.miss_done()

        return self.state, self.reward, done, {}

    def check_done(self):
        good_done = self.targets_eaten == self.num_targets
        if good_done:
            self.reward += 10.0
        break_done = self.counter > self.termination_step
        if self.num_arms == 2:
            out_done = self.head[0] > self.right or self.head[2] > self.top
            done = good_done or break_done or out_done
        elif self.num_arms == 1:
            out_done = self.head[0] > self.right
            done = good_done or break_done or out_done
        else:
            done = good_done or break_done
        return done

    def update_state(self):
        self.state_list = []
        self.flag_list = []
        self.wall_list = []
        for i in range(self.num_arms):
            self.state_list.append(self.arm_list[i].cal_state())
            self.arm_list[i].cal_reachable_list()
            self.flag_list.append(int(len(self.arm_list[i].obs_list) > 0))
            self.wall_list.append(self.arm_list[i].wall)
        self.state = []
        for i in range(self.num_arms):
            # arm_state = self.state_fn(i)
            arm_state = np.concatenate([self.state_list[i], self.action, self.flag_list])
            self.state.append(arm_state)
        if not self.multi_agent:
            return self.multi_agent_state_conversion(np.array(self.state))
        else:
            return np.array(self.state)

    def multi_agent_state_conversion(self, state):
        arm_state = np.concatenate([state[i, 4:-(self.num_arms * 3 + 1)] for i in range(self.num_arms)])
        shared_state = np.concatenate([state[0, :4], state[0, -(self.num_arms * 3 + 1):-1]])
        state = np.concatenate([arm_state, shared_state])[np.newaxis, :]
        return state

    def sum_crawl(self, crawl_arm_list, total_crawl):
        num_crawl = len(crawl_arm_list)
        base_if_crawl = self.head + total_crawl
        self.do_crawl = True
        if self.use_arena:
            # after crawl, arm is still in the arena
            if not check_point_in_polygon_with_boundary(
                    base_if_crawl[0::2], self.arena
            ):
                self.do_crawl = False
            for obs in self.obstacle:
                if check_point_in_polygon_with_boundary(base_if_crawl[0::2], obs):
                    self.do_crawl = False

        if self.do_crawl:
            self.head = base_if_crawl
            for i in crawl_arm_list:
                self.arm_list[i].crawl(self.head)
            for j in np.setdiff1d(range(self.num_arms), crawl_arm_list):
                self.arm_list[j].update_head(self.head)
            for arm in self.arm_list:
                if check_point_in_polygon_with_boundary(arm.pos[0::2], self.arena):
                    arm.wall = 0
                    for obs in self.obstacle:
                        if check_point_in_polygon_with_boundary(arm.pos[0::2], obs):
                            arm.wall = 1
                            break
                else:
                    arm.wall = 1

        else:
            for i in crawl_arm_list:
                self.arm_list[i].reward = -self.penalty_coeff

    def arm_wait(self, arm, max_render, arm_index, head):
        for _ in range(arm.render_counter, max_render):
            arm.arm_render(-1, arm_index=arm_index, head=head)

    def render(self, current_action):
        head = self.prev_head
        if sum(current_action) == -self.num_arms:
            for i in range(self.num_arms):
                head = self.arm_list[i].arm_render(
                    current_action[i], arm_index=i, head=head
                )
        else:
            for i in self.reach_arm_list:
                self.arm_list[i].arm_render(current_action[i], arm_index=i, head=head)
                for j in np.setdiff1d(range(self.num_arms), [i]):
                    self.arm_wait(
                        self.arm_list[j],
                        self.arm_list[i].render_counter,
                        arm_index=j,
                        head=head,
                    )
            if self.do_crawl:
                for i in self.crawl_arm_list:
                    head = self.arm_list[i].arm_render(
                        current_action[i], arm_index=i, head=head
                    )
                    for j in np.setdiff1d(range(self.num_arms), [i]):
                        self.arm_wait(
                            self.arm_list[j],
                            self.arm_list[i].render_counter,
                            arm_index=j,
                            head=head,
                        )
        # print("arm head",[arm.head for arm in self.arm_list])
        # print("state arm head",[arm.log_state['log_head'][-1][0] for arm in self.arm_list])
        # print("state head",self.head)
        self.prev_head = head

    def reset(self):
        for i in range(self.num_arms):
            self.arm_list[i].arm_reset(self.target_range, self.obstacle_pts_list)

        self.head = np.array([0.0, 0.0, 0.0])
        self.prev_head = self.head
        self.counter = 0
        self.total_cost = 0.0
        self.do_crawl = False

        self.reward = 0.0
        self.action = [-1 for _ in range(self.num_arms)]
        self.reach_arm_list = []
        self.crawl_arm_list = []
        self.reached_map = np.zeros(self.num_targets, dtype=int)

        self.total_energy = 0.0
        self.state = self.update_state()
        self.ep += 1
        self.reach_flag = False
        self.targets_eaten = 0
        if self.render_plot:
            for arm in self.arm_list:
                arm.set_fig_ax(self.ep, self.render_dir)
            self.render(self.action)
            self.check_targets()

        self.right = max(self.target_range[:, 0])
        self.top = max(self.target_range[:, 2])

        return self.state

    def generate_n_targets_1arm(self):
        self.target_range = []
        for i in range(self.num_targets):
            flag = True
            while flag:
                target = generate_targets_inside_workspace(self.polygon)
                target[0] += np.random.rand() * self.ground_range
                flag = check_point_in_polygon(np.abs(target), self.polygon)
            self.target_range.append(target)
        self.target_range = np.array(self.target_range)

    def generate_n_targets_constrain(self):
        self.target_range = []
        crawl_shift_x, crawl_shift_y = 1, 1
        prev_target = (
                np.array([0.0, 0.0, 0.0]) + self.arm1.crawl_amount + self.arm2.crawl_amount
        )
        for i in range(self.num_targets):
            # to_upright_flag = False
            # while to_upright_flag == False:
            target = generate_targets_inside_workspace(self.polygon)
            flip = np.random.randint(2)
            if flip == 0:
                crawl_increment = np.random.randint(2)
                crawl = (
                        crawl_shift_x + crawl_increment
                )  # min(crawl_shift_x + crawl_increment, 21)
                target = y_rotation(target, self.arm2.angle)
                target += self.arm1.crawl_amount * crawl
                target[2] += crawl_shift_y * self.arm2.crawl_amount[2]
            elif flip == 1:
                crawl_increment = np.random.randint(2)
                crawl = (
                        crawl_shift_y + crawl_increment
                )  # min(crawl_shift_y + crawl_increment, 15)
                target += self.arm2.crawl_amount * crawl
                target[0] += crawl_shift_x * self.arm1.crawl_amount[0]
            prev_target = target
            if flip == 0:
                crawl_shift_x += crawl_increment
            else:
                crawl_shift_y += crawl_increment
            self.target_range.append(target)
        self.target_range = np.array(self.target_range)
        # self.check_targets(plot_arm=False)

    def generate_n_targets_random(self):
        self.target_range = []
        for i in range(self.num_targets):
            arena_flag = False
            while not arena_flag:
                target = generate_targets_inside_workspace(self.polygon)
                target[0] += np.random.rand() * self.ground_range
                flip = np.random.randint(2)
                if flip == 0:
                    crawl = np.random.randint(21)
                    target = y_rotation(target, self.arm2.angle)
                    target += self.arm1.crawl_amount * crawl
                    arena_flag = check_point_in_polygon(
                        np.array([target[0], np.abs(target[2])]), self.arena
                    )

                elif flip == 1:
                    crawl = np.random.randint(15)
                    target += self.arm2.crawl_amount * crawl
                    arena_flag = check_point_in_polygon(
                        np.array([target[0], np.abs(target[2])]), self.arena
                    )

            self.target_range.append(target)
        self.target_range = np.array(self.target_range)
        # self.check_targets(plot_arm=False)

    def generate_n_targets_grid(self):
        self.target_range = []
        for i in range(self.num_targets):
            arena_flag = False
            while not arena_flag:
                target = np.zeros(3)
                target[1] = (
                        np.random.rand() * 1.1122104966113382
                )  # max y value of the workspace
                for arm in self.arm_list[:2]:
                    target += np.random.randint(1, 25) * arm.flip * arm.crawl_amount
                arena_flag = check_point_in_polygon(
                    np.array([target[0], target[2]]), self.arena
                )

            self.target_range.append(target)
        self.target_range = np.array(self.target_range)
        # self.check_targets(plot_arm=False)

    def generate_n_targets_gridmid(self):
        self.target_range = []
        for i in range(self.num_targets):
            arena_flag = False
            while not arena_flag:
                target = np.zeros(3)
                target[1] = np.random.rand() * 1.1122104966113382
                shift_steps = np.random.randint(-11, 11, size=2)
                target[0] += shift_steps[0] * self.arm_list[0].crawl_amount[0]
                target[2] += shift_steps[1] * self.arm_list[0].crawl_amount[0]
                arena_flag = check_point_in_polygon(
                    np.array([target[0], target[2]]), self.arena
                )

            self.target_range.append(target)
        self.target_range = np.array(self.target_range)

    def check_targets(self, plot_arm=True):

        f = plt.figure(figsize=(16, 16))
        ax1 = f.add_subplot(self.num_arms + 1, 1, 1)
        # f = plt.figure(figsize=(8, 8))
        # ax1 = f.add_subplot( 1, 1, 1)
        if plot_arm:
            # assert np.all(self.arm_list[0].unreached_target_index == self.arm_list[1].unreached_target_index)
            plt.plot(self.arena_pts[0], self.arena_pts[1], color="black")
            for i in range(self.num_arms):
                ax1.plot(
                    self.target_range[self.arm_list[i].unreached_target_index, 0],
                    self.target_range[self.arm_list[i].unreached_target_index, 2],
                    "rx",
                )
                plot_index = self.arm_list[
                    i
                ].obs_list  # np.intersect1d(self.arm_list[i].observe_index, self.arm_list[i].unreached_target_index)
                ax1.plot(
                    self.target_range[plot_index, 0],
                    self.target_range[plot_index, 2],
                    "bo",
                    alpha=0.5,
                )
            ax1.plot(
                self.target_range[self.closest_target_index, 0],
                self.target_range[self.closest_target_index, 2],
                "ro",
                alpha=0.5,
            )
        else:
            plt.plot(self.arena_pts[0], self.arena_pts[1], color="black")
            ax1.plot(self.target_range[:, 0], self.target_range[:, 2], "rx")
        if self.use_obstacle:
            # line obstacle
            # plt.plot([self.obstacle[:, 0,0],self.obstacle[:, 0,1]], [self.obstacle[:, 1,0],self.obstacle[:, 1,1]], "--")
            # rectangel obstacle
            for obs in self.obstacle_pts_list:
                plot_polygon_workspace(obs)

        # for i in range(25):
        #     ax1.hlines(i * self.arm_list[0].crawl_amount, 0, 6, linestyles='dotted')
        #     # ax1.hlines(i * self.arm1.crawl_amount, 0, 6, linestyles='dotted')
        # for i in range(28):
        #     ax1.vlines(i * self.arm_list[1].crawl_amount, 0, 6, linestyles='dotted')
        #     # ax1.vlines(i * self.arm2.crawl_amount, 0, 6, linestyles='dotted')
        if plot_arm:
            for i in range(self.num_arms):
                ax1.plot(
                    [self.arm_list[i].base[0], self.arm_list[i].pos[0]],
                    [self.arm_list[i].base[2], self.arm_list[i].pos[2]],
                    ":",
                    color="C%d" % i,
                    alpha=0.5,
                    linewidth=4,
                )

        if plot_arm:
            # arm1_reward = self.arm_list[0].reward
            # arm2_reward = self.arm_list[1].reward
            # arm3_reward = self.arm_list[2].reward
            # arm4_reward = self.arm_list[3].reward
            title_string = (
                    "action"
                    + str(self.action)
                    + "flag"
                    + str(self.flag_list)
                    + str("wall")
                    + str(self.wall_list)
            )
            # title_string = title_string + "\nreward %.2f arm1 %.2f arm2 %.2f arm3 %.2f arm4 %.2f\nstate " % (
            # self.reward, arm1_reward, arm2_reward, arm3_reward, arm4_reward)+np.array2string(self.state[:,2:6], formatter={'float_kind':lambda x: "%.2f" % x})
            # title_string = title_string +"\nclosest target index "+np.array2string(self.closest_target_index)
            # title_string = title_string +"\nclosest target "+np.array2string(self.target_range[self.closest_target_index], formatter={'float_kind':lambda x: "%.2f" % x})
            ax1.set_title(title_string)
        if self.arena_type == "mid-square":
            ax1.set_xlim(-3.2, 3.2)
            ax1.set_ylim(-3.2, 3.2)
        else:
            ax1.set_xlim(-0.5, 7.0)
            ax1.set_ylim(-0.5, 7.0)

        ax1.set_aspect("equal", "box")

        if plot_arm:
            for i in range(self.num_arms):
                ax = f.add_subplot(self.num_arms + 1, 1, i + 2)
                self.plot_arm_perspective(ax, self.arm_list[i], color="C%d" % i)

        if plot_arm:
            f.tight_layout()
            plt.savefig(
                os.path.join(
                    self.render_dir, "eps%03d_target%d.png" % (self.ep, self.counter)
                )
            )
            plt.close("all")
        else:
            plt.show()
            exit()

    def plot_arm_perspective(self, ax, arm, color):
        target = self.target_range.copy()
        for i in range(len(target)):
            target[i] = y_rotation(target[i], -arm.angle)
        plot_index = (
            arm.obs_list
        )  # np.intersect1d(arm.observe_index, arm.unreached_target_index)
        ax.plot(
            target[arm.unreached_target_index, 0],
            target[arm.unreached_target_index, 1],
            "x",
            alpha=0.3,
        )
        ax.plot(
            target[plot_index, 0], target[plot_index, 1], "o", color=color, alpha=0.5
        )

        pos = y_rotation(arm.pos, -arm.angle)
        base = y_rotation(arm.base, -arm.angle)
        plot_polygon_workspace(self.hull_pts, x_shift=base[0], ax=ax)
        plot_polygon_workspace(self.view_pts, x_shift=base[0], ax=ax)
        ax.plot(
            [base[0], pos[0]],
            [base[1], pos[1]],
            color="purple",
            alpha=0.5,
            linewidth=12,
        )
        ax.set_xlim(-0.2, 7.0)
        ax.set_ylim(-0.2, 1.3)
        ax.set_aspect("equal", "box")

    def cal_energy_for_redundant_step(self):
        target = generate_targets_inside_workspace(self.polygon)
        target = target[:2]
        curr_activation = generate_straight(eager=False)[0]
        _, energy_cost, _ = self.arm_list[0].agent.env.cal_activation_and_energy(
            target, curr_activation
        )
        return energy_cost

    # Class constructor

    @classmethod
    def make_env(
            cls,
            render_plot,
            penalty_coeff,
            bonus_coeff,
            termination_step,
            state_type,
            reward_type,
            target_type,
            arena_type,
            use_arena,
            use_obstacle,
            flip_crawl_direction,
            food_weights,
            num_targets,
            obs_num_targets,
            num_arms,
            simulation_dir="nnes_model/",
            **kwargs,
    ):
        # num_targets = 10
        ground_range = 8.0
        online = True
        return cls(
            num_targets=num_targets,
            num_arms=num_arms,
            obs_num_targets=obs_num_targets,
            ground_range=ground_range,
            penalty_coeff=penalty_coeff,
            bonus_coeff=bonus_coeff,
            termination_step=termination_step,
            state_type=state_type,
            reward_type=reward_type,
            target_type=target_type,
            use_arena=use_arena,
            arena_type=arena_type,
            use_obstacle=use_obstacle,
            flip_crawl_direction=flip_crawl_direction,
            food_weights=food_weights,
            simulation_dir=simulation_dir,
            render_plot=render_plot,
            online=online,
            **kwargs
        )
