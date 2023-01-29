import numpy as np
import time


class GreedyPath:
    def __init__(self, env, test_sample_file):
        self.env = env
        self.test_sample_file = test_sample_file

    def record_greedy_solution(self, save_file):
        from tqdm import tqdm
        # self.env.generate_n_targets(from_n_map=0)
        self.env.map_target = self.env.num_targets
        target_sample = np.load(self.test_sample_file)
        greedy_energy_record = []
        greedy_action_record = []
        greedy_duration_record = []
        greedy_energy_history_record = []
        greedy2_energy_record = []
        greedy2_action_record = []
        greedy2_duration_record = []
        greedy2_energy_history_record = []

        for eps in tqdm(range(100)):
            self.env.target_range = target_sample[eps]
            start = time.time()
            greedy2_actions, greedy2_energy = self.greedy2_energy()
            greedy2_duration = time.time() - start

            start = time.time()
            greedy_actions, greedy_energy = self.greedy_energy()
            greedy_duration = time.time() - start

            # greedy_energy, _, greedy_energy_history = self.env.cal_total_energy(greedy_actions)
            # greedy2_energy, _, greedy2_energy_history = self.env.cal_total_energy(greedy2_actions)

            greedy_energy_record.append(greedy_energy)
            greedy_action_record.append(greedy_actions)
            greedy_duration_record.append(greedy_duration)
            # greedy_energy_history_record.append(greedy_energy_history)

            greedy2_energy_record.append(greedy2_energy)
            greedy2_action_record.append(greedy2_actions)
            greedy2_duration_record.append(greedy2_duration)
            # greedy2_energy_history_record.append(greedy2_energy_history)

        print("greedy duration", np.mean(greedy_duration_record))
        print("greedy2 duration", np.mean(greedy2_duration_record))

        np.savez(save_file,
                 greedy_energy=greedy_energy_record,
                 greedy_action=greedy_action_record,
                 greedy_duration=greedy_duration_record,
                 # greedy_energy_history=greedy_energy_history_record,
                 greedy2_energy=greedy2_energy_record,
                 greedy2_action=greedy2_action_record,
                 greedy2_duration=greedy2_duration_record,
                 # greedy2_energy_history=greedy2_energy_history_record,
                 )

    def greedy_energy(self):
        cur_state, done, rewards = self.env.reset(), False, 0
        greedy_actions = []
        total_energy = 0
        for _ in range(self.env.map_target):
            greedy_state = self.env.energy_cost_map[:self.env.map_target].copy()
            greedy_state[greedy_actions] = np.inf
            action = np.argmin(greedy_state)
            greedy_actions.append(action)
            total_energy += greedy_state[action]
            _, _, _ = self.env.step(action)
        total_energy += self.env.cal_return_energy(self.env.activation)
        return greedy_actions, total_energy

    # def greedy2_energy(self):
    #     _, _, _ = self.env.reset(), False, 0
    #     greedy_actions = []
    #     total_energy=0
    #     penalty_matrix=np.eye(self.env.num_targets)*1000
    #     for _ in range(self.env.num_targets-1):
    #         greedy_state = self.env.energy_cost_map.copy()
    #         greedy_state[greedy_actions] = np.inf
    #         look_forward_matrix = []
    #         for cur_action in range(self.env.num_targets):  # look_forward
    #             potential_activation, _, _ = self.env.cal_activation_and_energy(self.env.target_range[cur_action][:2],
    #                                                                             self.env.activation)
    #             greedy2_state = []
    #             for next_action in range(self.env.num_targets):
    #                 next_potential_activation, energy_cost, _ = self.env.cal_activation_and_energy(
    #                     self.env.target_range[next_action][:2],
    #                     potential_activation)
    #                 # energy_cost+=energy_cost2
    #                 if self.env.num_targets - len(greedy_actions) == 2:
    #                     return_energy = self.env.cal_return_energy(next_potential_activation)
    #                     energy_cost+=return_energy
    #                 greedy2_state.append(energy_cost.numpy())
    #             look_forward_matrix.append(greedy2_state)
    #         look_forward_matrix = np.array(look_forward_matrix)
    #         look_forward_matrix[:,greedy_actions]=np.inf
    #         two_step_cost = greedy_state[:, np.newaxis] + look_forward_matrix + penalty_matrix
    #
    #         action, go_to = np.unravel_index(np.argmin(two_step_cost, axis=None), two_step_cost.shape)
    #         # total_energy+=
    #         total_energy+= two_step_cost[action,go_to] if ((self.env.num_targets - len(greedy_actions)) == 2) else greedy_state[action]
    #         greedy_actions.append(action)
    #         _, _, _ = self.env.step(action)
    #     # if self.env.num_targets - len(greedy_actions) == 1:
    #     greedy_actions.append(np.where(self.env.reached_map==0)[0][0])
    #     return greedy_actions, total_energy

    def greedy2_energy(self):
        _, _, _ = self.env.reset(), False, 0
        greedy_actions = []
        total_energy = 0
        penalty_matrix = np.eye(self.env.num_targets) * 1000
        for _ in range(self.env.num_targets - 1):
            greedy_state = self.env.energy_cost_map.copy()
            greedy_state[greedy_actions] = np.inf
            look_forward_matrix = np.zeros((self.env.num_targets, self.env.num_targets))
            unreached_index = np.setdiff1d(range(self.env.num_targets), greedy_actions)
            for cur_action in unreached_index:  # range(self.env.num_targets):  # look_forward
                potential_activation, _, _ = self.env.cal_activation_and_energy(self.env.target_range[cur_action][:2],
                                                                                self.env.activation)
                greedy2_state = np.zeros(self.env.num_targets)
                next_action_list = np.setdiff1d(unreached_index, [cur_action])
                for next_action in next_action_list:  # range(self.env.num_targets):
                    next_potential_activation, energy_cost, _ = self.env.cal_activation_and_energy(
                        self.env.target_range[next_action][:2],
                        potential_activation)
                    # energy_cost+=energy_cost2
                    if self.env.num_targets - len(greedy_actions) == 2:
                        return_energy = self.env.cal_return_energy(next_potential_activation)
                        energy_cost += return_energy
                    greedy2_state[next_action] = energy_cost.numpy()
                look_forward_matrix[cur_action] = greedy2_state
            # look_forward_matrix = np.array(look_forward_matrix)
            look_forward_matrix[:, greedy_actions] = np.inf
            two_step_cost = greedy_state[:, np.newaxis] + look_forward_matrix + penalty_matrix

            action, go_to = np.unravel_index(np.argmin(two_step_cost, axis=None), two_step_cost.shape)
            # total_energy+=
            total_energy += two_step_cost[action, go_to] if ((self.env.num_targets - len(greedy_actions)) == 2) else \
                greedy_state[action]
            greedy_actions.append(action)
            _, _, _ = self.env.step(action)
        # if self.env.num_targets - len(greedy_actions) == 1:
        greedy_actions.append(np.where(self.env.reached_map == 0)[0][0])
        return greedy_actions, total_energy
