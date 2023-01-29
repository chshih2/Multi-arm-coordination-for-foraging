import tensorflow as tf
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
import datetime
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
import tensorflow_probability as tfp
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from upper_controller._legacy_switch_env import *
import random
import copy


# import gym

def get_gaes(rewards, dones, values, next_values, gamma, lamda, normalize):
    deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
    deltas = np.stack(deltas)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

    target = gaes + values
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    target = target.numpy()
    target = (target - target.mean()) / (target.std() + 1e-8)
    return gaes, target


class PPO(Model):
    def __init__(self, action_size):
        super(PPO, self).__init__()
        self.layer1 = tf.keras.layers.Dense(32, activation='relu')
        self.layer2 = tf.keras.layers.Dense(32, activation='relu')
        self.layer_a1 = tf.keras.layers.Dense(16, activation='relu')
        self.layer_c1 = tf.keras.layers.Dense(16, activation='relu')
        self.logits = tf.keras.layers.Dense(action_size, activation='linear')  # activation='softmax')
        self.value = tf.keras.layers.Dense(1, activation='linear')

    @tf.function
    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)

        layer_a1 = self.layer_a1(layer2)
        logits = self.logits(layer_a1)

        layer_c1 = self.layer_c1(layer2)
        value = self.value(layer_c1)

        return logits, value


class Policy(Model):
    def __init__(self, action_size):
        super(Policy, self).__init__()
        self.layer_pi1 = tf.keras.layers.Dense(64, activation='relu')
        self.layer_pi2 = tf.keras.layers.Dense(64, activation='relu')
        self.layer_pi3 = tf.keras.layers.Dense(64, activation='relu')
        self.output_logits = tf.keras.layers.Dense(action_size, activation='linear')  # activation='softmax')

    @tf.function
    def call(self, state):
        layer_pi1 = self.layer_pi1(state)
        layer_pi2 = self.layer_pi2(layer_pi1)
        layer_pi3 = self.layer_pi3(layer_pi2)
        # layer_pi3 = self.layer_pi3(layer_pi1)
        logits = self.output_logits(layer_pi3)

        return logits


class Value(Model):
    def __init__(self):
        super(Value, self).__init__()
        self.layer_v1 = tf.keras.layers.Dense(64, activation='relu')
        self.layer_v2 = tf.keras.layers.Dense(64, activation='relu')
        self.layer_v3 = tf.keras.layers.Dense(64, activation='relu')
        self.output_value = tf.keras.layers.Dense(1, activation='linear')

    @tf.function
    def call(self, state):
        layer_v1 = self.layer_v1(state)
        layer_v2 = self.layer_v2(layer_v1)
        #
        layer_v3 = self.layer_v3(layer_v2)
        # layer_v3 = self.layer_v3(layer_v1)
        value = self.output_value(layer_v3)

        return value


def TransferPPO(state_shape):
    # layer1 = tf.keras.layers.Dense(128, activation='relu')
    # layer2 = tf.keras.layers.Dense(128, activation='relu')
    # layer_a1 = tf.keras.layers.Dense(128, activation='relu')
    layer_a1 = tf.keras.layers.Dense(64, activation='relu')
    supervised_dir = "supervised_model128128128/"
    base_model = tf.keras.models.load_model(supervised_dir + "/supervised_model.h5")
    # base_model.trainable = False

    layer_c1 = tf.keras.layers.Dense(64, activation='relu')
    logits = tf.keras.layers.Dense(5, )  # activation='softmax')
    value = tf.keras.layers.Dense(1)

    state = tf.keras.layers.Input(shape=state_shape)
    layer = base_model(state)
    # layer_a1 = layer_a1(layer)
    logits = logits(layer)
    layer_c1 = layer_c1(layer)
    value = value(layer_c1)

    # layer1 = layer1(state)
    # layer2 = layer2(layer1)
    # layer_a1 = layer_a1(layer2)
    # logits = logits(layer_a1)
    # layer_c1 = layer_c1(layer2)
    # value = value(layer_c1)
    model = tf.keras.models.Model(inputs=state, outputs=[logits, value])
    return model


# @tf.function
# def call(self, state):
#     layer1 = self.layer1(state)
#     layer2 = self.layer2(layer1)
#
#     layer_a1 = self.layer_a1(layer2)
#     logits = self.logits(layer_a1)
#
#     layer_c1 = self.layer_c1(layer2)
#     value = self.value(layer_c1)
#
#     return logits, value
class Agent:
    def __init__(self, action_size=5, lr=0.001, eps=0.2, batch_size=128,
                 rollout=1024, separate_model=False):
        self.lr = lr
        self.gamma = 0.99
        self.lamda = 0.96

        self.rollout = rollout
        self.batch_size = batch_size
        self.action_size = action_size
        self.epoch = 3
        self.ppo_eps = eps
        self.normalize = True
        self.separate_model = separate_model
        if separate_model:
            self.policy_model = Policy(self.action_size)
            self.value_energy_model = Value()
        else:
            self.ppo = PPO(self.action_size)

        self.opt = optimizers.RMSprop(self.lr)  # optimizers.Adam(lr=self.lr, )

    # @tf.function
    def get_action(self, state, greedy=False, random=False, greedy_policy=False):

        state = tf.convert_to_tensor(state[tf.newaxis, :], dtype=tf.float32)

        if greedy_policy:
            policy, _ = self.target_ppo(state)
            # flag = state[:, -self.action_size:]
            # policy = masking(policy, flag)
            policy = tf.nn.softmax(policy)
            action = tf.math.argmax(policy[0])
            return action
        else:
            if random:
                action = np.random.randint(0, 2)
                return action
            else:
                if self.separate_model:
                    policy = self.policy_model(state)
                else:
                    policy, _ = self.ppo(state)
                # policy = masking(policy, flag)
                policy = tf.nn.softmax(policy)
                if greedy:
                    action = tf.math.argmax(policy[0])
                    return action
                else:
                    dist = tfp.distributions.Categorical(probs=policy)
                    action = dist.sample()
        return action[0]

    # @tf.function
    def update(self, state, next_state, reward, done, action, baseline_value=[], dense_reward=[]):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        if self.separate_model:
            old_policy = self.policy_model(state)
            current_value = self.value_energy_model(state)
        else:
            old_policy, current_value = self.ppo(state)

        # flag = state[:, -self.action_size:]
        # old_policy = masking(old_policy, flag)
        old_policy = tf.nn.softmax(old_policy)

        if baseline_value == []:
            if self.separate_model:
                next_value = self.value_energy_model(state)
            else:
                _, next_value = self.ppo(tf.convert_to_tensor(next_state, dtype=tf.float32))
        else:
            next_value = tf.reshape(baseline_value, (self.rollout, 1))

        current_value, next_value = tf.squeeze(current_value), tf.squeeze(next_value)

        adv, target = get_gaes(
            rewards=np.array(reward),
            dones=np.array(done),
            values=current_value,
            next_values=next_value,
            gamma=self.gamma,
            lamda=self.lamda,
            normalize=self.normalize)
        return self.learn(adv, target, old_policy, state, done, action, dense_reward)

    # @tf.function
    def learn(self, adv, target, old_policy, state, done, action, dense_reward):
        loss_p = []
        loss_v = []
        loss_e = []
        loss_H = []
        loss_Total = []
        loss_Pred = []
        grad_norms = []
        grad_norms_v = []
        for _ in range(self.epoch):
            len_sample = len(state)
            # sample_range = np.arange(len_sample)  # np.arange(self.rollout)
            sample_range = np.arange(self.rollout)
            np.random.shuffle(sample_range)
            # for i in range(len_sample // self.batch_size):  # range(self.rollout // self.batch_size):
            for i in range(self.rollout // self.batch_size):
                sample_idx = sample_range[self.batch_size * i:self.batch_size * (i + 1)]

                batch_state = [state[i] for i in sample_idx]
                # batch_done = [done[i] for i in sample_idx]
                batch_action = [action[i] for i in sample_idx]
                batch_target = [target[i] for i in sample_idx]
                batch_adv = [adv[i] for i in sample_idx]
                batch_old_policy = [old_policy[i] for i in sample_idx]
                train_adv = tf.convert_to_tensor(batch_adv, dtype=tf.float32)
                train_target = tf.convert_to_tensor(batch_target, dtype=tf.float32)
                train_action = tf.convert_to_tensor(batch_action, dtype=tf.int32)
                train_old_policy = tf.convert_to_tensor(batch_old_policy, dtype=tf.float32)

                if self.separate_model:
                    # batch_reward = [dense_reward[i] for i in sample_idx]
                    policy_variable = self.policy_model.trainable_variables
                    value_energy_variable = self.value_energy_model.trainable_variables
                    batch_state = tf.convert_to_tensor(batch_state, dtype=tf.float32)
                    with tf.GradientTape() as tape:
                        tape.watch(policy_variable)
                        train_policy = self.policy_model(batch_state)
                        pi_loss, entropy = self.cal_pi_loss_entropy(batch_state, train_policy, train_action,
                                                                    train_old_policy, train_adv)
                        policy_loss = pi_loss + entropy * 0.1

                    policy_grads = tape.gradient(policy_loss, policy_variable)
                    policy_grads, policy_grad_norm = tf.clip_by_global_norm(policy_grads, 1000)
                    self.opt.apply_gradients(zip(policy_grads, policy_variable))

                    with tf.GradientTape() as tape:
                        tape.watch(value_energy_variable)
                        train_current_value = self.value_energy_model(batch_state)
                        train_current_value = tf.squeeze(train_current_value)
                        value_loss = tf.reduce_mean(tf.square(train_target - train_current_value))
                        # energy_loss = tf.reduce_mean(tf.square(batch_reward - pred_energy))
                        value_energy_loss = value_loss * self.value_coeff  # + energy_loss

                    value_energy_grads = tape.gradient(value_energy_loss, value_energy_variable)
                    value_energy_grads, value_energy_grad_norm = tf.clip_by_global_norm(value_energy_grads, 1000)
                    self.opt.apply_gradients(zip(value_energy_grads, value_energy_variable))

                    loss_p.append(pi_loss)
                    loss_v.append(value_loss)
                    loss_H.append(entropy)
                    # loss_e.append(energy_loss)
                    loss_Total.append(policy_loss)
                    loss_Pred.append(value_energy_loss)
                    grad_norms.append(policy_grad_norm)
                    grad_norms_v.append(value_energy_grad_norm)
                else:
                    ppo_variable = self.ppo.trainable_variables
                    with tf.GradientTape() as tape:
                        tape.watch(ppo_variable)
                        batch_state = tf.convert_to_tensor(batch_state, dtype=tf.float32)
                        train_policy, train_current_value = self.ppo(batch_state)
                        pi_loss, entropy = self.cal_pi_loss_entropy(batch_state, train_policy, train_action,
                                                                    train_old_policy, train_adv)
                        train_current_value = tf.squeeze(train_current_value)
                        value_loss = tf.reduce_mean(tf.square(train_target - train_current_value))
                        total_loss = (pi_loss - entropy * 1e-3) + value_loss * self.value_coeff
                    grads = tape.gradient(total_loss, ppo_variable)
                    grads, grad_norm = tf.clip_by_global_norm(grads, 100)
                    self.opt.apply_gradients(zip(grads, ppo_variable))
                    loss_p.append(pi_loss)
                    loss_v.append(value_loss)
                    loss_H.append(entropy)
                    loss_Total.append(total_loss)
                    grad_norms.append(grad_norm)

        return loss_p, loss_v, loss_H, loss_e, loss_Total, loss_Pred, grad_norms, grad_norms_v

    # @tf.function
    def cal_pi_loss_entropy(self, batch_state, train_policy, train_action, train_old_policy, train_adv):
        # flag = batch_state[:, -self.action_size:]
        # train_policy = masking(train_policy, flag)
        train_policy = tf.nn.softmax(train_policy)

        # avail_action_count = tf.reduce_sum(tf.cast(tf.equal(flag, 0), tf.float32), axis=-1)
        # H = -tf.reduce_sum(train_policy * tf.math.log(train_policy + 1e-8), axis=-1)
        # entropy = tf.reduce_mean(H / avail_action_count)
        entropy = tf.reduce_mean(-train_policy * tf.math.log(train_policy + 1e-8))

        onehot_action = tf.one_hot(train_action, self.action_size)
        selected_prob = tf.reduce_sum(train_policy * onehot_action, axis=1)
        selected_old_prob = tf.reduce_sum(train_old_policy * onehot_action, axis=1)
        logpi = tf.math.log(selected_prob + 1e-8)
        logoldpi = tf.math.log(selected_old_prob + 1e-8)

        ratio = tf.exp(logpi - logoldpi)
        clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1 - self.ppo_eps,
                                         clip_value_max=1 + self.ppo_eps)
        minimum = tf.minimum(tf.multiply(train_adv, clipped_ratio), tf.multiply(train_adv, ratio))
        pi_loss = -tf.reduce_mean(minimum)
        return pi_loss, entropy

    def run(self, n_map, num_targets, obs_num_targets, total_epochs, penalty_coeff, bonus_coeff, termination_step,
            state_type,
            load=False):
        self.value_coeff = 10.0  # 0.5#5e-3#1.0
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        parameter_stamp = "t%d-target%d-epoch%d-pen%f-bon%f-lr%f-eps%.3f-rollout%d-batch%d-valu%f-" % (
            termination_step, num_targets, total_epochs, penalty_coeff, bonus_coeff, self.lr, self.ppo_eps,
            self.rollout,
            self.batch_size, self.value_coeff) + state_type
        train_log_dir = 'logs_%dtarget/' % num_targets + parameter_stamp + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)
        if load:
            self.ppo = tf.keras.models.load_model("logs/ppo.h5py", compile=False)

        # Environment configuration
        env = CoordinationEnv.make_env(render_plot=False, penalty_coeff=penalty_coeff, bonus_coeff=bonus_coeff,
                       termination_step=termination_step, state_type=state_type, num_targets=num_targets,
                       obs_num_targets=obs_num_targets)

        env.generate_n_targets(from_n_map=n_map)

        # target_range = []
        # target_range.append([np.array([2.0, 0.5, 0.0]), np.array([2.0, -0.5, 0.0])])
        # target_range.append([np.array([1.9, 0.5, 0.0]), np.array([2.1, 0.4, 0.0])])
        # target_range.append([np.array([2.0, 0.5, 0.0]), np.array([1.0, -0.5, 0.0])])
        # # env.target_range=target_range[np.random.randint(3)]
        # env.target_range=target_range[2]

        # _, [_, optimal_action, _] = self.cal_optimal_tour(env)
        state = env.reset()
        # episode = 0
        early_terminate_count = 0
        score = 0
        optimal_actions = [1, 1, 1, 1, 0]
        for epoch in range(total_epochs):
            state_list, next_state_list = [], []
            reward_list, done_list, action_list = [], [], []
            greedy_baseline_list = []
            dense_reward_list = []
            epoch_reward = []
            energy_list = []
            _epoch_reward = 0.0
            epoch_hit = 0.0
            episode = 0
            for _ in range(self.rollout):
                # while episode < 50:

                # reached_target_index = np.where(env.reached_map != 0)[0]
                action = self.get_action(state, greedy=False, random=False)
                # action=int(input("input"))

                # assert action == 0 or action == 1
                next_state, reward, done = env.step(action)
                # print("eat %d targets, get reward %f"%(env.targets_eaten,reward))
                # reward *= 0.1
                # print(action,optimal_actions,env.counter)

                # reward=1 if action==optimal_actions[env.counter-1] else 0.0
                # if env.counter>5:
                #     reward=0.0

                score += reward
                # if done:
                #     reward = score  # + env.targets_eaten
                # else:
                #     reward = -0.0
                # if done and env.counter==5 and np.isclose(score,5.0):
                #     # print("got it")
                #     reward+=5
                # else:
                #     reward=0

                state_list.append(state)
                next_state_list.append(next_state)
                reward_list.append(reward)
                done_list.append(done)
                action_list.append(action)

                state = next_state

                if done:
                    # print(reward)
                    epoch_reward.append(score)
                    energy_list.append(env.total_energy)
                    _epoch_reward += score
                    epoch_hit += env.targets_eaten
                    env.generate_n_targets(from_n_map=n_map)
                    # env.target_range = target_range[np.random.randint(3)]
                    # env.target_range = target_range[2]

                    state = env.reset()
                    episode += 1
                    score = 0

            loss_p, loss_v, loss_H, loss_e, loss_Total, loss_Pred, grad_norms, grad_norms_v = self.update(
                state=state_list, next_state=next_state_list,
                reward=reward_list, done=done_list, action=action_list, dense_reward=dense_reward_list,
                baseline_value=greedy_baseline_list)

            # val_energy = self.val_in_run(env)
            val_energy = 0

            print(epoch, "epoch reward%.2f" % np.mean(epoch_reward),  # "std%.2f" % np.std(epoch_reward),
                  # "total hit! %.2f" % (epoch_hit),
                  "energy", np.mean(energy_list),
                  "average hit! %.2f" % (epoch_hit / len(epoch_reward)),
                  "evaluate %.2f" %
                  val_energy)

            with summary_writer.as_default():
                tf.summary.scalar('Main/episode_reward', np.mean(epoch_reward), step=epoch)
                tf.summary.scalar('Main/episode_energy', np.mean(energy_list), step=epoch)
                tf.summary.scalar('Main/episode_targets', epoch_hit / len(epoch_reward), step=epoch)
                tf.summary.scalar('Loss/policy', tf.reduce_mean(loss_p), step=epoch)
                tf.summary.scalar('Loss/value', tf.reduce_mean(loss_v), step=epoch)
                # tf.summary.scalar('Loss/energy', tf.reduce_mean(loss_e), step=epoch)
                tf.summary.scalar('Loss/entropy', tf.reduce_mean(loss_H), step=epoch)
                tf.summary.scalar('Loss/_Total', tf.reduce_mean(loss_Total), step=epoch)
                # tf.summary.scalar('Loss/_Pred', tf.reduce_mean(loss_Pred), step=epoch)
                tf.summary.scalar('Update/total_grad_norm', tf.reduce_mean(grad_norms), step=epoch)
                # tf.summary.scalar('Update/total_grad_norm_v', tf.reduce_mean(grad_norms_v), step=epoch)

            summary_writer.flush()
            if self.separate_model:
                self.policy_model.save(train_log_dir + "/policy")
                self.value_energy_model.save(train_log_dir + "/value")
            else:
                self.ppo.save(train_log_dir + "/ppo")

            # if (epoch_hit / len(epoch_reward))<1 and epoch>50:
            #     return
            # if (epoch_hit / len(epoch_reward)) >= num_targets - 0.01 and epoch > 50:
            # if np.isclose(val_energy,480.74,atol=0.01) and epoch > 50:
            #     early_terminate_count += 1
            #     if early_terminate_count >= 50:
            #         return
            # else:
            #     early_terminate_count = 0

    def val_in_run(self, env):
        env.target_range = [np.array([1.9, 0.5, 0.0]), np.array([2.1, 0.4, 0.0])]  # target_range[i]
        state = env.reset()
        done = False
        step = 0
        while not done:
            action = self.get_action(state, greedy=True, random=random)
            # action=int(input("action?"))

            next_state, reward, done = env.step(action)
            state = next_state
            step += 1
        return env.total_energy

    def cal_greedy_rollout(self, env):
        state = env.reset()
        score = 0
        for _ in range(self.action_size):
            action = self.get_action(state, greedy_policy=True)
            state, reward, _ = env.step(action)
            score += reward
        return score

    def cal_optimal_tour(self, env):
        l = range(self.action_size)
        from itertools import permutations
        permutation_list = list(permutations(l, self.action_size))

        energy_record = []
        energy_history_record = []
        for i in permutation_list:
            energy, _, energy_history = env.cal_total_energy(i)
            energy_record.append(energy)
            energy_history_record.append(energy_history)
        energy_record = np.array(energy_record).flatten()
        optimal_energy = np.min(energy_record)
        optimal_index = np.argmin(energy_record)
        optimal_action = permutation_list[optimal_index]
        optimal_energy_history = energy_history_record[optimal_index]
        return energy_record, [optimal_energy, optimal_action, optimal_energy_history]

    def test(self, env, n_map, load_model, n_sample, random, target_range=None, greedy=False, simplified_soln=False):
        if self.separate_model:
            self.policy_model = tf.keras.models.load_model(load_model + "/policy", compile=False)
        else:
            self.ppo = tf.keras.models.load_model(load_model + "/ppo", compile=False)

        # env=CoordinationEnv.make_env(render_plot=True)
        if simplified_soln:
            workspace_min_x = min(env.hull_pts[0, :])
        step_list = []
        eaten_list = []
        energy_list = []
        modified_energy_list = []
        redundant_step_list = []
        crawl_step_list = []
        log_targets = []
        log_activations = []
        log_BCs = []
        log_marks = []
        crawl_activation = np.zeros((3, 99))
        crawl_activation[2] = 1.0
        for i in tqdm(range(n_sample)):
            if target_range == None:
                env.generate_n_targets(from_n_map=n_map)
            else:
                env.target_range = target_range[i]
            state = env.reset()
            done = False
            step = 0
            crawl_step = 0
            prev_action = 1
            redundant_step = 0
            redundant_energy = 0.0
            while not done:
                if greedy:
                    action = 0 if sum(env.state_reachable_list) > 0 else 1
                elif simplified_soln:
                    if len(env.new_reached_list) == 0:
                        action = 1
                    else:
                        action = np.random.randint(2)
                        for i in env.new_reached_list:
                            x_dist = env.target_range[i][0] - env.base[0] - workspace_min_x
                            if x_dist < env.crawl_amount[0]:
                                action = 0
                                break
                else:
                    action = self.get_action(state, greedy=True, random=random)
                    # action = int(input("action?"))
                if action == 1:
                    crawl_step += 1
                    if prev_action == 0:
                        log_targets.append(env.pos.copy())
                        log_BCs.append(0)
                        log_marks.append(0)
                        log_activations.append(np.zeros((3, 99)))

                    log_targets.append(env.pos.copy() + env.crawl_amount.copy())
                    log_targets.append(env.pos.copy() + env.crawl_amount.copy() - env.arm_length.copy())

                    log_BCs.append(0)
                    log_BCs.append(-1)

                    log_marks.append(0)
                    log_marks.append(0)

                    log_activations.append(crawl_activation.copy())
                    log_activations.append(np.zeros((3, 99)))
                next_state, reward, done = env.step(action)

                if action == 0:
                    for i, t in enumerate(env.log_targets):
                        log_targets.append(t.copy() + env.base.copy())
                        log_BCs.append(0)
                        log_marks.append(1)
                        log_activations.append(env.log_activations[i].copy())
                    if np.isclose(env.energy, 0.0):
                        redundant_energy += env.cal_energy_for_redundant_step()
                        redundant_step += 1

                state = next_state
                step += 1
                prev_action = action
            step_list.append(step)
            eaten_list.append(env.targets_eaten)
            energy_list.append(env.total_energy / 300.0)
            redundant_step_list.append(redundant_step)
            modified_energy_list.append(env.total_energy / 300.0 + redundant_energy/300.0)
            crawl_step_list.append(crawl_step)
        np.savez(load_model + "/test_score_random%d_greedy%d_simplified%d" % (random, greedy, simplified_soln),
                 step=step_list,
                 crawl_step=crawl_step_list, redundant_step=redundant_step_list,
                 targets_eaten=eaten_list, energy=energy_list, modified_energy=modified_energy_list,
                 target_range=target_range, log_targets=log_targets,
                 log_activations=log_activations, log_BCs=log_BCs, log_marks=log_marks)


def read_result(filename):
    result = dict(np.load(filename))
    # redundant = result['modified_energy'] - result['energy']
    # result['redundant'] = redundant
    return result


def plot_box(model_dir, num_targets):
    simplified_result = read_result(model_dir + "/test_score_random0_greedy0_simplified1.npz")
    ppo_result = read_result(model_dir + "/test_score_random0_greedy0_simplified0.npz")
    random_result = read_result(model_dir + "/test_score_random1_greedy0_simplified0.npz")
    greedy_result = read_result(model_dir + "/test_score_random1_greedy1_simplified0.npz")

    result_list = [ppo_result, simplified_result, greedy_result, random_result]
    missed_index = np.where(ppo_result['targets_eaten'] < num_targets)[0]

    # remove the index without success
    greater = np.where(greedy_result["energy"] > ppo_result["energy"])[0]
    video_index = np.setdiff1d(greater, missed_index)
    print("cases that ppo has better results /n", video_index)

    target_range = np.load(model_dir + "target%d_test_across_model.npy" % num_targets)

    check_targets = bool(int(input("check target positions in each case? (0 -> False / 1 -> True)")))
    if check_targets:
        import matplotlib.pyplot as plt
        for i in video_index:
            print("case %d" % i)
            marker_size = np.linspace(150, 10, 100)
            arm_x = np.linspace(0.0, 1.0, 100)
            arm_y = np.zeros_like(arm_x)
            for k in range(100):
                plt.scatter(arm_x[k], arm_y[k], s=marker_size[k], color='k', alpha=0.2)
            plt.plot(target_range[i, :, 0], target_range[i, :, 1], 'bx', markersize=8, label="target%d" % i)
            plt.gca().set_aspect('equal')
            plt.gca().set_xlim([-1.0, 10.5])
            plt.gca().set_ylim([-1.5, 1.5])
            plt.savefig(model_dir + "/ppo_better_%d.png" % i)
            plt.close("all")

    # for g in video_index:
    #     os.system(
    #         "ffmpeg -r 1 -i " + model_dir + "/frames_ppo/%03d_%%03d.png -b:v 90M -c:v libx264 -pix_fmt yuv420p -f mov -y " % (
    #                     g + 1) + model_dir + "/sample%d-0.0.mov" % (g + 1))
    # for g in video_index:
    #     os.system(
    #         "ffmpeg -r 1 -i " + model_dir + "/frames_greedy/%03d_%%03d.png -b:v 90M -c:v libx264 -pix_fmt yuv420p -f mov -y " % (
    #                     g + 1) + model_dir + "/sample%d-0.0.mov" % (g + 1))

    x_axis_list = ["ppo", "Q", "greedy", "random"]

    label_list = ['targets_eaten', 'step', 'energy', 'crawl_step', 'modified_energy_list', 'redundant']
    color_list = ['mediumpurple', 'yellowgreen', 'cornflowerblue', 'red']

    for i, label in enumerate(label_list):
        f = lambda x, y: x[y]
        box_config(label, x_axis_list, result_list, color_list, f,load_model=model_dir, note="_normalized")

    label_list = ['step', 'energy', 'crawl_step', 'modified_energy_list', 'redundant']

    for i, label in enumerate(label_list):
        f = lambda x, y: x[y] / x['targets_eaten']
        box_config(label, x_axis_list, result_list, color_list, f, note="")


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


if __name__ == '__main__':
    map = 0
    eps = 0.2

    state_type = "loc_map"
    total_epochs = 5000
    train = True
    box_plot = False
    obs_num_targets = 5
    if not box_plot:
        if train:
            for lr in [1e-4]:
                for termination_step in [59]:
                    for bonus_coeff in [0.18]:
                        for num_targets in [20]:
                            for penalty_coeff in [0.1]:
                                for batch_size in [30]:
                                    for rollout in [300]:
                                        agent = Agent(action_size=2, lr=lr, separate_model=False,
                                                      eps=eps, batch_size=batch_size,
                                                      rollout=rollout)  # 32,256 # 65,260 * 4)
                                        agent.run(n_map=map, num_targets=num_targets, obs_num_targets=obs_num_targets,
                                                  total_epochs=total_epochs,
                                                  penalty_coeff=penalty_coeff,
                                                  bonus_coeff=bonus_coeff,
                                                  termination_step=termination_step, state_type=state_type)
        else:

            num_targets = 40
            target_range = []
            n_sample = 100
            bonus_coeff = 0.18
            penalty_coeff = 0.1
            lr = 5e-5
            termination_step = 59
            obs_num_targets = 8
            rollout = 900
            batch_size = 30

            env = CoordinationEnv.make_env(render_plot=False, penalty_coeff=penalty_coeff, bonus_coeff=bonus_coeff,
                           termination_step=termination_step, state_type=state_type, num_targets=num_targets,
                           obs_num_targets=obs_num_targets)
            # for _ in range(n_sample):
            #     env.generate_n_targets(from_n_map=map)
            #     target_range.append(env.target_range)
            # target_range.append(np.array([[1.9, 0.5, 0.0],[2.2, 0.4, 0.0]]))
            # target_range.append(np.array([[1.2, 0.5, 0.0],[2.2, 0.4, 0.0]]))
            # target_range.append(np.array([[1.2, 0.8, 0.0],[4.2, 0.4, 0.0]]))

            total_epochs = 5

            agent = Agent(action_size=2, lr=lr, separate_model=False,
                          eps=eps, batch_size=batch_size, rollout=rollout)  # 65,260 * 4)
            # env = CoordinationEnv.make_env(render_plot=True, penalty_coeff=penalty_coeff,
            #                bonus_coeff=bonus_coeff,
            #                termination_step=termination_step, state_type=state_type,
            #                num_targets=num_targets, obs_num_targets=obs_num_targets)

            # load_model = "logs_20target/t59-target20-epoch500-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout900-batch30-valu10.000000-loc_map20210603-052936/"
            # load_model = "logs_2target/one-model-64/t59-target2-epoch500-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch6-valu10.000000-loc_map20210709-144428/"
            # load_model = "logs_2target/t59-target2-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210709-194550/"
            # load_model = "logs_2target/t59-target2-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210712-105814/"
            # load_model = "logs_20target/t59-target20-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210712-090148/"
            # load_model = "logs_20target/t59-target20-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210712-222136/"
            # load_model = "0723/t59-target20-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210723-115218/"
            # load_model = "obs8_arc/logs_20target/t59-target20-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210816-004416/"
            load_model = "0816_more obs/obs8_arc/logs_40target/t59-target40-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210816-004518/"
            # load_model = "t59-target2-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210712-084428/"

            target_range = list(
                np.load(load_model + "target%d_test_across_model.npy" % num_targets))  # [0][np.newaxis,:,:])
            # np.save(load_model + "target%d_test_across_model" % num_targets, target_range)

            # env.ep = 0
            # env.render_dir = load_model + "/frames_ppo/"
            # os.makedirs(env.render_dir, exist_ok=True)
            # agent.test(env, n_map=0, n_sample=n_sample, load_model=load_model, random=False,
            #            target_range=target_range, greedy=False)

            # env.render_dir = load_model + "/frames_simplified_soln/"
            # os.makedirs(env.render_dir, exist_ok=True)
            # env.ep = 0
            # agent.test(env, n_map=0, n_sample=n_sample, load_model=load_model, random=False,
            #            target_range=target_range, greedy=False, simplified_soln=True)
            #
            env.render_dir = load_model + "/frames_random/"
            os.makedirs(env.render_dir, exist_ok=True)
            env.ep = 0
            agent.test(env, n_map=0, n_sample=n_sample, load_model=load_model, random=True,
                       target_range=target_range, greedy=False)
            # # #
            # env.render_dir = load_model + "/frames_greedy/"
            # os.makedirs(env.render_dir, exist_ok=True)
            # env.ep = 0
            # agent.test(env, n_map=0, n_sample=n_sample, load_model=load_model, random=True,
            #            target_range=target_range,
            #            greedy=True)

            # env.render_dir = load_model + "/frames_optimal/"
            # os.makedirs(env.render_dir, exist_ok=True)
            # env.ep = 0
            # agent.test(env, n_map=0, n_sample=n_sample, load_model=load_model, random=False, target_range=target_range,
            #            greedy=False)
    else:
        # load_model = "logs_20target/t59-target20-epoch1000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout900-batch30-valu10.000000-loc_map20210607-234905/"
        # load_model = "logs_2target/t59-target2-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210709-194550/"
        # load_model = "t59-target2-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210712-084428/"
        # load_model = "logs_20target/t59-target20-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210710-003700/"
        # load_model = "logs_2target/t59-target2-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210712-105814/"
        # load_model = "logs_20target/t59-target20-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210712-222136/"
        # load_model = "0723/t59-target20-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210723-115218/"
        # load_model = "obs8_arc/logs_20target/t59-target20-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210816-004416/"
        load_model = "0816_more obs/obs8_arc/logs_40target/t59-target40-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout300-batch30-valu10.000000-loc_map20210816-004518/"

        num_targets = 40
        plot_box(load_model, num_targets)
