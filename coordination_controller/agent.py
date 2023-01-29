__all__ = ["Agent"]

import os
import time
import copy
import shutil

import numpy as np
import datetime
from itertools import permutations

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorboard.plugins.hparams import api as hp

from tqdm import tqdm


def get_gaes(rewards, dones, values, next_values, gamma, lamda, normalize):
    values = values.numpy()
    next_values = next_values.numpy()
    gamma = np.float32(gamma)
    lamda = np.float32(lamda)
    gaes = rewards[:, None] + gamma * (1 - dones[:, None]) * next_values - values

    for t in reversed(range(len(gaes) - 1)):
        gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

    target = gaes + values
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    target = (target - target.mean())
    return gaes, target


class PPO(Model):
    def __init__(self, action_size):
        super(PPO, self).__init__()
        self.layer1 = tf.keras.layers.Dense(64, activation="relu")
        self.layer2 = tf.keras.layers.Dense(64, activation="relu")
        self.layer_a1 = tf.keras.layers.Dense(64, activation="relu")
        self.layer_c1 = tf.keras.layers.Dense(64, activation="relu")
        self.logits = tf.keras.layers.Dense(
            action_size, activation="linear"
        )  # activation='softmax')
        self.value = tf.keras.layers.Dense(1, activation="linear")

    def call(self, inputs, training=None, mask=None):
        layer1 = self.layer1(inputs)
        layer2 = self.layer2(layer1)

        layer_a1 = self.layer_a1(layer2)
        logits = self.logits(layer_a1)

        layer_c1 = self.layer_c1(layer2)
        value = self.value(layer_c1)

        return logits, value


class SeparatedPPO(Model):
    def __init__(self, action_size):
        super().__init__()
        self.layer_pi1 = tf.keras.layers.Dense(64, activation="relu")
        self.layer_pi2 = tf.keras.layers.Dense(64, activation="relu")
        self.layer_pi3 = tf.keras.layers.Dense(64, activation="relu")
        self.output_logits = tf.keras.layers.Dense(action_size, activation="linear")
        self.layer_v1 = tf.keras.layers.Dense(64, activation="relu")
        self.layer_v2 = tf.keras.layers.Dense(64, activation="relu")
        self.layer_v3 = tf.keras.layers.Dense(64, activation="relu")
        self.output_value = tf.keras.layers.Dense(1, activation="linear")

    def call(self, inputs, training=None, mask=None):
        layer_pi1 = self.layer_pi1(inputs, training=None, mask=None)
        layer_pi2 = self.layer_pi2(layer_pi1)
        layer_pi3 = self.layer_pi3(layer_pi2)
        logits = self.output_logits(layer_pi3)

        layer_v1 = self.layer_v1(inputs)
        layer_v2 = self.layer_v2(layer_v1)
        layer_v3 = self.layer_v3(layer_v2)
        value = self.output_value(layer_v3)

        return logits, value


class Agent:
    def __init__(
            self,
            action_size=2,
            lr=0.001,
            eps=0.2,
            batch_size=128,
            rollout=1024,
            separate_model=False,
            optimizer_type="RMS",
            **kwargs,
    ):
        self.lr = lr
        self.gamma = 0.99
        self.lamda = 0.98

        self.rollout = rollout
        self.batch_size = batch_size
        self.action_size = action_size
        self.epoch = 1  # Repeat gradient update
        self.ppo_eps = eps
        self.normalize = True
        self.separate_model = separate_model
        if separate_model:
            self.ppo = SeparatedPPO(self.action_size)
        else:
            self.ppo = PPO(self.action_size)

        self.optimizer_type = optimizer_type
        if optimizer_type == "RMS":
            self.opt = optimizers.RMSprop(self.lr)
        elif optimizer_type == "Adam":
            self.opt = optimizers.Adam(self.lr)
        else:
            raise NotImplementedError("Optimizer RMS or ADAM")

    @tf.function
    def get_action_tf(self, state):
        policy, _ = self.ppo(state)
        policy = tf.nn.softmax(policy)
        return policy

    def get_action(self, state, greedy=False):
        policy = self.get_action_tf(state)
        if greedy:
            action = tf.math.argmax(policy, axis=1)
            return action
        else:
            action = [np.random.choice(self.action_size, p=p) for p in policy.numpy()]
        return action

    def get_random_action(self, n):
        return [np.random.randint(0, self.action_size) for _ in range(n)]

    def update(
            self,
            state,
            next_state,
            reward,
            done,
            action,
    ):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        old_policy, current_value = self.ppo(state)

        old_policy = tf.nn.softmax(old_policy)

        _, next_value = self.ppo(tf.convert_to_tensor(next_state, dtype=tf.float32))

        current_value, next_value = tf.squeeze(current_value, axis=-1), tf.squeeze(
            next_value, axis=-1
        )

        adv, target = get_gaes(
            rewards=np.array(reward, dtype=np.float32),
            dones=np.array(done, dtype=np.float32),
            values=current_value,
            next_values=next_value,
            gamma=self.gamma,
            lamda=self.lamda,
            normalize=self.normalize,
        )
        return self.learn(adv, target, old_policy, state, done, action)

    @tf.function
    def tf_train_step(
            self,
            model,
            batch_state,
            train_action,
            train_target,
            train_old_policy,
            train_adv,
            policy_coeff,
            value_coeff,
            entropy_coeff,
            grad_clip,
            opt,
            eps,
    ):
        model_variable = model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_variable)
            train_policy, train_current_value = model(batch_state)
            pi_loss, entropy = self.cal_pi_loss_entropy(
                batch_state,
                train_policy,
                train_action,
                train_old_policy,
                train_adv,
                eps,
            )
            train_current_value = tf.squeeze(train_current_value)
            value_loss = tf.reduce_mean(tf.square(train_target - train_current_value))
            total_loss = (
                    policy_coeff * pi_loss
                    - entropy * entropy_coeff
                    + value_loss * value_coeff
            )
        grads = tape.gradient(total_loss, model_variable)
        grads, grad_norm = tf.clip_by_global_norm(grads, grad_clip)
        opt.apply_gradients(zip(grads, model_variable))

        return pi_loss, value_loss, entropy, total_loss, grad_norm

    def learn(self, adv, target, old_policy, state, done, action):
        loss_p = []
        loss_v = []
        loss_e = []
        loss_H = []
        loss_Total = []
        grad_norms = []
        sample_range = np.arange(self.rollout)
        for _ in range(self.epoch):
            np.random.shuffle(sample_range)
            for i in range(self.rollout // self.batch_size):
                sample_idx = sample_range[
                             self.batch_size * i: self.batch_size * (i + 1)
                             ]

                batch_state = tf.gather(state, sample_idx)
                train_adv = tf.gather(adv, sample_idx)
                train_target = tf.gather(target, sample_idx)
                train_action = tf.gather(action, sample_idx)
                train_old_policy = tf.gather(old_policy, sample_idx)

                (
                    pi_loss,
                    value_loss,
                    entropy,
                    total_loss,
                    grad_norm,
                ) = self.tf_train_step(
                    self.ppo,
                    batch_state,
                    train_action,
                    train_target,
                    train_old_policy,
                    train_adv,
                    self.policy_coeff,
                    self.value_coeff,
                    self.entropy_coeff,
                    self.grad_clip,
                    self.opt,
                    self.ppo_eps,
                )

                loss_p.append(pi_loss.numpy())
                loss_v.append(value_loss.numpy())
                loss_H.append(entropy.numpy())
                loss_Total.append(total_loss.numpy())
                grad_norms.append(grad_norm.numpy())

        return (
            loss_p,
            loss_v,
            loss_H,
            loss_e,
            loss_Total,
            grad_norms,
        )

    @tf.function
    def cal_pi_loss_entropy(
            self, batch_state, train_policy, train_action, train_old_policy, train_adv, eps
    ):

        train_policy_sf = tf.nn.softmax(train_policy)

        entropy = tf.reduce_mean(-train_policy_sf * tf.math.log(train_policy_sf + 1e-8))

        onehot_action = tf.one_hot(train_action, self.action_size)
        selected_prob = tf.reduce_sum(train_policy_sf * onehot_action, axis=-1)
        selected_old_prob = tf.reduce_sum(train_old_policy * onehot_action, axis=-1)
        logpi = tf.math.log(selected_prob + 1e-8)
        logoldpi = tf.math.log(selected_old_prob + 1e-8)

        ratio = tf.exp(logpi - logoldpi)
        clipped_ratio = tf.clip_by_value(
            ratio, clip_value_min=1 - eps, clip_value_max=1 + eps
        )
        minimum = tf.minimum(
            tf.multiply(train_adv, clipped_ratio), tf.multiply(train_adv, ratio)
        )
        pi_loss = -tf.reduce_mean(minimum)
        return pi_loss, entropy

    def run(
            self,
            env,
            num_targets,
            total_epochs,
            penalty_coeff,
            bonus_coeff,
            value_coeff,
            policy_coeff,
            entropy_coeff,
            grad_clip,
            termination_step,
            state_type,
            reward_type,
            target_type,
            use_arena,
            arena_type,
            use_obstacle,
            flip_crawl_direction,
            food_weights,
            num_arms,
            multi_agent=True,
            render=False,
            manual_control=False,
            ex_log_file_writer=None,
            experiment_id: int = 0,
            verbose: bool = True,
            **kwargs,
    ):
        # Run Parameter
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.value_coeff = tf.constant(value_coeff, tf.float32)
        self.policy_coeff = tf.constant(policy_coeff, tf.float32)
        self.entropy_coeff = tf.constant(entropy_coeff, tf.float32)
        self.grad_clip = tf.constant(grad_clip, tf.float32)
        self.ppo_eps = tf.constant(self.ppo_eps, tf.float32)

        # Tensorboard Logdir
        os.makedirs("logs", exist_ok=True)
        # Depending on what you want to include in log-directory name, adjust the below string.
        # All of the important hyperparameters will be visible in tensorboard anyway.
        parameter_stamp = "-".join(
            [
                f"{experiment_id}",
                f"{entropy_coeff=}",
                f"{current_time}",
            ]
        )
        train_log_dir = os.path.join("logs", parameter_stamp)
        with ex_log_file_writer:  # Register summary_writer on sacred
            summary_writer = tf.summary.create_file_writer(train_log_dir)

        # Tensorboard hparams prep
        hparams = self.get_hp_params(
            self.lr,
            termination_step,
            bonus_coeff,
            num_targets,
            penalty_coeff,
            self.batch_size,
            self.rollout,
            self.ppo_eps.numpy(),
            value_coeff,
            policy_coeff,
            entropy_coeff,
            grad_clip,
            self.optimizer_type,
            arena_type,
            food_weights,
            flip_crawl_direction,
            target_type,
        )

        # Environment configuration
        if render:
            env.render_dir = os.path.join(train_log_dir, "frames_ppo/")
            os.makedirs(env.render_dir, exist_ok=True)

        # Prepare environment
        env.generate_n_targets()
        state = env.reset()
        # np.save(train_log_dir+"/1map",env.target_range)

        score = 0
        for epoch in tqdm(range(total_epochs)):
            state_list, next_state_list = [], []
            reward_list, done_list, action_list = [], [], []
            epoch_reward = []
            energy_list = []
            action_collection_list = []
            _epoch_reward = 0.0
            epoch_hit = 0.0
            episode = 0
            stime = time.time()
            for _ in range(self.rollout):
                if manual_control:
                    action_input = int(input("which arm should move?"))
                    if action_input == 1:
                        action = [1, 0, 0, 0]
                    elif action_input == 2:
                        action = [0, 1, 0, 0]
                    elif action_input == 3:
                        action = [0, 0, 1, 0]
                    elif action_input == 4:
                        action = [0, 0, 0, 1]
                    else:
                        raise NotImplementedError("Provide 1-4 arm action value")
                else:
                    action = self.get_action(state, greedy=False)

                if not multi_agent:
                    action_collection_list.append(action[0])
                next_state, reward, done, _ = env.step(action)

                score += reward

                state_list.append(state)
                next_state_list.append(next_state)
                reward_list.append(reward)
                done_list.append(done)
                action_list.append(action)

                state = next_state

                if done:
                    epoch_reward.append(score)
                    energy_list.append(env.total_energy)
                    _epoch_reward += score
                    epoch_hit += env.targets_eaten
                    env.generate_n_targets()

                    state = env.reset()
                    episode += 1
                    score = 0

            (loss_p, loss_v, loss_H, loss_e, loss_Total, grad_norms,) = self.update(
                state=state_list,
                next_state=next_state_list,
                reward=reward_list,
                done=done_list,
                action=action_list,
            )

            if verbose:
                print(
                    epoch,
                    f"mean_epoch_reward={np.mean(epoch_reward):.2f}",
                    "average hit! %.2f" % (epoch_hit / len(epoch_reward)),
                    f"vloss={np.mean(loss_v):.3f}",
                    f"epoch-walltime {time.time() - stime:.2f} sec",
                )

            if epoch % 10 == 0:
                with summary_writer.as_default():
                    hp.hparams(hparams)
                    # fmt: off
                    tf.summary.scalar(r"Main/episode_reward", np.mean(epoch_reward), step=epoch)
                    tf.summary.scalar(r"Main/episode_energy", np.mean(energy_list), step=epoch)
                    tf.summary.scalar(r"Main/episode_targets", epoch_hit / len(epoch_reward), step=epoch)
                    tf.summary.scalar(r"Loss/policy", np.mean(loss_p), step=epoch)
                    tf.summary.scalar(r"Loss/value", np.mean(loss_v), step=epoch)
                    tf.summary.scalar(r"Loss/entropy", np.mean(loss_H), step=epoch)
                    tf.summary.scalar(r"Loss/total_loss", np.mean(loss_Total), step=epoch)
                    tf.summary.scalar(r"Update/total_grad_norm", np.mean(grad_norms), step=epoch)
                    tf.summary.histogram("action_collection(centralized)", action_collection_list, step=epoch)
                    tf.summary.histogram("reward_distribution", reward_list, step=epoch)
                    # fmt: on
                summary_writer.flush()

            if epoch % 100 == 0:
                self.ppo.save(os.path.join(train_log_dir, "ppo"))
                if epoch % 500 == 0:
                    shutil.copytree(
                        os.path.join(train_log_dir, "ppo"),
                        os.path.join(train_log_dir, f"ppo{epoch}"),
                    )
        summary_writer.close()

    def cal_greedy_rollout(self, env):
        state = env.reset()
        score = 0
        for _ in range(self.action_size):
            action = self.get_action(state, greedy=True)
            state, reward, _, _ = env.step(action)
            score += reward
        return score

    def cal_optimal_tour(self, env):
        l = range(self.action_size)

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

    def cal_analytic_soln_simplified_env(self, env, workspace_min_x):
        if env.num_arms == 1:
            if len(env.arm_list[0].new_reached_list) == 0:
                action = [1]
            else:
                action = [np.random.randint(2)]
                for i in env.arm_list[0].new_reached_list:
                    x_dist = env.target_range[i][0] - env.arm_list[0].base[0]  # - workspace_min_x
                    if x_dist < env.arm_list[0].crawl_amount[0]:
                        action = [0]
                        break
        else:
            arm1_obs = len(env.arm_list[0].obs_list)
            arm1_get = len(env.arm_list[0].new_reached_list)
            arm2_obs = len(env.arm_list[1].obs_list)
            arm2_get = len(env.arm_list[1].new_reached_list)
            if arm1_obs + arm2_obs == 0:  # cc
                action = [1, 1]
            else:  # if arm1_get>0 and arm2_get>0:
                num_in_rx = 0
                for i in env.arm_list[0].new_reached_list:
                    x_dist = env.target_range[i][0] - env.head[0] - workspace_min_x
                    if x_dist < env.arm_list[0].crawl_amount[0]:
                        num_in_rx += 1
                num_in_ry = 0
                for i in env.arm_list[1].new_reached_list:
                    x_dist = env.target_range[i][2] - env.head[2] - workspace_min_x
                    if x_dist < env.arm_list[1].crawl_amount[2]:
                        num_in_ry += 1
                if num_in_rx > 0 and num_in_ry > 0:  # rr
                    action = [0, 0]
                else:
                    rc_value = arm1_get + arm2_obs - num_in_ry
                    cr_value = arm1_obs - num_in_rx + arm2_get
                    if rc_value > cr_value:  # rc
                        action = [0, 1]
                    elif rc_value < cr_value:  # cr
                        action = [1, 0]
                    else:  # rc=cr
                        action1 = np.random.randint(2)
                        action2 = np.random.randint(2)
                        action = [action1, action2]

        return action

    def test(
            self,
            env,
            load_model,
            n_sample,
            random,
            target_range=None,
            greedy=False,
            simplified_soln=False,
            obstacle_range=None,
            use_obstacle=False,
            multi_agent=True,
            verbose=True,
            **kwargs,
    ):
        if not obstacle_range:
            obstacle_range = []
        if greedy == False and random == False and simplified_soln == False:
            self.ppo = tf.keras.models.load_model(
                os.path.join(load_model, "ppo"), compile=False
            )
            log_dir = os.path.join(load_model, "frames_ppo/")
        else:
            log_dir = os.path.join(load_model, "frames_simplified/")
        os.makedirs(log_dir, exist_ok=True)

        if simplified_soln:
            workspace_min_x = min(env.hull_pts[0, :])

        step_list = []
        eaten_list = []
        arm_targets_eaten_list = []
        energy_list = []
        arm_energy_list = []
        crawl_step_list = []
        redundant_step_list = []
        arm_redundant_energy_list = []
        modified_energy_list = []
        log_arm_degree = []
        for arm in env.arm_list:
            log_arm_degree.append(arm.angle)

        for i in tqdm(range(n_sample), desc="sample: ", position=0):
            if target_range is None:
                env.generate_n_targets()
            else:
                env.target_range = target_range[i]

            if use_obstacle:
                env.obstacle_pts_list = obstacle_range[i]
                env.load_obstacle(env.obstacle_pts_list)
            state = env.reset()
            log_head = [env.head]
            log_action = [env.action]
            log_sum_crawl = [env.do_crawl]
            log_reached_map = [
                [env.arm_list[i].n_target_collected for i in range(env.num_arms)]
            ]

            done = False
            step = 0
            crawl_step = np.zeros(env.num_arms)
            arm_energy = np.zeros(env.num_arms)
            arm_redundant_energy = np.zeros(env.num_arms)
            arm_redundant_step = np.zeros(env.num_arms)

            pbar = tqdm(desc="episode steps: ", position=1)
            while not done:
                pbar.update(1)
                if greedy:
                    action = []
                    for i in range(env.num_arms):
                        action.append(
                            0 if len(env.arm_list[i].new_reached_list) > 0 else 1
                        )
                elif simplified_soln:
                    if env.num_arms in [1, 2]:
                        action = self.cal_analytic_soln_simplified_env(
                            env, workspace_min_x
                        )
                    else:
                        raise NotImplementedError("No analytical soln for 4-arm case")
                elif random:
                    action = self.get_random_action(env.num_arms)
                else:
                    action = self.get_action(state, greedy=False)

                next_state, reward, done, _ = env.step(action)
                for i in range(env.num_arms):
                    if env.action[i] == 1:
                        crawl_step[i] += 1
                log_head.extend(copy.deepcopy(env.head))
                log_action.extend(env.action)
                log_sum_crawl.append(env.do_crawl)
                log_reached_map.extend(
                    [env.arm_list[i].n_target_collected for i in range(env.num_arms)]
                )

                for i in range(env.num_arms):
                    arm_energy[i] += env.arm_list[i].energy

                if np.isclose(env.energy, 0.0):
                    for i in range(env.num_arms):
                        arm_redundant_energy[i] += env.cal_energy_for_redundant_step()
                        arm_redundant_step[i] += 1

                state = next_state
                step += 1
            pbar.close()

            if verbose:
                if greedy:
                    print('\ngreedy: \n', env.targets_eaten)
                elif simplified_soln:
                    print('\nanalytical soln: \n', env.targets_eaten)
                elif random:
                    print('\nrandom: \n', env.targets_eaten)
                else:
                    print('\nppo: \n', env.targets_eaten)

            step_list.append(step)
            eaten_list.append(env.targets_eaten)
            arm_targets_eaten_list.append([arm.targets_eaten for arm in env.arm_list])
            arm_energy_list.append(arm_energy)
            energy_list.append(env.total_energy / 300.0)
            redundant_step_list.append(arm_redundant_step)
            arm_redundant_energy_list.append(arm_redundant_energy)
            modified_energy_list.append(
                env.total_energy / 300.0 + sum(arm_redundant_energy) / 300.0
            )
            crawl_step_list.append(crawl_step)

            np.savez(
                os.path.join(
                    load_model,
                    "test_score_random%d_greedy%d_simplified%d"
                    % (random, greedy, simplified_soln),
                ),
                step=step_list,
                arm_crawl_step=crawl_step_list,
                targets_eaten=eaten_list,
                arm_targets_eaten=arm_targets_eaten_list,
                arm_redundant_step=redundant_step_list,
                arm_redundant_energy=arm_redundant_energy_list,
                modified_energy=modified_energy_list,
                arm_energy=arm_energy_list,
                energy=energy_list,
                target_range=target_range,
                log_arm_degree=log_arm_degree,
                log_head=log_head,
                log_action=log_action,
                log_sum_crawl=log_sum_crawl,
                log_reached_map=log_reached_map,
            )

            np.savez(
                os.path.join(
                    log_dir,
                    "ep%d_log_random%d_greedy%d_simplified%d"
                    % (env.ep, random, greedy, simplified_soln),
                ),
                target_range=target_range,
                log_arm_degree=log_arm_degree,
                log_head=log_head,
                log_action=log_action,
                log_sum_crawl=log_sum_crawl,
                log_reached_map=log_reached_map,
                log_targets=[arm.log_state["log_targets"] for arm in env.arm_list],
                log_BCs=[arm.log_state["log_BCs"] for arm in env.arm_list],
                log_marks=[arm.log_state["log_marks"] for arm in env.arm_list],
                log_activations=[
                    arm.log_state["log_activations"] for arm in env.arm_list
                ],
                log_arm_head=[arm.log_state["log_head"] for arm in env.arm_list],
                log_arm=[arm.log_state["log_pos"] for arm in env.arm_list],
            )

    def get_hp_params(
            self,
            lr,
            termination_step,
            bonus_coeff,
            num_targets,
            penalty_coeff,
            batch_size,
            rollout,
            eps,
            value_coeff,
            policy_coeff,
            entropy_coeff,
            grad_clip,
            optimizer_type,
            arena_type,
            food_weights,
            flip_crawl_direction,
            target_type,
    ):
        # Register any hyperparameters that we want to monitor and compare
        return {
            hp.HParam("lr", hp.RealInterval(1e-7, 1e-3)): lr,
            hp.HParam("termination_step", hp.IntInterval(1, 100)): termination_step,
            hp.HParam("bonus_coeff", hp.RealInterval(0.0, 1.0)): bonus_coeff,
            hp.HParam("num_targets", hp.IntInterval(1, 50)): num_targets,
            hp.HParam("penalty_coeff", hp.RealInterval(0.0, 10.0)): penalty_coeff,
            hp.HParam(
                "batch_size", hp.Discrete([16, 32, 64, 128, 25256, 512, 1024])
            ): batch_size,
            hp.HParam("rollout", hp.IntInterval(10, 1000)): rollout,
            hp.HParam("eps", hp.RealInterval(0.1, 0.3)): eps,
            hp.HParam("value_coeff", hp.RealInterval(1e-3, 1e3)): value_coeff,
            hp.HParam("policy_coeff", hp.RealInterval(1e-3, 1e3)): policy_coeff,
            hp.HParam("entropy_coeff", hp.RealInterval(1e-3, 1e2)): entropy_coeff,
            hp.HParam("grad_clip", hp.RealInterval(0.0, 100.0)): grad_clip,
            hp.HParam("optimizer_type", hp.Discrete(["RMS", "Adam"])): optimizer_type,
            hp.HParam(
                "arena_type", hp.Discrete(["mid-square", "square", "diagonal"])
            ): arena_type,
            hp.HParam("food_weights", hp.RealInterval(0.0, 10.0)): food_weights,
            hp.HParam(
                "flip_crawl_direction", hp.Discrete([True, False])
            ): flip_crawl_direction,
            hp.HParam(
                "target_type", hp.Discrete(["constrain", "random", "gird", "obstacle"])
            ): target_type,
        }

    def get_hp_metrics(self):
        return [
            hp.Metric(r"Main/episode_reward", display_name="reward (val.)"),
            hp.Metric(r"Main/episode_energy", display_name="energy (val.)"),
            hp.Metric(r"Main/episode_targets", display_name="targets (val.)"),
        ]
