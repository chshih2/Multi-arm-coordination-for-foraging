import tensorflow as tf
import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import sys
# # Keras outputs warnings using `print` to stderr so let's direct that to devnull temporarily
# stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')
import numpy as np
import datetime
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
import tensorflow_probability as tfp
import sys
from tqdm import tqdm

# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()

# from switch_env import *
from end2end_env import *
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
    def __init__(self, action_size, n_component, n_muscles, n_high_state):
        super(PPO, self).__init__()
        self.layer1 = tf.keras.layers.Dense(128 + 32, activation='relu')
        self.layer2 = tf.keras.layers.Dense(128 + 32, activation='relu')
        self.layer_a1 = tf.keras.layers.Dense(16, activation='relu')
        self.layer_c1 = tf.keras.layers.Dense(16, activation='relu')
        self.logits = tf.keras.layers.Dense(action_size, activation='linear')  # activation='softmax')
        self.value = tf.keras.layers.Dense(1, activation='linear')

        self.a_layer = tf.keras.layers.Dense(128, activation='relu')
        # self.activation = tf.keras.layers.Dense(n_component * n_muscles, activation='linear')
        self.activation_mean = tf.keras.layers.Dense(n_component * n_muscles, activation='tanh')
        # self.activation_mean_scale = tf.keras.layers.Lambda(lambda x: x * self.action_bound)
        self.activation_std = tf.keras.layers.Dense(n_component * n_muscles, activation='softplus')
        self.n_high_state = n_high_state

    @tf.function
    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)

        layer_a1 = self.layer_a1(layer2)
        logits = self.logits(layer_a1)

        layer_c1 = self.layer_c1(layer2)
        value = self.value(layer_c1)

        a_layer = self.a_layer(layer2)
        activation_mean = self.activation_mean(a_layer)
        # activation_mean = self.activation_mean_scale(activation_mean)
        activation_std = self.activation_std(a_layer)
        return logits, value, [activation_mean, activation_std]


class Agent:
    def __init__(self, action_size=5, lr=0.001, eps=0.2, batch_size=128,
                 rollout=1024, obs_num_targets=5):
        self.lr = lr
        self.gamma = 0.99
        self.lamda = 0.96

        self.rollout = rollout
        self.batch_size = batch_size
        self.action_size = action_size
        self.epoch = 3
        self.ppo_eps = eps
        self.normalize = True

        self.n_component = 11
        self.n_muscles = 3
        self.ppo = PPO(self.action_size, self.n_component, self.n_muscles, obs_num_targets * 2)

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
                policy, _, activation = self.ppo(state)
                muscle_activation = np.random.normal(activation[0][0], activation[1][0], size=33)[np.newaxis, :]
                # policy = masking(policy, flag)
                policy = tf.nn.softmax(policy)
                if greedy:
                    action = tf.math.argmax(policy[0])
                    return action, muscle_activation
                else:
                    dist = tfp.distributions.Categorical(probs=policy)
                    action = dist.sample()
        return action[0], muscle_activation

    def log_pdf(self, mu, std, action):
        # std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        action = tf.convert_to_tensor(action, dtype=tf.float32)[:, 0, :]
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
                         var - 0.5 * tf.math.log(var * 2 * np.pi)
        return log_policy_pdf  # tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def normal_entropy(self, std):
        var = std ** 2
        normal_entropy = 0.5 * tf.math.log(var * 2 * np.pi) + 0.5
        return tf.reduce_mean(normal_entropy, axis=0)

    # @tf.function
    def update(self, state, next_state, reward, done, action, activation, baseline_value=[], dense_reward=[]):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        old_policy, current_value, old_activation = self.ppo(state)
        old_log_activation_policy = self.log_pdf(old_activation[0], old_activation[1], activation)

        old_policy = tf.nn.softmax(old_policy)

        if baseline_value == []:
            _, next_value, _ = self.ppo(tf.convert_to_tensor(next_state, dtype=tf.float32))
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
        return self.learn(adv, target, old_policy, state, done, action, activation, old_log_activation_policy)

    # @tf.function
    def learn(self, adv, target, old_policy, state, done, action, activation, old_log_activation_policy):
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
                batch_activation = [activation[i] for i in sample_idx]
                batch_target = [target[i] for i in sample_idx]
                batch_adv = [adv[i] for i in sample_idx]
                batch_old_policy = [old_policy[i] for i in sample_idx]
                batch_old_log_activation_policy = [old_log_activation_policy[i] for i in sample_idx]
                train_adv = tf.convert_to_tensor(batch_adv, dtype=tf.float32)
                train_target = tf.convert_to_tensor(batch_target, dtype=tf.float32)
                train_action = tf.convert_to_tensor(batch_action, dtype=tf.int32)
                train_activation = tf.convert_to_tensor(batch_activation, dtype=tf.float32)
                train_old_policy = tf.convert_to_tensor(batch_old_policy, dtype=tf.float32)
                train_old_log_activation_policy = tf.convert_to_tensor(batch_old_log_activation_policy,
                                                                       dtype=tf.float32)

                ppo_variable = self.ppo.trainable_variables
                with tf.GradientTape() as tape:
                    tape.watch(ppo_variable)
                    batch_state = tf.convert_to_tensor(batch_state, dtype=tf.float32)
                    train_policy, train_current_value, new_activation = self.ppo(batch_state)
                    new_log_activation_policy = self.log_pdf(new_activation[0], new_activation[1], train_activation)

                    onehot_action = tf.one_hot(train_action, self.action_size)
                    selected_prob = self.cal_onehot_action_logprob(train_policy, onehot_action)
                    selected_old_prob = self.cal_onehot_action_logprob(train_old_policy, onehot_action)
                    combine_policy = tf.concat([selected_prob[:, tf.newaxis], new_log_activation_policy], axis=-1)
                    combine_old_policy = tf.concat([selected_old_prob[:, tf.newaxis], train_old_log_activation_policy],
                                                   axis=-1)
                    pi_loss = self.cal_pi_loss(combine_policy, combine_old_policy, train_adv)
                    train_policy = tf.nn.softmax(train_policy)
                    entropy = self.cal_entropy(train_policy)
                    normal_entropy = self.normal_entropy(new_activation[1])
                    total_entropy = tf.concat([entropy[tf.newaxis], normal_entropy], axis=0)
                    train_current_value = tf.squeeze(train_current_value)
                    value_loss = tf.reduce_mean(tf.square(train_target - train_current_value))
                    action_loss = tf.reduce_mean(pi_loss - total_entropy * 1e-3)
                    total_loss = action_loss + value_loss * self.value_coeff
                grads = tape.gradient(total_loss, ppo_variable)
                grads, grad_norm = tf.clip_by_global_norm(grads, 100)
                self.opt.apply_gradients(zip(grads, ppo_variable))
                loss_p.append(pi_loss)
                loss_v.append(value_loss)
                loss_H.append(entropy)
                loss_Total.append(total_loss)
                grad_norms.append(grad_norm)

        return loss_p, loss_v, loss_H, loss_e, loss_Total, loss_Pred, grad_norms, grad_norms_v

    def cal_onehot_action_logprob(self, policy, onehot_action):
        train_policy = tf.nn.softmax(policy)
        selected_prob = tf.reduce_sum(train_policy * onehot_action, axis=1)
        logpi = tf.math.log(selected_prob + 1e-8)
        return logpi

    @tf.function
    def cal_pi_loss(self, selected_prob, selected_old_prob, train_adv):

        ratio = tf.exp(selected_prob - selected_old_prob)
        clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1 - self.ppo_eps,
                                         clip_value_max=1 + self.ppo_eps)
        minimum = tf.minimum(tf.multiply(train_adv[:, tf.newaxis], clipped_ratio),
                             tf.multiply(train_adv[:, tf.newaxis], ratio))
        pi_loss = -tf.reduce_mean(minimum, axis=0)
        return pi_loss

    def cal_entropy(self, train_policy):
        return tf.reduce_mean(-train_policy * tf.math.log(train_policy + 1e-8))

    def run(self, n_map, num_targets, obs_num_targets, total_epochs, penalty_coeff, bonus_coeff, energy_coeff,
            dist_coeff,dist_tol, termination_step,
            state_type,
            load=False):
        self.value_coeff = 10.0  # 0.5#5e-3#1.0
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        parameter_stamp = "t%d-target%d-epoch%d-pen%f-bon%f-eg%f-dist%f-lr%f-eps%.3f-rollout%d-batch%d-valu%f-" % (
            termination_step, num_targets, total_epochs, penalty_coeff, bonus_coeff, energy_coeff, dist_coeff, self.lr,
            self.ppo_eps,
            self.rollout,
            self.batch_size, self.value_coeff) + state_type
        train_log_dir = 'end2end_logs_%dtarget/dist_coeff%f_tol%f/' % (num_targets,dist_coeff,dist_tol) + parameter_stamp + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)
        if load:
            self.ppo = tf.keras.models.load_model("logs/ppo.h5py", compile=False)

        # Environment configuration
        env = make_env(render_plot=False, penalty_coeff=penalty_coeff, bonus_coeff=bonus_coeff,
                       energy_coeff=energy_coeff,
                       dist_coeff=dist_coeff,dist_tol=dist_tol,
                       termination_step=termination_step, state_type=state_type, num_targets=num_targets,
                       obs_num_targets=obs_num_targets)
        # env.render_dir = train_log_dir + "/frames_ppo/"
        # os.makedirs(env.render_dir, exist_ok=True)

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
        # log_step = []
        for epoch in range(total_epochs):
            state_list, next_state_list = [], []
            reward_list, done_list, action_list, activation_list = [], [], [], []
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
                action, activation = self.get_action(state, greedy=False, random=False)

                next_state, reward, done = env.step(action, activation)
                # log_step.append([env.n_target_collected, env.energy / 300.0, env.dist, action])

                score += reward

                state_list.append(state)
                next_state_list.append(next_state)
                reward_list.append(reward)
                done_list.append(done)
                action_list.append(action)
                activation_list.append(activation)

                state = next_state

                if done:
                    # print(reward)
                    if np.isnan(score) or np.isnan(env.total_energy):
                        print("nan", score, env.total_energy, env.curr_activation, env.state)
                        continue
                    else:
                        epoch_reward.append(score)
                        energy_list.append(env.total_energy)
                        _epoch_reward += score
                        epoch_hit += env.targets_eaten
                        episode += 1
                    env.generate_n_targets(from_n_map=n_map)

                    state = env.reset()

                    score = 0

            loss_p, loss_v, loss_H, loss_e, loss_Total, loss_Pred, grad_norms, grad_norms_v = self.update(
                state=state_list, next_state=next_state_list,
                reward=reward_list, done=done_list, action=action_list, activation=activation_list,
                dense_reward=dense_reward_list,
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
                tf.summary.scalar('Update/total_grad_norm_v', tf.reduce_mean(grad_norms_v), step=epoch)

            summary_writer.flush()

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
        np.save("log_step", log_step)

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

    def test(self, env, n_map, load_model, n_sample, random, target_range=None, greedy=False):
        self.ppo = tf.keras.models.load_model(load_model + "/ppo", compile=False)

        # env=make_env(render_plot=True)
        score_list=[]
        step_list = []
        eaten_list = []
        energy_list = []
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
            score=0
            while not done:
                if greedy:
                    action = 0 if sum(env.state_reachable_list) > 0 else 1
                else:
                    action, activation = self.get_action(state, greedy=True, random=random)
                    # action = int(input("action?"))
                if action == 1:
                    crawl_step += 1
                    # if prev_action == 0:
                    #     log_targets.append(env.pos.copy())
                    #     log_BCs.append(0)
                    #     log_marks.append(0)
                    #     log_activations.append(np.zeros((3, 99)))
                    #
                    # log_targets.append(env.pos.copy() + env.crawl_amount.copy())
                    # log_targets.append(env.pos.copy() + env.crawl_amount.copy() - env.arm_length.copy())
                    #
                    # log_BCs.append(0)
                    # log_BCs.append(-1)
                    #
                    # log_marks.append(0)
                    # log_marks.append(0)
                    #
                    # log_activations.append(crawl_activation.copy())
                    # log_activations.append(np.zeros((3, 99)))
                next_state, reward, done = env.step(action, activation)
                score+=reward

                # if action == 0:
                #     for i, t in enumerate(env.log_targets):
                #         log_targets.append(t.copy() + env.base.copy())
                #         log_BCs.append(0)
                #         log_marks.append(1)
                #         log_activations.append(env.log_activations[i].copy())
                state = next_state
                step += 1
                prev_action = action
            step_list.append(step)
            eaten_list.append(env.targets_eaten)
            energy_list.append(env.total_energy / 300.0)
            crawl_step_list.append(crawl_step)
            score_list.append(score)
        np.savez(load_model + "/test_score_random%d_greedy%d" % (random, greedy),
                 score=score_list,
                 step=step_list,
                 crawl_step=crawl_step_list,
                 targets_eaten=eaten_list, energy=energy_list, target_range=target_range, log_targets=log_targets,
                 log_activations=log_activations, log_BCs=log_BCs, log_marks=log_marks)


def plot_box(model_dir, num_targets):
    e2e_result = np.load(model_dir + "/test_score_e2e.npz")
    e2e_result=dict(e2e_result)
    e2e_result['modified_energy_list']=e2e_result['energy']
    # e2e30_result = np.load(model_dir + "/test_score_30.npz")
    ppo_result = np.load(model_dir + "/test_score_random0_greedy0_simplified0.npz")
    simplified_result = np.load(model_dir + "/test_score_random0_greedy0_simplified1.npz")
    random_result = np.load(model_dir + "/test_score_random1_greedy0_simplified0.npz")
    random_result=dict(random_result)
    random_result['modified_energy_list'] = random_result['modified_energy']
    greedy_result = np.load(model_dir + "/test_score_random1_greedy1_simplified0.npz")
    result_list = [ppo_result,simplified_result, greedy_result, random_result, e2e_result]
    # result_list = [ppo_result, e2e30_result,e2e_result]
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

    # x_axis_list = ["high+low","e2e-30%","e2e-5%"]
    x_axis_list = ["ppo","Q","greedy","random","e2e"]

    # label_list = ['step', 'targets_eaten', 'energy', 'crawl_step']
    # import matplotlib.pyplot as plt
    # for label in label_list:
    #     fig, ax = plt.subplots(figsize=(9, 4))
    #     all_data = [result[label] for result in
    #                 result_list]  # [ppo1_result[label], ppo2_result[label], ppo3_result[label], random_result[label]]
    #     ax.boxplot(all_data)
    #     plt.setp(ax, xticks=[y + 1 for y in range(len(all_data))],
    #              xticklabels=x_axis_list)  # ['0.001', "0.01", "0.1", 'random'])
    #     plt.title(label)
    #     plt.savefig(load_model + "/" + label + ".png")
    # plt.show()

    # label_list = ['step', 'energy', 'crawl_step']
    # import matplotlib.pyplot as plt
    # for label in label_list:
    #     fig, ax = plt.subplots(figsize=(9, 4))
    #     all_data = [result[label] / result['targets_eaten'] for result in
    #                 result_list]  # [ppo1_result[label], ppo2_result[label], ppo3_result[label], random_result[label]]
    #     ax.boxplot(all_data)
    #     plt.setp(ax, xticks=[y + 1 for y in range(len(all_data))],
    #              xticklabels=x_axis_list)  # ['0.001', "0.01", "0.1", 'random'])
    #     plt.title(label)
    #     plt.savefig(load_model + "/" + label + "_normalized.png")
    #     # plt.show()

    # label_list = ['energy', 'crawl_step']
    # import matplotlib.pyplot as plt
    # for label in label_list:
    #     fig, ax = plt.subplots(figsize=(9, 4))
    #     all_data = [np.delete(result[label], missed_index) for result in
    #                 result_list[:-1]]  # [ppo1_result[label], ppo2_result[label], ppo3_result[label]]
    #     ax.boxplot(all_data)
    #     plt.setp(ax, xticks=[y + 1 for y in range(len(all_data))],
    #              xticklabels=x_axis_list[:-1])  # ['0.001', "0.01", "0.1"])
    #     plt.title(label)
    #     plt.savefig(load_model + "/" + label + "out_of_success.png")
    #     # plt.show()

    label_list = ['step', 'modified_energy_list', 'crawl_step']
    color_list = ['mediumpurple', 'hotpink','yellowgreen', 'cornflowerblue','gold']
    # color_list = ['mediumpurple', 'cornflowerblue', 'gold']
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    for i, label in enumerate(label_list):
        fig, ax = plt.subplots()

        # fig, (ax, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]},figsize=(5.0, 4.0))

        all_data = [result[label]/(result['targets_eaten']+1) for result in
                    result_list[:-1]]  # [ppo1_result[label], ppo2_result[label], ppo3_result[label], random_result[label]]

        bp=ax.boxplot(all_data, widths=0.7,showfliers=False, patch_artist=True)
        # ax1_data=[data for data in all_data[:3]]
        # # ax1_data.append([])
        # bp = ax.boxplot(ax1_data, widths=0.5,showfliers=False, patch_artist=True)
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
                 xticklabels=x_axis_list[:-1])
        # plt.setp(ax, xticks=[y + 1 for y in range(len(all_data))],
        #          xticklabels=x_axis_list)




        # ax2 = ax.twinx()
        # ax2_data = [[] for _ in range(3)]
        # ax2_data.append(all_data[-1])
        # bp2=ax2.boxplot(ax2_data, showfliers=False, patch_artist=True)

        # bp2=ax2.boxplot(all_data[-1], widths=0.5,showfliers=False, patch_artist=True)
        # for j, box in enumerate(bp2['boxes']):
        #     box.set(linewidth=2)
        #     box.set(facecolor=color_list[-1])
        # for whisker in bp2['whiskers']:
        #     whisker.set(linewidth=2)
        # for cap in bp2['caps']:
        #     cap.set(linewidth=2)
        # for median in bp2['medians']:
        #     median.set(linewidth=2)
        # plt.setp(ax2, xticks=[1],
        #          xticklabels=[x_axis_list[-1]])


        plt.title(label)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # ax.tick_params(which='both', width=2, labelsize=16)
        ax.tick_params(which='both', width=2, labelsize=24)
        ax.tick_params(which='major', length=7)
        ax.tick_params(which='minor', length=4)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)

        # ax2.xaxis.set_minor_locator(AutoMinorLocator())
        # ax2.yaxis.set_minor_locator(AutoMinorLocator())
        #
        # ax2.tick_params(which='both', width=2, labelsize=16)
        # ax2.tick_params(which='major', length=7)
        # ax2.tick_params(which='minor', length=4)
        # for axis in ['top', 'bottom', 'left', 'right']:
        #     ax2.spines[axis].set_linewidth(2)
        # ax2.set_ylim(ymin=-0.5)

        fig.tight_layout()
        fig.savefig(load_model + "/" + label + ".png", transparent=True)
        plt.close(fig)


def make_env(render_plot, penalty_coeff, bonus_coeff, energy_coeff, dist_coeff,dist_tol, termination_step, state_type,
             num_targets, obs_num_targets):
    simulation_dir = "simulation/"
    # num_targets = 10
    ground_range = 8.0
    online = True
    env = CoordinationEnv(num_targets=num_targets,
                          obs_num_targets=obs_num_targets,
                          ground_range=ground_range,
                          penalty_coeff=penalty_coeff,
                          bonus_coeff=bonus_coeff,
                          termination_step=termination_step,
                          state_type=state_type,
                          simulation_dir=simulation_dir,
                          simulation_model=simulation_dir + '/0704_model.h5',
                          energy_coeff=energy_coeff,
                          dist_coeff=dist_coeff,
                          dist_tol=dist_tol,
                          render_plot=render_plot, online=online)
    return env


def check_reward():
    data = np.load("./log_step.npy") #[env.n_target_collected, env.energy / 300.0, env.dist, action]
    index_reached = np.where(np.isclose(data[:, 0], 1))[0]
    index_crawl = np.where(np.isclose(data[:, -1], 1))[0]
    index_reach = np.where(np.isclose(data[:, -1], 0))[0]
    index_reachednothing = np.setdiff1d(index_reach, index_reached)

    def reach_function(data):
        return (data[:, 0] - energy_coeff * data[:, 1] - dist_coeff * data[:, 2]) / 2.0 + bonus_coeff

    # reached_reward=(1 - energy_coeff * data[index_reached, 1] - dist_coeff * data[index_reached, 2]) / 2.0 + bonus_coeff
    reached_reward = reach_function(data[index_reached])
    reachednothing_reward = reach_function(data[index_reachednothing])
    crawl_reward = (-energy_coeff * data[index_crawl, 1]) / 2.0 + bonus_coeff

    # mean_reach_reward=(np.sum(reached_reward)+len(index_reachednothing)*(-penalty_coeff))/len(index_reach)
    mean_reach_reward = (np.sum(reached_reward) + np.sum(reachednothing_reward)) / len(index_reach)
    mean_crawling_reward = np.mean(crawl_reward)
    mean_reached_reward = np.mean(reached_reward)
    mean_reached_nothing_reward = np.mean(reachednothing_reward)
    print("mean of reaching reward", mean_reach_reward)
    print("mean of crawling reward", mean_crawling_reward)
    print("mean of reached reward", mean_reached_reward)
    print("mean of reached nothing reward", mean_reached_nothing_reward)

    assert np.any(reached_reward < max(crawl_reward)) == False, "please have: reached reward > crawl reward"
    assert max(reachednothing_reward) < min(reached_reward), "please have: all reached reward > reached nothing reward"
    assert mean_reach_reward > mean_crawling_reward, "please have: reach > crawl"
    assert mean_reached_nothing_reward < mean_reached_reward, "please have: mean reached > mean reached nothing"
    assert mean_reached_nothing_reward < mean_crawling_reward, "please have: mean reached nothing < mean crawling"

    exit()


if __name__ == '__main__':
    map = 0
    eps = 0.2

    state_type = "loc_map"
    total_epochs = 5000
    train = True
    box_plot = False
    obs_num_targets = 8
    num_targets = 40
    lr = 1e-4
    termination_step = 60 * 2
    batch_size = 30
    rollout = 300 * 2

    bonus_coeff = 0.2
    penalty_coeff = 10000.0
    energy_coeff = 1.0
    dist_coeff = 0.4
    dist_tol = 0.05 #5%->0.05
    # check_reward()

    if not box_plot:
        if train:
            agent = Agent(action_size=2, lr=lr,
                          eps=eps, batch_size=batch_size,
                          rollout=rollout,
                          obs_num_targets=obs_num_targets)
            agent.run(n_map=map, num_targets=num_targets, obs_num_targets=obs_num_targets,
                      total_epochs=total_epochs,
                      penalty_coeff=penalty_coeff,
                      bonus_coeff=bonus_coeff,
                      energy_coeff=energy_coeff, dist_coeff=dist_coeff,dist_tol=dist_tol,
                      termination_step=termination_step, state_type=state_type)
        else:
            target_range = []
            n_sample = 100

            env = make_env(render_plot=False, penalty_coeff=penalty_coeff, bonus_coeff=bonus_coeff,
                           energy_coeff=energy_coeff, dist_coeff=dist_coeff,dist_tol=dist_tol,
                           termination_step=termination_step, state_type=state_type, num_targets=num_targets,
                           obs_num_targets=obs_num_targets)
            for _ in range(n_sample):
                env.generate_n_targets(from_n_map=map)
                target_range.append(env.target_range)

            total_epochs = 500

            agent = Agent(action_size=2, lr=lr,
                          eps=eps, batch_size=batch_size, rollout=rollout)

            # load_model = "end2end_logs_20target/t120-target20-epoch2-pen0.100000-bon0.180000-eg1.000000-dist0.100000-lr0.000100-eps0.200-rollout600-batch30-valu10.000000-loc_map20210730-135633/"
            # load_model = "seung_end2end0.05_last/t120-target20-epoch5000-pen10000.000000-bon0.200000-eg1.000000-dist0.400000-lr0.000010-eps0.200-rollout600-batch30-valu10.000000-loc_map20210731-073000/"
            # load_model = "arc_end2end/t120-target20-epoch5000-pen10000.000000-bon0.200000-eg1.000000-dist0.400000-lr0.000050-eps0.200-rollout600-batch30-valu10.000000-loc_map20210801-224137/"
            # load_model = "seung_plus/t120-target20-epoch5000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout600-batch30-valu10.000000-loc_map20210726-144307/"
            load_model = "0823_end2end_40target/t120-target40-epoch5000-pen10000.000000-bon0.200000-eg1.000000-dist0.400000-lr0.000010-eps0.200-rollout600-batch30-valu10.000000-loc_map20210821-210153/"

            target_range = list(
                np.load(load_model + "target%d_test_across_model.npy" % num_targets))  # [96][np.newaxis,:,:])
            # np.save(load_model + "target%d_test_across_model" % num_targets, target_range)

            env.ep = 0
            env.render_dir = load_model + "/frames_ppo/"
            os.makedirs(env.render_dir, exist_ok=True)
            agent.test(env, n_map=0, n_sample=n_sample, load_model=load_model, random=False,
                       target_range=target_range, greedy=False)

            # env.render_dir = load_model + "/frames_random/"
            # os.makedirs(env.render_dir, exist_ok=True)
            # env.ep = 0
            # agent.test(env, n_map=0, n_sample=n_sample, load_model=load_model, random=True,
            #            target_range=target_range, greedy=False)
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
        # load_model = "seung_end2end0.05/t120-target20-epoch5000-pen10000.000000-bon0.200000-eg1.000000-dist0.400000-lr0.000010-eps0.200-rollout600-batch30-valu10.000000-loc_map20210731-073000/no reward/"
        load_model = "0823_end2end_40target/t120-target40-epoch5000-pen10000.000000-bon0.200000-eg1.000000-dist0.400000-lr0.000010-eps0.200-rollout600-batch30-valu10.000000-loc_map20210821-210153/"

        num_targets = 40
        plot_box(load_model, num_targets)
