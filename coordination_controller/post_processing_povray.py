import numpy as np

import matplotlib.pyplot as plt
import os
import sys

from povray.rod import POVRAYRod
from povray.ball import POVRAYBall
from draw_body import draw_body

# target_color = ['Coral', 'SpringGreen']  # , 'SpringGreen', 'SlateBlue', 'DarkOrchid', 'LightWood']
target_color = ['Coral', 'SpringGreen', 'SlateBlue', 'DarkOrchid','CornflowerBlue']

import pickle
import numpy as np
from tqdm import tqdm
import shutil

from utility.geometry import y_rotation

num_arms = 4
num_targets = 20
# load_model = "arc_Narms/"+"%darm_logs_%dtarget" % (num_arms, num_targets) +"/t59-target40-epoch10000-pen0.100000-bon0.180000-lr0.000010-eps0.200-rollout600-batch30-valu10.000000-target-wall-action-flag-shift-grid20210912-235509/"
# load_model = "0924_rel_target/2arm_logs_40target_obs0_flip0_arena-diagonal/t59-target40-epoch10000-pen0.100000-bon0.180000-lr0.000010-eps0.200-rollout600-batch30-valu10.000000-food4-obs0-flip0-target-wall-action-flag-shift-grid-diagonal20210924-164418/"
# load_model = "1002_update_target/4arm_logs_60target_obs1_flip1_arena-mid-square/t299-target60-epoch10000-pen0.100000-bon0.180000-lr0.000010-eps0.200-rollout600-batch30-valu10.000000-food4-obs1-flip1-target-wall-action-flag-shift-obstacle-mid-square20211002-123855/"
# load_model = "1002_update_target/4arm_logs_40target_obs0_flip1_arena-diagonal/t59-target40-epoch10000-pen0.100000-bon0.180000-lr0.000010-eps0.200-rollout600-batch30-valu10.000000-food4-obs0-flip1-target-wall-action-flag-shift-grid-diagonal20211002-123250/"
# load_model = "1010/4arm_logs_20target_obs0_flip1_arena-diagonal/t119-target20-epoch10000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout600-batch30-valu10.000000-food4-obs0-flip1-target-wall-action-flag-shift-grid-diagonal20211009-205839/"
# load_model = "1010/4arm_logs_20target_obs0_flip0_arena-mid-square/t59-target20-epoch10000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout600-batch30-valu10.000000-food4-obs0-flip0-pos-target-wall-action-flag-shift-grid-mid-square20211009-174102/"
# load_model = "1010/4arm_logs_20target_obs0_flip0_arena-mid-square/t59-target20-epoch10000-pen0.100000-bon0.180000-lr0.000050-eps0.200-rollout600-batch30-valu10.000000-food4-obs0-flip0-pos-target-wall-action-flag-shift-grid-mid-square20211009-174048/"
# load_model = "1014/4arm_logs_40target_obs0_flip0_arena-mid-square/t59-target40-epoch10000-pen0.100000-bon0.180000-lr0.000050-eps0.200-rollout600-batch30-valu10.000000-food1-obs0-flip0-target-wall-action-flag-shift-grid-mid-square20211014-125521/"
# load_model = "1016/4arm_logs_40target_obs0_flip0_arena-mid-square/t59-target40-epoch10000-pen0.100000-bon0.180000-value0.010000-lr0.001000-eps0.200-rollout600-batch30-valu0.010000-food1-obs0-flip0-target-wall-action-flag-shift-grid-mid-square20211016-094224/"
# load_model = "4arm_logs_20target_obs0_flip0_arena-mid-square/t59-target20-epoch10000-pen0.100000-bon0.180000-value10.000000-lr0.000050-eps0.200-rollout600-batch30-valu10.000000-food1-obs0-flip0-pos-target-wall-action-flag-shift-grid-mid-square20211108-101728/"

# load_model = "1011/4arm_logs_40target_obs0_flip1_arena-diagonal/t119-target40-epoch10000-pen0.100000-bon0.180000-lr0.000100-eps0.200-rollout600-batch30-valu10.000000-food1-obs0-flip1-target-wall-action-flag-shift-grid-diagonal20211010-212720/"
# load_model = "1109/4arm_logs_20target_obs0_flip0_arena-mid-square/t59-target20-epoch10000-pen0.100000-bon0.180000-value10.000000-lr0.000500-eps0.200-rollout600-batch30-valu10.000000-food1-obs0-flip0-pos-target-wall-action-flag-shift-grid-mid-square20211108-115745/"
load_model = "final_4arm/1221_4arm_logs_20target_obs0_flip0_arena-mid-square/t179-target20-epoch15000-pen1.000000-bon0.180000-value1.000000-lr0.000050-eps0.200-rollout1800-batch120-valu1.000000-food5-obs0-flip0-pos-near-target-wall-action-flag-index-no-shift-grid-mid-square20211221-185924/"

eps_max=[6,7,4,5]
# good obs (arm,eps)
# good_obs=[
#     [1], #arm0 - eps 0, 1, 3 #0#3
#     [1,3],
#     [1],#2,3
#     [2]#3
#     ]

draw_dir = os.path.join(load_model , "simple_draw")
os.makedirs(draw_dir, exist_ok=True)

check_activation_dir = os.path.join(load_model , "check_activation_index")
os.makedirs(check_activation_dir, exist_ok=True)
# target_dir = load_model + "/1save_target_activation/"
ep=int(sys.argv[1])
ppo_flag=int(sys.argv[2])
if ppo_flag==1:
    # os.makedirs(load_model + "frames_ppo/ep%d"%ep, exist_ok=True)
    target_dir = load_model + "frames_ppo/ep%d/activation/"%ep#12_cylinderwater_save_target_activation/"
else:
    # os.makedirs(load_model + "frames_simplified/ep%d" % ep, exist_ok=True)
    target_dir = load_model + "frames_simplified/ep%d/activation/" % ep  # 12_cylinderwater_save_target_activation/"
for j in range(num_arms):
    arm_target_dir = target_dir + "/arm%d/" % j
    os.makedirs(arm_target_dir, exist_ok=True)

sample_dir = load_model
simulation_dir = sample_dir

target_scale = 2
generate_povray = True
do_frame = None  # [254,2276,2382,2476,2645,2880]#,  # None will do all
do_povray = True
with_ground = True
if ppo_flag==1:
    data = np.load(load_model + "frames_ppo/ep%d_log_random0_greedy0_simplified0.npz"%ep)
else:
    data = np.load(load_model + "frames_simplified/ep%d_log_random0_greedy0_simplified1.npz"%ep)
data = dict(data)
keys = data.keys()
print("wh")
log_arm_degree = data['log_arm_degree']
target_range = data['target_range'][0]
log_head = data['log_head']
log_arm_head = data['log_arm_head']
log_arm = data['log_arm']
log_BC = data['log_BCs']
log_marks = data['log_marks']
log_targets = data['log_targets']
log_activations = data['log_activations']
log_action = data['log_action']
log_sum_crawl = data['log_sum_crawl']
log_reached_map = data['log_reached_map']
# log_reached=[[0,0,1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,2,0,3,0,0,0,0,0,0,0,0,1,0,3,0,0,0,2,0,0,1,0,1,0,0,0,0,2],
#              [0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0]]
# log_reached=[
#     [1,0,2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
#     [2,0,0,0,2,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
#     [1,0,2,0,0,0,0,0,0,0,0,0,0,0,1,0,2,0,0,0,0,0,0,1,0,0],
#     [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
log_reached = log_reached_map.T

for key in keys:
    print(key, data[key].shape)

arm_length = np.array([1.0, 0.0, 0.0])
crawl_amount = np.array([0.2775787711143494, 0.0, 0.0])
# crawl_amount = np.array([-0.2, 0.0, 0.0])
length_list = [y_rotation(arm_length, log_arm_degree[j]) for j in range(num_arms)]
crawl_list = [y_rotation(crawl_amount, log_arm_degree[j]) for j in range(num_arms)]
total_steps = log_targets.shape[1]



def cal_arm(t):
    abs_arm = []
    for i in range(num_arms):
        pos = log_arm[i][t]
        pos_rot = y_rotation(pos.T, log_arm_degree[i])
        head = log_arm_head[i][t]
        pos_abs = (pos_rot + head).T
        abs_arm.append(pos_abs)
    return np.array(abs_arm)


def cal_arm_from_pos(pos, head, degree):
    pos_rot = y_rotation(pos.T, degree)
    return (pos_rot + head).T


def plot_top_view(ax, abs_arm, change_index):
    for i in range(num_arms):
        ax.plot(abs_arm[i, 0], abs_arm[i, 2])
    ax.plot(real_target[change_index:, 0], real_target[change_index:, 2], 'rx')
    ax.set_xlim([-0.5, 7.0])
    ax.set_ylim([-0.5, 7.0])
    ax.set_aspect('equal')


def plot_side_view(ax, pos, head, arm_index, get):
    p = pos + head[:, np.newaxis]
    ax.plot(p[0], p[1])
    ax.plot(rel_target[arm_index][arm_change_index[arm_index]:, 0],
            rel_target[arm_index][arm_change_index[arm_index]:, 1], 'rx')
    ax.set_title("arm %d get? %d" % (arm_index, get))
    ax.set_xlim([-0.2, 7.0])
    ax.set_ylim([-0.2, 1.2])
    ax.set_aspect('equal')


# real_target_index = np.array(np.where(log_marks == 1))
# real_target=log_targets[real_target_index[0],real_target_index[1],:]
mark = np.sum(log_marks, axis=0)

real_target = []
for i in range(log_marks.shape[1]):
    for j in range(num_arms):
        if log_marks[j, i] == 1:
            real_target.append(log_targets[j, i])
real_target = np.array(real_target)

real_arm_target = []
real_arm_base = []
real_arm_activation = []
real_arm = []
arm_change_index = []
for j in range(num_arms):
    real_arm_target.append([])
    real_arm_base.append([])
    real_arm_activation.append([])
    real_arm.append([])
    arm_change_index.append(0)
for i in range(log_marks.shape[1]):
    for j in range(num_arms):
        if log_marks[j, i] == 1:
            real_arm_target[j].append(log_targets[j, i])
            real_arm_base[j].append(log_arm_head[j, i])
            real_arm_activation[j].append(log_activations[j, i])
            real_arm[j].append(log_arm[j, i])
print("real_arm_base",[len(real_arm_base[i]) for i in range(num_arms)])
print("real_arm_target",[len(real_arm_target[i]) for i in range(num_arms)])
rel_target = []
for j in range(num_arms):
    t = real_arm_target[j]
    if len(t) == 0:
        rel_target += []
    else:
        rel_target += [y_rotation(np.array(t), -log_arm_degree[j])]


def save_activation(generate_cylinder=False):
    k = 0
    povray_target = []
    povray_obstacle = []
    rest_target = np.zeros(3)
    rest_target[0] = 1.0
    for j in range(num_arms):
        arm_target_dir = target_dir + "/arm%d/" % j
        prev_target = np.zeros(3)
        episode_target = []
        episode_base = []
        episode_activation = []
        target_batch = []
        activation_batch = []
        base_batch=[]

        for i in range(len(real_arm_target[j])):
            rel_arm_target = y_rotation(real_arm_target[j][i] - real_arm_base[j][i], -log_arm_degree[j])
            # flag = False
            # while not flag:
            #     plt.plot(rel_arm_target[0],rel_arm_target[1],'rx')
            #     plt.plot(real_arm[j][i][0], real_arm[j][i][1])
            #
            #     theta=np.linspace(0 ,2*np.pi,500)
            #     plt.plot(cylinder_params['start'][0]+cylinder_params['radius']*np.cos(theta),cylinder_params['start'][1]+cylinder_params['radius']*np.sin(theta))
            #     plt.xlim([-0.5,1.5])
            #     plt.xlim([-0.5,1.2])
            #     plt.axis("equal")
            #     plt.savefig(check_activation_dir + "/%d.png" % k)
            #     plt.show()
            #     flag=bool(int(input("good enough? Y -> 1, N-> 0")))
            #     if not flag:
            #         cylinder_params['start'][0]=float(input("start x"))
            #         cylinder_params['start'][1]=float(input("start y"))
            #         cylinder_params['radius']=float(input("radius"))
            #
            #
            # plt.close('all')
            k += 1
            if np.isclose(prev_target, real_arm_base[j][i], atol=1e-6).all():
                target_batch.append(rel_arm_target)
                activation_batch.append(real_arm_activation[j][i])
                base_batch.append(real_arm_base[j][i])
            else:
                target_batch.append(rest_target)
                episode_target.append(target_batch)
                target_batch = [rel_arm_target]

                activation_batch.append(np.zeros((3, 99)))
                episode_activation.append(activation_batch)
                activation_batch = [real_arm_activation[j][i]]

                episode_base.append(base_batch)
                base_batch=[real_arm_base[j][i]]


            prev_target = real_arm_base[j][i]
        target_batch.append(rest_target)
        episode_target.append(target_batch)
        activation_batch.append(np.zeros((3, 99)))
        episode_activation.append(activation_batch)
        episode_base.append(base_batch)
        episode_target = episode_target[1:]
        episode_activation = episode_activation[1:]
        episode_base=episode_base[1:]
        if generate_cylinder and len(episode_target)>0:



            for episode in range(len(episode_target)):
                cylinder_params = {}
                max_target = episode_target[episode][-2]
                max_angle = np.arctan2(max_target[0], max_target[1])
                random_angle=np.random.rand()*(np.pi/2-max_angle)
                print("max angle",max_angle)

                # print(np.arctan2(-max_target[0], max_target[1]),np.arctan2(-max_target[0], max_target[1])*180/np.pi)
                # print(max_target)
                # print(-max_target)

                random_length = np.random.rand() * 0.15 + np.linalg.norm(max_target)
                print("random_length",random_length)
                cylinder_params['start'] = np.array(
                    [random_length * np.cos(random_angle), random_length * np.sin(random_angle),
                     -0.15])  # np.random.rand(3)*0.4+0.4
                # cylinder_params['start'][-1]=-0.15
                print("random angle",random_angle)
                print("obs angle",np.arctan2(cylinder_params['start'][1], cylinder_params['start'][0]))
                cylinder_params['direction'] = np.array([0, 0, 1])
                cylinder_params['normal'] = np.array([1, 0, 0])
                cylinder_params['length'] = 0.3
                cylinder_params['radius'] = np.random.rand() * 0.025+0.025  # +0.2

                np.savez(arm_target_dir + "/ul_activation%03d" % episode,
                         target=episode_target[episode],
                         activation=episode_activation[episode])
                cylinder_params['base'] = episode_base[episode][0]
                flag = False
                while not flag:
                    for t in episode_target[episode]:
                        plt.plot(t[0],t[1],'rx')
                    # plt.plot(rel_arm_target[0],rel_arm_target[1],'rx')
                    # plt.plot(real_arm[j][i][0], real_arm[j][i][1])

                    theta=np.linspace(0 ,2*np.pi,500)
                    plt.plot(cylinder_params['start'][0], cylinder_params['start'][1],'ro',alpha=0.5)
                    plt.plot(cylinder_params['start'][0]+cylinder_params['radius']*np.cos(theta),cylinder_params['start'][1]+cylinder_params['radius']*np.sin(theta))
                    plt.xlim([-0.5,1.5])
                    plt.xlim([-0.5,1.2])
                    plt.axis("equal")
                    plt.savefig(target_dir + "/%d.png" % k)
                    flag=True
                    # plt.show()
                    # flag=bool(int(input("good enough? Y -> 1, N-> 0")))
                    # if not flag:
                    #     cylinder_params['start'][0]=float(input("start x"))
                    #     cylinder_params['start'][1]=float(input("start y"))
                    #     # cylinder_params['radius']=float(input("radius"))


                plt.close('all')
                np.savez(arm_target_dir + "/cylinder%03d" % episode,
                         start=cylinder_params['start'],
                         direction=cylinder_params['direction'],
                         normal=cylinder_params['normal'],
                         length=cylinder_params['length'],
                         radius=cylinder_params['radius'])
        else:
            try:
                for episode in range(len(episode_target)):
                    # if episode in good_obs[j]:
                    cylinder_params=np.load(arm_target_dir + "/cylinder%03d.npz" % episode)
                    cylinder_params=dict(cylinder_params)
                    cylinder_params['base'] = episode_base[episode][0]
                    cylinder_params['line'] = np.repeat(cylinder_params['start'][np.newaxis, :], 6, axis=0)
                    cylinder_params['line'][:, 2] = np.linspace(-0.15, 0.15, 6)
                    cylinder_params['line']= y_rotation(cylinder_params['line'], log_arm_degree[j]) + episode_base[episode][0]
                    povray_obstacle.append(cylinder_params)
            except:
                do_cylinder=[]
        povray_target.append(episode_target)

        # cylinder_list.append()
    return povray_target,povray_obstacle



def render_static_video():
    change_index = 0
    for t in range(total_steps):
        abs_arm = cal_arm(t)

        ax1 = plt.subplot(121)
        plot_top_view(ax1, abs_arm, change_index)
        if mark[t] == 1:
            change_index += 1

        ax2 = plt.subplot(222)
        ax3 = plt.subplot(224)

        ax = [ax2, ax3]
        for i in range(num_arms):
            head = y_rotation(log_arm_head[i, t], -log_arm_degree[i])
            plot_side_view(ax[i], log_arm[i, t], head, i, log_marks[i, t])

        for j in range(num_arms):
            if log_marks[j, t] == 1:
                arm_change_index[j] += 1

        plt.savefig(draw_dir + "/%d.png" % t)
        plt.close('all')


# render_static_video()
# povray_target, povray_obstacle = save_activation()
povray_target, povray_obsacle = save_activation(generate_cylinder=True)
# do_cylinder=list(range(len(povray_obstacle)))
# do_cylinder.remove(5)
# do_cylinder.remove(4)
# do_cylinder.remove(6)
# do_cylinder.remove(10)
# do_cylinder.remove(12)
# do_cylinder.remove(14)
# do_cylinder.remove(9)

import os, sys


def include_parent_folders(parent_folders):
    for parent_folder in parent_folders:
        path = os.path.abspath(__file__)
        for directory in path.split("/")[::-1]:
            if directory == parent_folder:
                break
            path = os.path.dirname(path)
        sys.path.append(path)


include_parent_folders(
    parent_folders=[
        "elastica-python",
        "crawling-reaching",
        "ControlCases",
    ]
)


def export_pov(n_balls, camera_file):
    activation_folder = sample_dir + "/activation/"
    os.makedirs(activation_folder, exist_ok=True)

    python_dyn_folder = sample_dir + "/python_dyn/"
    os.makedirs(python_dyn_folder, exist_ok=True)

    shutil.copy(camera_file, povray_data_folder + "/snake.inc")

    arm_target_dir = target_dir + "/crawl/" + "eps%d_online0_low_Ce0.010_Re0.020_Cr5.000_Rr0.100/" % 0
    filename = arm_target_dir + "/pkl/simulation"

    crawl_target_index = np.load(arm_target_dir + "/index_to_change_target.npy")
    with open(filename + "_data000.pickle", "rb") as f:
        crawl_data = pickle.load(f)
        crawl_rod_data = crawl_data['rods'][0]
        radius_data = crawl_rod_data['radius']

    arm_data = []
    arm_index = []
    for j in range(num_arms):
        arm_data.append([])
        arm_index.append([])
        for i in range(len(povray_target[j])):
            # arm_target_dir = target_dir + "/arm%d/" % j + "eps%d_online0_low_Ce0.010_Re0.020_Cr5.000_Rr0.100/" % i
            arm_target_dir = target_dir + "/arm%d/" % j + "eps%d_online0_low_Ce0.010_Re0.030_Cr5.000_Rr0.010/" % i
            filename = arm_target_dir + "/pkl/simulation"

            target_index = np.load(arm_target_dir + "/index_to_change_target.npy")
            with open(filename + "_data000.pickle", "rb") as f:
                data = pickle.load(f)
                rod_data = data['rods'][0]
            arm_data[j].append(rod_data)
            arm_index[j].append(target_index)

    change_index = 0
    arm_change_index = [0 for _ in range(num_arms)]
    rest_pos = np.zeros((3, 101))
    rest_pos[0, :] = np.linspace(0, 1, 101)
    eps = [0 for _ in range(num_arms)]
    time = 0
    head = log_arm_head[0][0]
    plot_index = [[0, 1, 2], [2, 1, 0]]
    real_arm_target[0]=real_arm_target[0][:-2]
    real_arm_target[1]=real_arm_target[1][:-1]
    # real_arm_target[2]=real_arm_target[2][:-3]
    # real_arm_target[3]=real_arm_target[3][:-1]
    print(log_action[0])
    for t in range(1,15):#range(1,len(log_action)):
        action = log_action[t]
        reach_index = np.where(action == 0)[0]
        # reach_index = np.setdiff1d(reach_index, [1])
        crawl_index = np.where(action == 1)[0]

        # rod_data = [arm_data[r][eps[r]]['position'] for r in reach_index]

        # for r in reach_index:
        rod_data = []
        rod_time = []
        rod_index = []
        print()
        print(t)
        print("reach_index", reach_index)
        if len(reach_index) > 0:
            # a = real_target - head
            # target_in_arm1=np.isclose(a[:, 2], 0.0).any()
            # target_in_arm2=np.isclose(a[:, 0], 0.0).any()
            print("reached?",log_reached_map[t])
            target_in_arm = [log_reached[j][t] > 0 for j in range(num_arms)]
            reach_in_arm = np.array(target_in_arm)[reach_index]
            # if ((target_in_arm1 and 0 in reach_index) or (target_in_arm2 and 1 in reach_index)):
            if any(reach_in_arm):
                print("target_in_arm",target_in_arm)
                print("reach_in_arm",np.array(target_in_arm)[reach_index])

                reach_index=np.where(log_reached_map[t]!=0)[0]
                max_reach_time = np.max([len(arm_data[r][eps[r]]['position']) for r in reach_index])
                for j in range(num_arms):
                    rest_data = cal_arm_from_pos(rest_pos, head, log_arm_degree[j])
                    data_zeros = np.zeros((max_reach_time, 3, 101))
                    time_zeros = np.zeros(max_reach_time)
                    print(j,"r",j in reach_index)
                    print(j,"t",target_in_arm[j])
                    if (j in reach_index) and target_in_arm[j]:
                        print("real_arm_base", real_arm_base[j][eps[j]])
                        data = np.array(arm_data[j][eps[j]]['position']) / 0.2
                        arm_time = arm_data[j][eps[j]]['time']
                        rod_index.append(np.concatenate([arm_index[j][eps[j]],[50.0]]))
                        for dyn_t in range(len(data)):
                            reach_data = cal_arm_from_pos(data[dyn_t], head, log_arm_degree[j])
                            data_zeros[dyn_t, ...] = reach_data
                            # if arm_time[dyn_t]>arm_index[j][eps[j]]:
                            #     change_index+=1
                        data_zeros[len(data):, ...] = rest_data
                        time_zeros[:len(arm_time)] = arm_time
                        # change_index += len(povray_target[j][eps[j]])
                        eps[j] +=1# min(eps[j]+1,eps_max[j])
                    else:
                        data_zeros[:, ...] = rest_data
                        rod_index.append([50])
                    rod_data.append(data_zeros)
                    rod_time.append(time_zeros)
                # print(rod_data[0][0,:,[0,-1]])
                # print(rod_data[1][0,:,[0,-1]])
                # exit()

        if len(rod_data) > 0:
            start_index = [0 for _ in range(num_arms)]
            for j in range(len(rod_data[0])):
                for i in range(num_arms):
                    if rod_time[i][j] > rod_index[i][start_index[i]]:
                        start_index[i] += 1
            ball = []
            for j in range(num_arms):
                ball.append(real_arm_target[j][arm_change_index[j]:])
                arm_change_index[j] += start_index[j]
            print("ball", len(ball[0]), len(ball[1]))

            time = povray_one_step(rod_data, ball, radius_data, time, rod_time, rod_index)

            # for k in range(max_reach_time):
            #     # ax1 = plt.subplot()
            #     ax1 = plt.subplot(121)
            #     ax2 = plt.subplot(222)
            #     ax3 = plt.subplot(224)
            #     ax = [ax2, ax3]
            #
            #     for j in range(2):#num_arms):
            #         arm=rod_data[j][k]
            #         rel_arm=arm[[plot_index[j][0],plot_index[j][1]],:]
            #
            #         ax1.plot(arm[0],arm[2])
            #
            #         obs = povray_obstacle[6]['start']
            #
            #         ax1.plot(obs[0],obs[2],'bx')
            #         ax[j].plot(rel_arm[0],rel_arm[1])
            #         ax[j].plot(obs[0],obs[1],'bx')
            #
            #         arm_target = rel_target[j][arm_change_index[j]:,]
            #         abs_arm_target = np.array(real_arm_target[j][arm_change_index[j]:])
            #
            #
            #
            #         ax[j].plot(arm_target[:,0],arm_target[:,1],'rx')
            #         ax1.plot(abs_arm_target[:,0],abs_arm_target[:,2],'rx')
            #         # ax1.plot(target_range[:,0],target_range[:,2],'bo',alpha=0.5)
            #
            #     plt.savefig(python_dyn_folder+"/%d.png"%(time+k))
                # plt.show()
                # exit()
            #     plt.close('all')
            for j in range(num_arms):
                #     arm_change_index[j] += log_reached[j][t-1]
                change_index += log_reached[j][t - 1]
            # time+=k
        crawl_or_not=log_sum_crawl[t]
        if crawl_or_not:
            if 0 in crawl_index and 2 in crawl_index:
                crawl_index=np.setdiff1d(crawl_index,[0,2])
            if 1 in crawl_index and 3 in crawl_index:
                crawl_index=np.setdiff1d(crawl_index,[1,3])
            for j in crawl_index:

                rod_data = []
                crawl_data = np.array(crawl_rod_data['position']) / 0.2
                crawl_time = len(crawl_data)
                for dyn_t in range(crawl_time):
                    data = crawl_data[dyn_t]
                    dyn_head = y_rotation(data[:, 0], log_arm_degree[j]) + head
                    arm_info = np.zeros((num_arms, 3, 101))
                    for k in range(num_arms):
                        if k == j:
                            arm_info[k, ...] = cal_arm_from_pos(data, head, log_arm_degree[j])
                        else:
                            arm_info[k, ...] = cal_arm_from_pos(rest_pos, dyn_head, log_arm_degree[k])
                    rod_data.append(arm_info)
                rod_data = np.swapaxes(rod_data, 0, 1)

                ball = []
                for k in range(num_arms):
                    ball.append(real_arm_target[k][arm_change_index[k]:])
                time = povray_one_step(rod_data, ball, radius_data, time)
                # for k in range(crawl_time):
                #     for l in range(num_arms):
                #         arm = rod_data[l][k]
                #         plt.plot(arm[0], arm[2])
                #     plt.plot(real_target[change_index:, 0], real_target[change_index:, 2], 'rx')
                #     plt.savefig(python_dyn_folder + "/%d.png" % (time + k))
                #     plt.close('all')
                # time += k
                head += crawl_list[j]
        print("time",time)
        print("head", head)
        print("log_head",log_head[t])
        # print("log_arm_head",log_arm_head[t])
        print("eps", eps)
        print("action", action, len(log_action), t)
        # print("mark",log_marks[:,])
        print("reached", log_reached[0][t - 1], log_reached[1][t - 1])
        print("targets", [arm_change_index[i] for i in range(num_arms)])
        print("eps", eps)
        print("sum_crawl",crawl_or_not)
        print("crawl_index",crawl_index)

        # povray_one_step(rod_data, n_balls, target_index)
        # if mark[t] == 1:
        #     change_index += 1
        #
        # ax2 = plt.subplot(222)
        # ax3 = plt.subplot(224)
        #
        # ax = [ax2, ax3]
        # for i in range(num_arms):
        #     head = y_rotation(log_arm_head[i, t], -log_arm_degree[i])
        #     plot_side_view(ax[i], log_arm[i, t], head, i, log_marks[i, t])
        #
        # for j in range(num_arms):
        #     if log_marks[j, t] == 1:
        #         arm_change_index[j] += 1

        # plt.savefig(draw_dir + "/%d.png" % t)
        # plt.close('all')


def povray_one_step(rod_data, ball, radius_data, save_index, rod_time=None, rod_index=None):
    n_balls = [len(ball[i]) for i in range(len(ball))]

    time_length = len(rod_data[0])
    povray_rod = []
    for j in range(num_arms):
        povray_rod.append(POVRAYRod("<0.00,0.50,1.00>"))

    povray_ball = [[] for _ in range(num_arms)]
    for j in range(num_arms):
        for i in range(n_balls[j]):
            povray_ball[j].append(POVRAYBall(target_color[0]))
    fail_povray_ball=POVRAYBall(target_color[0])
    povray_obs = []
    for j in range(len(povray_obstacle)):
        povray_obs.append(POVRAYRod(target_color[-1]))

    start_index = [0 for _ in range(num_arms)]

    # do_frame = tqdm(range(time_length))
    skip=20
    do_frame = tqdm(np.concatenate([range(0,time_length,skip),[time_length-1]]))

    for k in do_frame:
        with open(povray_data_folder + 'frame%04d.inc' % (save_index), 'w') as file_inc:
            # plot arm
            for j in range(num_arms):
                plot_body=True if j==1 else False
                rod_info = povray_rod[j].generate_string(
                    rod_data[j][k] * 0.2, radius_data[0], 0.0, plot_body=plot_body,
                )
                file_inc.writelines(
                    rod_info[0]
                )
            if with_ground:
                draw_body(save_index, povray_data_folder, rod_info[1][0][0], rod_info[1][0][2])

            # plot ball
            for j in range(num_arms):
                for i in range(start_index[j], n_balls[j]):
                    target_info = povray_ball[j][i].generate_string(
                        ball[j][i] * 0.2, 0.004 * target_scale, 0
                    )
                    file_inc.writelines(
                        target_info[0]
                    )
            fail_target=real_arm_target[0][1]
            target_info = fail_povray_ball.generate_string(
                fail_target * 0.2, 0.004 * target_scale, 0
            )
            file_inc.writelines(
                target_info[0]
            )
            fail_target = real_arm_target[1][2]
            target_info = fail_povray_ball.generate_string(
                fail_target * 0.2, 0.004 * target_scale, 0
            )
            file_inc.writelines(
                target_info[0]
            )
            fail_target = real_arm_target[3][2]
            target_info = fail_povray_ball.generate_string(
                fail_target * 0.2, 0.004 * target_scale, 0
            )
            file_inc.writelines(
                target_info[0]
            )

            # plot obs
            if len(do_cylinder) > 0:
                for j in range(len(povray_obstacle)):
                    if j in do_cylinder:
                        r=np.repeat(povray_obstacle[j]['radius'],5)
                        rod_info = povray_obs[j].generate_string(
                            povray_obstacle[j]['line'].T * 0.2, r*0.5*0.2 * target_scale, 0.0, plot_body=False,
                        )
                        file_inc.writelines(
                            rod_info[0]
                        )

                # target_info = povray_obs[j].generate_string(
                #     povray_obstacle[j]['start'] * 0.2, povray_obstacle[j]['radius']*0.5*0.2 * target_scale, 0.8
                # )
                # file_inc.writelines(
                #     target_info[0]
                # )

            if rod_time != None:
                for i in range(num_arms):
                    if rod_time[i][min(k+skip,len(rod_time[i])-1)] > rod_index[i][start_index[i]]:
                        start_index[i] += 1
                        print(k)

                change_index = np.sum(start_index)
            #     change_index += 1
            #     print(start_index, real_target[start_index])



        with open(povray_data_folder + 'frame%04d.pov' % save_index, 'w') as file_pov:
            file_pov.writelines('#include \"' + povray_data_folder + 'snake.inc\"\n')
            file_pov.writelines('#include \"' + povray_data_folder + 'frame%04d.inc\"\n' % (save_index))
            if with_ground:
                file_pov.writelines('#include \"' + povray_data_folder + 'frame_body%04d.inc\"\n' % (save_index))
        save_index += 1
    return save_index
    # exit()


def do_povray():
    # if do_povray:
    import subprocess
    skip=1
    for k in [394]:#range(396,410):#range(484):#range(621,863):
        # subprocess.run(
        #     ["povray", "-H270", "-W480", "Quality=11", "Antialias=on", "+ua", povray_data_folder + "frame%04d.pov" % (k)])
        # subprocess.run(["povray", "-H1080", "-W1920", "Quality=11", "Antialias=on", "+ua",
        #                 povray_data_folder + "frame%04d.pov" % k])
        # subprocess.run(["povray", "-H200", "-W290", "Quality=11", "Antialias=on", "+ua",
        #                 povray_data_folder + "frame%04d.pov" % k])
        # subprocess.run(["povray", "-H640", "-W1920", "Quality=11", "Antialias=on", "+ua",
        #                 povray_data_folder + "frame%04d.pov" % k])
        subprocess.run(["povray", "-H940", "-W1920", "Quality=11", "Antialias=on", "+ua",
                        povray_data_folder + "frame%04d.pov" % k])
        # subprocess.run(["povray", "-H2160", "-W3840", "Quality=11", "Antialias=on", "+ua", povray_data_folder+"frame%04d.pov" % k])

    # ffmpeg -r 40 -i frame%04d.png -b:v 90M -c:v libx264 -pix_fmt yuv420p -f mov -y longitudinal_muscle.mov


# povray_data_folder = "./python_dyn/"
# # os.makedirs(povray_data_folder, exist_ok=True)
# export_pov(len(real_target),camera_file="./snake-darkline.inc")
# povray_data_folder = "./povray_muscles-side2/"
# os.makedirs(povray_data_folder, exist_ok=True)
# export_pov(len(real_target),camera_file="./snake-%darm-side.inc"%num_arms)
# # do_povray()
# povray_data_folder = "./povray_muscles-top10/"
# os.makedirs(povray_data_folder, exist_ok=True)
# # export_pov(len(real_target), camera_file="./snake-%darm-top2.inc" % num_arms)
# do_povray()
# povray_data_folder = "./povray_muscles-side10-1/"
# os.makedirs(povray_data_folder, exist_ok=True)
# # export_pov(len(real_target), camera_file="./snake-%darm-side.inc" % num_arms)
# do_povray()
