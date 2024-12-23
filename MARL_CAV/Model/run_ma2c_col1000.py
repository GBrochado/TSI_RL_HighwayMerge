from __future__ import print_function, division
from MAA2C import MAA2C
from common.utils import agg_double_list, copy_file, init_dir
from datetime import datetime

import argparse
import configparser
import sys
sys.path.append("../highway-env")

import gym
import os
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


def parse_args():
    
    default_base_dir = "./results/"
    default_config_dir = 'configs/configs_col1000.ini'
    parser = argparse.ArgumentParser(description=('Train or evaluate policy on RL environment '
                                                  'using MA2C'))
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--option', type=str, required=False,
                        default='train', help="train or evaluate")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--model-dir', type=str, required=False,
                        default='', help="pretrained model path")
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="random seeds for evaluation, split by ,")
    args = parser.parse_args()
    return args


def train(args):
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # create an experiment folder
    now = "ma2c_col1000"    #datetime.now().strftime("%b-%d_%H_%M_%S")
    output_dir = base_dir + now
    dirs = init_dir(output_dir)
    copy_file(dirs['configs'])

    if os.path.exists(args.model_dir):
        model_dir = args.model_dir
    else:
        model_dir = dirs['models']

    # model configs
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    training_strategy = config.get('MODEL_CONFIG', 'training_strategy')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')
    epsilon = config.getfloat('MODEL_CONFIG', 'epsilon')
    alpha = config.getfloat('MODEL_CONFIG', 'alpha')
    state_split = config.getboolean('MODEL_CONFIG', 'state_split')
    shared_network = config.getboolean('MODEL_CONFIG', 'shared_network')
    reward_type = config.get('MODEL_CONFIG', 'reward_type')
    

    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    MAX_EPISODES = config.getint('TRAIN_CONFIG', 'MAX_EPISODES')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    EVAL_INTERVAL = config.getint('TRAIN_CONFIG', 'EVAL_INTERVAL')
    EVAL_EPISODES = config.getint('TRAIN_CONFIG', 'EVAL_EPISODES')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')
    curriculum=config.getboolean('TRAIN_CONFIG','Curriculum')

    # init env
    env = gym.make('merge-multi-agent-v0')
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    if curriculum:
        env.config['traffic_density']=1
        traffic_density=1
        
    else:
        env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    
    
    env.config['safety_guarantee'] = config.getboolean('ENV_CONFIG', 'safety_guarantee')
    env.config['n_step'] = config.getint('ENV_CONFIG', 'n_step')
    traffic_density = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')

    assert env.T % ROLL_OUT_N_STEPS == 0

    env_eval = gym.make('merge-multi-agent-v0')
    env_eval.config['seed'] = config.getint('ENV_CONFIG', 'seed') + 1
    env_eval.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env_eval.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env_eval.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env_eval.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env_eval.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env_eval.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env_eval.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env_eval.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    if curriculum:
        env_eval.config['traffic_density']=1        
    else:
        env_eval.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env_eval.config['safety_guarantee'] = config.getboolean('ENV_CONFIG', 'safety_guarantee')
    env_eval.config['n_step'] = config.getint('ENV_CONFIG', 'n_step')
    env_eval.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')

    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds
    
    
 
    ma2c = MAA2C(env, state_dim=state_dim, action_dim=action_dim,
                 memory_capacity=MEMORY_CAPACITY, max_steps=None,
                 roll_out_n_steps=ROLL_OUT_N_STEPS,
                 reward_gamma=reward_gamma, reward_scale=reward_scale, done_penalty=None,
                 actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                 actor_lr=actor_lr, critic_lr=critic_lr,
                 optimizer_type="rmsprop", entropy_reg=ENTROPY_REG,
                 max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
                 episodes_before_train=EPISODES_BEFORE_TRAIN,
                 use_cuda=False, training_strategy=training_strategy,
                 epsilon=epsilon, alpha=alpha, traffic_density=traffic_density, test_seeds=test_seeds,
                 state_split=state_split, shared_network=shared_network, reward_type=reward_type)

    # load the model if exist
    ma2c.load(model_dir, train_mode=True)
    env.seed = env.config['seed']
    env.unwrapped.seed = env.config['seed']
    # print(env.seed)
    episodes = []
    eval_rewards = []
    best_eval_reward = -100
    eval_speeds=[]
    eval_crash=[]
    
    
    # Initialize rewards_mu with a default value
    rewards_mu = 0.0
    avg_speeds=0.0
    crashes=0.0

    # Start time for time estimation
    start_time = datetime.now()
    writer = SummaryWriter(log_dir=output_dir + '/logs')
    with tqdm(total=MAX_EPISODES, desc="Training Progress") as pbar:
        while ma2c.n_episodes < MAX_EPISODES:
            ma2c.explore()
            if ma2c.n_episodes >= EPISODES_BEFORE_TRAIN:
                ma2c.train()

            if ma2c.episode_done and ((ma2c.n_episodes + 1) % EVAL_INTERVAL == 0): # EVAL_INTERVAL
                rewards, _, _, speeds,crashes = ma2c.evaluation(env_eval, dirs['train_videos'], EVAL_EPISODES)
                rewards_mu, rewards_std = agg_double_list(rewards)
                avg_speeds,_ = agg_double_list(speeds)
                #print("Episode %d, Average Reward %.2f" % (ma2c.n_episodes + 1, rewards_mu))
                episodes.append(ma2c.n_episodes + 1)
                eval_rewards.append(rewards_mu)
                eval_speeds.append(avg_speeds)
                eval_crash.append(crashes)
                writer.add_scalar("Average Reward", rewards_mu, ma2c.n_episodes)
                writer.add_scalar("Average Speed", avg_speeds, ma2c.n_episodes)
                writer.add_scalar("Collision rate", crashes, ma2c.n_episodes)
                # save the model
                if rewards_mu > best_eval_reward:
                    ma2c.save(dirs['models'], 100000)
                    ma2c.save(dirs['models'], ma2c.n_episodes + 1)
                    best_eval_reward = rewards_mu
                else:
                    ma2c.save(dirs['models'], ma2c.n_episodes + 1)
                    
                
                  
                if  traffic_density < 3 and rewards_mu >= 30: 
                    traffic_density += 1
                    print(f"Increasing traffic density to {traffic_density}.")
                    
                    
                    # Update traffic density in environments
                    env.config['traffic_density'] = traffic_density
                    env_eval.config['traffic_density'] = traffic_density

                    # Update the MAA2C instance with the new environment
                    ma2c.update_environment(env,traffic_density)
                    #print("controlled",len(env.controlled_vehicles))
                        
                    
                        
                        
            np.save(output_dir + '/{}'.format('eval_rewards'), np.array(eval_rewards))
            # save training data
            np.save(output_dir + '/{}'.format('episode_rewards'), np.array(ma2c.episode_rewards))
            np.save(output_dir + '/{}'.format('epoch_steps'), np.array(ma2c.epoch_steps))
            np.save(output_dir + '/{}'.format('average_speed'), np.array(ma2c.average_speed))
            
            # Update tqdm progress
            pbar.update(1)
            pbar.set_postfix({
                "episode": ma2c.n_episodes + 1,
                "average_reward": f"{rewards_mu:.2f}", 
                "average speed":  f"{avg_speeds:.2f}",
                "collision":  f"{crashes:.2f}", 
                "elapsed_time": str(datetime.now() - start_time).split('.')[0]
            })

    # save the model
    ma2c.save(dirs['models'], MAX_EPISODES + 2)

    plt.figure()
    plt.plot(eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Training Reward")
    plt.legend(["MAA2C"])
    plt.savefig(output_dir + '/' + "maa2c_train_reward.png")
    plt.show()
    
    plt.figure()
    plt.plot(eval_speeds)
    plt.xlabel("Episode")
    plt.ylabel("Average Speed")
    plt.legend(["MAA2C"])
    plt.savefig(output_dir + '/' + "maa2c_train_speed.png")
    plt.show()
    
    plt.figure()
    plt.plot(eval_crash)
    plt.xlabel("Episode")
    plt.ylabel("Collision Rate")
    plt.legend(["MAA2C"])
    plt.savefig(output_dir + '/' + "maa2c_train_collision.png")
    plt.show()


def evaluate(args):
    if os.path.exists(args.model_dir):
        model_dir = args.model_dir + '/models/'
    else:
        raise Exception("Sorry, no pretrained models")
    config_dir = args.model_dir + '/configs/configs.ini'
    config = configparser.ConfigParser()
    config.read(config_dir)

    video_dir = args.model_dir + '/eval_videos'
    eval_logs = args.model_dir + '/eval_logs'

    # model configs
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    training_strategy = config.get('MODEL_CONFIG', 'training_strategy')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')
    epsilon = config.getfloat('MODEL_CONFIG', 'epsilon')
    alpha = config.getfloat('MODEL_CONFIG', 'alpha')
    state_split = config.getboolean('MODEL_CONFIG', 'state_split')
    shared_network = config.getboolean('MODEL_CONFIG', 'shared_network')
    reward_type = config.get('MODEL_CONFIG', 'reward_type')

    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')

    # init env
    env = gym.make('merge-multi-agent-v0')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['safety_guarantee'] = config.getboolean('ENV_CONFIG', 'safety_guarantee')
    env.config['n_step'] = config.getint('ENV_CONFIG', 'n_step')
    traffic_density = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')

    assert env.T % ROLL_OUT_N_STEPS == 0
    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds
    seeds = [int(s) for s in test_seeds.split(',')]

    ma2c = MAA2C(env, state_dim=state_dim, action_dim=action_dim,
                 memory_capacity=MEMORY_CAPACITY, max_steps=None,
                 roll_out_n_steps=ROLL_OUT_N_STEPS,
                 reward_gamma=reward_gamma, reward_scale=reward_scale, done_penalty=None,
                 actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                 actor_lr=actor_lr, critic_lr=critic_lr,
                 optimizer_type="rmsprop", entropy_reg=ENTROPY_REG,
                 max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
                 episodes_before_train=EPISODES_BEFORE_TRAIN,
                 use_cuda=False, training_strategy=training_strategy,
                 epsilon=epsilon, alpha=alpha, traffic_density=traffic_density, test_seeds=test_seeds,
                 state_split=state_split, shared_network=shared_network, reward_type=reward_type)

    # load the model if exist
    ma2c.load(model_dir, train_mode=False)
    rewards, (vehicle_speed, vehicle_position), steps, avg_speeds = ma2c.evaluation(env, video_dir, len(seeds), is_train=False)
    rewards_mu, rewards_std = agg_double_list(rewards)
    success_rate = sum(np.array(steps) == 100) / len(steps)
    avg_speeds_mu, avg_speeds_std = agg_double_list(avg_speeds)

    print("Evaluation Reward and std %.2f, %.2f " % (rewards_mu, rewards_std))
    print("Collision Rate %.2f" % (1 - success_rate))
    print("Average Speed and std %.2f , %.2f " % (avg_speeds_mu, avg_speeds_std))

    np.save(eval_logs + '/{}'.format('eval_rewards'), np.array(rewards))
    np.save(eval_logs + '/{}'.format('eval_steps'), np.array(steps))
    np.save(eval_logs + '/{}'.format('eval_avg_speeds'), np.array(avg_speeds))
    np.save(eval_logs + '/{}'.format('vehicle_speed'), np.array(vehicle_speed))
    np.save(eval_logs + '/{}'.format('vehicle_position'), np.array(vehicle_position))


if __name__ == "__main__":
    args = parse_args()
    # train or eval
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
