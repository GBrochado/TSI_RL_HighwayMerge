import torch as th
from torch import nn
import configparser

config_dir = 'configs/configs_ppo.ini'
config = configparser.ConfigParser()
config.read(config_dir)
torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
th.manual_seed(torch_seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True
import time
from torch.optim import Adam, RMSprop
import copy

import numpy as np
import os, logging
from copy import deepcopy
from single_agent.Memory_common import OnPolicyReplayMemory
from common.Model import ActorNetwork, CriticNetwork, ActorCriticNetwork
from common.utils import entropy, index_to_one_hot, to_tensor_var, VideoRecorder
from common.Agent import Agent







class MAPPO(Agent):
    
    
    
    """
    An multi-agent learned with PPO
    reference: https://github.com/ChenglongChen/pytorch-DRL
    """
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=1, target_tau=1.,done_penalty=None,
                 target_update_steps=5, clip_param=0.2,
                 reward_gamma=0.99, reward_scale=20,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.0001, critic_lr=0.0001, test_seeds=0,
                 optimizer_type="rmsprop", entropy_reg=0.01, epsilon=1e-5, alpha=0.99,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 use_cuda=True, traffic_density=1, reward_type="regionalR",
                 state_split=True, shared_network=True, training_strategy="concurrent"
                 ):
        super(MAPPO, self).__init__(env, state_dim, action_dim,
                                    memory_capacity, max_steps,
                                    reward_gamma, reward_scale, done_penalty,
                                    actor_hidden_size, critic_hidden_size, critic_loss,
                                    actor_lr, critic_lr,
                                    optimizer_type, entropy_reg,
                                    max_grad_norm, batch_size, episodes_before_train,
                                    use_cuda)

        assert traffic_density in [1, 2, 3]
        assert reward_type in ["regionalR", "global_R"]
        
        self.reward_type = reward_type
        #self.env = env
        #self.state_dim = state_dim
        #self.action_dim = action_dim
        #self.env_state, self.action_mask = self.env.reset()
        #self.n_episodes = 0
        #self.n_steps = 0
        #self.max_steps = max_steps
        self.test_seeds = test_seeds
        #self.reward_gamma = reward_gamma
        #self.reward_scale = reward_scale
        self.traffic_density = traffic_density
        #self.memory = OnPolicyReplayMemory(memory_capacity)
        #self.actor_hidden_size=actor_hidden_size
        #self.critic_hidden_size=critic_hidden_size
        #self.actor_output_act = actor_output_act
        #self.critic_loss = critic_loss
        #self.actor_lr=actor_lr
        #self.critic_lr=critic_lr
        #self.optimizer_type = optimizer_type
        #self.entropy_reg = entropy_reg
        #self.max_grad_norm = max_grad_norm
        #self.batch_size = batch_size
        #self.episodes_before_train = episodes_before_train
        #self.use_cuda = use_cuda and th.cuda.is_available()
        self.roll_out_n_steps = roll_out_n_steps
        #self.target_tau = target_tau
        #self.target_update_steps = target_update_steps
        #self.clip_param = clip_param
        self.training_strategy = training_strategy
        self.shared_network = shared_network
        #self.dt= 1 / self.env.config["simulation_frequency"]
        
        # maximum number of CAVs in each mode
        if self.traffic_density == 1:
            max_num_vehicle = 3
        elif self.traffic_density == 2:
            max_num_vehicle = 4
        elif self.traffic_density == 3:
            max_num_vehicle = 6
        
        if not self.shared_network:
            """separate actor and critic network"""
            self.actors = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, state_split)

            if self.training_strategy == "concurrent":
                self.critics = CriticNetwork(self.state_dim, self.critic_hidden_size, 1, state_split)
            elif self.training_strategy == "centralized":
                critic_state_dim = max_num_vehicle * self.state_dim
                self.critics = CriticNetwork(critic_state_dim, self.critic_hidden_size, 1, state_split)

            if optimizer_type == "adam":
                self.actor_optimizers = Adam(self.actors.parameters(), lr=self.actor_lr)
                self.critic_optimizers = Adam(self.critics.parameters(), lr=self.critic_lr)
            elif optimizer_type == "rmsprop":
                self.actor_optimizers = RMSprop(self.actors.parameters(), lr=self.actor_lr, eps=epsilon, alpha=alpha)
                self.critic_optimizers = RMSprop(self.critics.parameters(), lr=self.critic_lr, eps=epsilon, alpha=alpha)
            if self.use_cuda:
                self.actors.cuda()
                self.critics.cuda()
        else:
            """An actor-critic network that sharing lower-layer representations but
            have distinct output layers"""
            self.policy = ActorCriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1, state_split)
            if optimizer_type == "adam":
                self.policy_optimizers = Adam(self.policy.parameters(), lr=self.actor_lr)
            elif optimizer_type == "rmsprop":
                self.policy_optimizers = RMSprop(self.policy.parameters(), lr=self.actor_lr, eps=epsilon, alpha=alpha)

            if self.use_cuda:
                self.policy.cuda()

        self.episode_rewards = [0]
        self.average_speed = [0]
        self.epoch_steps = [0]
        
    def update_environment(self, new_env,density):
        """
        Update the agent's environment to a new one.
        """
        self.env = new_env
        #self.env_state, _ = self.env.reset()
        #self.n_steps = 0
        self.traffic_density=density
        #self.n_agents = len(self.env.controlled_vehicles)
    
    
    
    
   
    
    
    # agent interact with the environment to collect experience
    def interact(self):
        
        #print("dt",self.dt)
        #self.env.render()
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state, self.action_mask = self.env.reset()
            self.n_steps = 0
        states = []
        actions = []
        rewards = []
        policies= []
        action_masks= []
        done = True
        average_speed = 0

        self.n_agents = len(self.env.controlled_vehicles)
        
        
        
        for i in range(self.roll_out_n_steps):
            
            states.append(self.env_state)
            action_masks.append(self.action_mask)
            action,policy = self.exploration_action(self.env_state,self.action_mask)
            next_state, global_reward, done, info = self.env.step(tuple(action))
            self.episode_rewards[-1] += global_reward
            self.epoch_steps[-1] += 1
            if self.reward_type == "regionalR":
                reward = info["regional_rewards"]
            elif self.reward_type == "global_R":
                reward = [global_reward] * self.n_agents
            average_speed += info["average_speed"]
            actions.append([index_to_one_hot(a, self.action_dim) for a in action])
            rewards.append(reward)
            policies.append(policy)
            final_state = next_state
            
            self.env_state = next_state
            self.action_mask = info["action_mask"]

            self.n_steps += 1
            if done:
                self.env_state, self.action_mask  = self.env.reset()
                break

        # discount reward
        if done:
            final_value = [0.0] * self.n_agents
            self.n_episodes += 1
            self.episode_done = True
            self.average_speed[-1] = average_speed / self.epoch_steps[-1]
            self.episode_rewards.append(0)
            self.average_speed.append(0)
            self.epoch_steps.append(0)
        else:
            self.episode_done = False
            final_action = self.action(final_state, self.n_agents, self.action_mask)
            one_hot_action = [index_to_one_hot(a, self.action_dim) for a in final_action]
            final_value = self.value(final_state, one_hot_action)

        if self.reward_scale > 0:
            rewards = np.array(rewards) / self.reward_scale

        for agent_id in range(self.n_agents):
            rewards[:, agent_id] = self._discount_reward(rewards[:, agent_id], final_value[agent_id])

        rewards = rewards.tolist()
                
        self.memory.push(states, actions, rewards, policies, action_masks)

    
    
    # train on a roll out batch
    def train(self):
        
        if self.n_episodes <= self.episodes_before_train:
            pass
        #print("entrou")
        batch = self.memory.sample(self.batch_size)
        #print("batch.action_masks:", batch.action_masks)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        action_masks_var = to_tensor_var(batch.action_masks, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)
        whole_states_var = states_var.view(-1, self.n_agents * self.state_dim)
       
        for agent_id in range(self.n_agents):
            if not self.shared_network:
                # update actor network
                self.actor_optimizers.zero_grad()
                action_log_probs = self.actors(states_var[:, agent_id, :], action_masks_var[:, agent_id, :])
                entropy_loss = th.mean(entropy(th.exp(action_log_probs) + 1e-8))
                action_log_probs = th.sum(action_log_probs * actions_var[:, agent_id, :], 1)

                if self.training_strategy == "concurrent":
                    values = self.critics(states_var[:, agent_id, :])
                elif self.training_strategy == "centralized":
                    values = self.critics(whole_states_var)

                advantages = rewards_var[:, agent_id, :] - values.detach()
                pg_loss = -th.mean(action_log_probs * advantages)
                actor_loss = pg_loss - entropy_loss * self.entropy_reg
                actor_loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.actors.parameters(), self.max_grad_norm)
                self.actor_optimizers.step()

                # update critic network
                self.critic_optimizers.zero_grad()
                target_values = rewards_var[:, agent_id, :]
                if self.critic_loss == "huber":
                    critic_loss = nn.functional.smooth_l1_loss(values, target_values)
                else:
                    critic_loss = nn.MSELoss()(values, target_values)
                critic_loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.critics.parameters(), self.max_grad_norm)
                self.critic_optimizers.step()
            else:
                # update actor-critic network
                self.policy_optimizers.zero_grad()
                action_log_probs = self.policy(states_var[:, agent_id, :], action_masks_var[:, agent_id, :])
                entropy_loss = th.mean(entropy(th.exp(action_log_probs) + 1e-8))
                action_log_probs = th.sum(action_log_probs * actions_var[:, agent_id, :], 1)
                values = self.policy(states_var[:, agent_id, :], out_type='v')

                target_values = rewards_var[:, agent_id, :]
                if self.critic_loss == "huber":
                    critic_loss = nn.functional.smooth_l1_loss(values, target_values)
                else:
                    critic_loss = nn.MSELoss()(values, target_values)

                advantages = rewards_var[:, agent_id, :] - values.detach()
                pg_loss = -th.mean(action_log_probs * advantages)
                loss = pg_loss - entropy_loss * self.entropy_reg + critic_loss
                loss.backward()

                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizers.step()

    # discount roll out rewards
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    # predict softmax action based on state
    def _softmax_action(self, state, n_agents,action_mask):
        state_var = to_tensor_var([state], self.use_cuda)  # Add batch dimension if needed
        action_mask_var = to_tensor_var([action_mask], self.use_cuda)
        softmax_action = []
        
        for agent_id in range(n_agents):
            if not self.shared_network:
                softmax_action_var = th.exp(self.actors(state_var[:, agent_id, :], action_mask_var[:, agent_id, :]))
            else:
                softmax_action_var = th.exp(self.policy(state_var[:, agent_id, :], action_mask_var[:, agent_id, :]))
            if self.use_cuda:
                softmax_action.append(softmax_action_var.data.cpu().numpy()[0])
            else:
                softmax_action.append(softmax_action_var.data.numpy()[0])
        return softmax_action

    # predict actions based on state, added random noise for exploration in training
    def exploration_action(self, state, action_mask):
        if self.n_steps == 100:
            print('')
        softmax_actions = self._softmax_action(state, self.n_agents, action_mask)
        policy = []
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
            policy.append(pi)
        return actions, policy

    # choose an action based on state for execution
    def action(self, state, n_agents, action_mask):
        softmax_actions = self._softmax_action(state, n_agents, action_mask)
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action_var = to_tensor_var([action], self.use_cuda)
        whole_state_var = state_var.view(-1, self.n_agents * self.state_dim)
        whole_action_var = action_var.view(-1, self.n_agents * self.action_dim)
        values = [0] * self.n_agents
        
        for agent_id in range(self.n_agents):
            if not self.shared_network:
                """conditions for different action types"""
                if self.training_strategy == "concurrent":
                    value_var = self.critics(state_var[:, agent_id, :])
                elif self.training_strategy == "centralized":
                    value_var = self.critics(whole_state_var)
            else:
                """conditions for different action types"""
                if self.training_strategy == "concurrent":
                    value_var = self.policy(state_var[:, agent_id, :], out_type='v')
                elif self.training_strategy == "centralized":
                    value_var = self.policy(whole_state_var, out_type='v')

            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values

    
    
    
    
    # evaluation the learned agent
    def evaluation(self, env, output_dir, eval_episodes=1, is_train=True):
        rewards = []
        infos = []
        avg_speeds = []
        steps = []
        crash_all=0
        vehicle_speed = []
        vehicle_position = []
        video_recorder = None
        seeds = [int(s) for s in self.test_seeds.split(',')]

        for i in range(eval_episodes):
            avg_speed = 0
            step = 0
            crash=0
            rewards_i = []
            infos_i = []
            done = False
            if is_train:
                if self.traffic_density == 1:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 1)
                elif self.traffic_density == 2:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 2)
                elif self.traffic_density == 3:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 4)
            else:
                state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i])

            n_agents = len(env.controlled_vehicles)
            rendered_frame = env.render(mode="rgb_array")
            video_filename = os.path.join(output_dir,
                                        "testing_episode{}".format(self.n_episodes + 1) + '_{}'.format(i) +
                                        '.mp4')

            # Init video recording with codec 'mp4v'
            if video_filename is not None:
                print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape, 5))
                video_recorder = VideoRecorder(video_filename, rendered_frame.shape[1::-1], 5)
                video_recorder.add_frame(rendered_frame)
            else:
                video_recorder = None

            while not done:
                step += 1
                action = self.action(state, n_agents,action_mask)
                state, reward, done, info = env.step(action)
                if (info["crashed"]):
                    crash+=1
                avg_speed += info["average_speed"]
                action_mask = info.get("action_mask", None)
                rendered_frame = env.render(mode="rgb_array")
                if video_recorder is not None:
                    video_recorder.add_frame(rendered_frame)

                rewards_i.append(reward)
                infos_i.append(info)
            if done:
                self.n_episodes += 1

            vehicle_speed.append(info["vehicle_speed"])
            vehicle_position.append(info["vehicle_position"])
            rewards.append(rewards_i)
            infos.append(infos_i)
            steps.append(step)
            avg_speeds.append(avg_speed / step)
            crash_all+=crash/step

        if video_recorder is not None:
            video_recorder.release()
        env.close()
        return rewards, (vehicle_speed, vehicle_position), steps, avg_speeds,crash_all

    

    # soft update the actor target network or critic target network
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)
            

    def load(self, model_dir, global_step=None, train_mode=False):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = model_dir + save_file
            checkpoint = th.load(file_path)
            print('Checkpoint loaded: {}'.format(file_path))
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            if train_mode:
                self.policy_optimizers.load_state_dict(checkpoint['optimizer_state_dict'])
                self.policy.train()
            else:
                self.policy.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def save(self, model_dir, global_step):
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)
        th.save({'global_step': global_step,
                     'model_state_dict': self.policy.state_dict(),
                     'optimizer_state_dict': self.policy_optimizers.state_dict()},
                    file_path)

import cv2

class VideoRecorder:
    def __init__(self, filename, frame_size, fps):
        # Specify the codec as 'mp4v' for .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)

    def add_frame(self, frame):
        self.video_writer.write(frame)

    def release(self):
        self.video_writer.release()
        
        
