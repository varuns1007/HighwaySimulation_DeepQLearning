
from env import HighwayEnv, ACTION_NO_OP, get_highway_env
import random
import numpy as np
from collections import deque
from typing import Tuple
import argparse
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha=0.6):
        self.buffer = deque(maxlen=buffer_size)
        self.alpha = alpha  # Controls prioritization strength

    def add_experience(self, state, action, reward, next_state, done, td_error=1.0):
        experience = (state, action, reward, next_state, done, td_error)
        self.buffer.append(experience)

    def sample(self, batch_size):
        priorities = np.array([abs(exp[5]) for exp in self.buffer])  # Use TD error
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        sample_indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[i] for i in sample_indices]

        states, actions, rewards, next_states, dones, td_errors = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), \
               np.array(next_states), np.array(dones), np.array(sample_indices)

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.buffer[idx] = (*self.buffer[idx][:5], abs(error) + 1e-5)  # Update TD error

    def size(self):
        return len(self.buffer)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def add_experience(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        sample_batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*sample_batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def size(self):
        return len(self.buffer)
    

class DQN(nn.Module):
    def __init__(self, in_features, out_features):
        super(DQN, self).__init__()

        self.input_dim = in_features
        self.out_dim = out_features

        self.fc1 = nn.Linear(*self.input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, self.out_dim)
        
        self.device = device
        self.to(self.device)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x


'''
import _ Agent
agent = Agent()
env = Env()
agent.train_policy(iterations = 1000)
 
'''
class DQNAgent:

    def __init__(self, 
                    env: HighwayEnv, 
                    alpha: float = 0.1, 
                    eps: float = 0.75, 
                    discount_factor: float = 0.99,
                    tau=0.005,
                    iterations: int = 100000, 
                    eps_type: str = 'constant',
                    validation_runs: int = 100,
                    validate_every: int = 50000,
                    visualize_runs: int = 10, 
                    visualize_every: int = 50000,
                    log_folder:str = './',
                    lr = 0.0001,
                    batch_size = 256,
                    eps_decay = (1 - 1e-4),
                    min_eps = 0.01
                    ):

        #TODO: You can add you code here
        self.env = env

        # For Soft update 
        self.tau = tau

        self.eps = eps
        self.eps_decay = eps_decay
        self.min_eps= min_eps

        self.df = discount_factor
        self.alpha = alpha
        self.iterations = iterations
        self.validation_runs = validation_runs
        self.validate_every = validate_every
        self.visualization_runs = visualize_runs
        self.visualization_every = visualize_every
        
        self.log_folder = log_folder
        os.makedirs(log_folder, exist_ok=True)

        # PER buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size=100000)
        
        # State variables = 6, Possible Actions = 5/env.num_actions
        self.dqn = DQN([6], self.env.num_actions)
        self.target_dqn = deepcopy(self.dqn)
        self.update_network_every = 100

        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), self.lr)
        self.loss = nn.MSELoss()


    def choose_action(self, state, greedy = False):

        '''
        Right now returning random action but need to add
        your own logic
        '''
        #TODO: You can add you code here

        action = None
    
        self.target_dqn.eval()
        with torch.no_grad():
            state_tensor = torch.tensor([state]).to(self.target_dqn.device).float()
            q_values = self.target_dqn(state_tensor)
            action = torch.argmax(q_values).item()

        if greedy:
            if np.random.rand() < self.eps:
                action = np.random.randint(0, 5)
                return action

            else:
                return action
            
        else:
            return action


    def validate_policy(self) -> Tuple[float, float]:
        '''
        Returns:
            tuple of (rewards, dist)
                rewards: average across validation run 
                        discounted returns obtained from first state
                dist: average across validation run 
                        distance covered by control car
        '''
        rewards = []
        dist = []

        for i in range(self.validation_runs):
            
            obs = self.env.reset(i) #don't modify this
            
            #TODO: You can add you code here
            done = False

            # Starting of highway environment, Control car has position set to 2
            initial_car_pos = self.env.control_car.pos

            total_reward = 0
            discount_factor = 1
            distance = 0

            while not done:
                action = self.choose_action(obs)
                next_obs, reward, done, _ = self.env.step(action)

                total_reward += discount_factor * reward  # Discounted return
                discount_factor *= self.df  # Apply gamma discounting

                distance = self.env.control_car.pos - initial_car_pos

                obs = next_obs

            rewards.append(total_reward)
            dist.append(distance)


        return sum(rewards) / len(rewards), sum(dist) / len(dist)

    def visualize_policy(self, i: int) -> None:
        '''
        Args:
            i: total iterations done so for
        
        Create GIF visulizations of policy for visualization_runs
        '''

        for j in range(self.visualization_runs):
            obs = self.env.reset(j)  #don't modify this
            done = False
            images = [self.env.render()]

            #TODO: You can add you code here

            while not done:
                action = self.choose_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                next_obs = tuple(next_obs)
                current_frame = self.env.render()
                images.append(current_frame)
                obs = next_obs

            images = [Image.fromarray(img) for img in images]
            images[0].save(
                f"{self.log_folder}/output_{j+1}th_trajectory.gif",
                save_all=True,
                append_images=images[1:],
                duration=200,  # Time per frame in milliseconds
                loop=0,  # Loop forever
                optimize=True  # Optimize GIF for smaller file size
            )

    def visualize_lane_value(self, i:int) -> None:
        '''
        Args:
            i: total iterations done so for
        
        Create image visulizations for no_op actions for particular lane
        '''
        
        lane_outputs = f"{self.log_folder}/visualize_lane"
        os.makedirs(lane_outputs, exist_ok=True)

        for j in range(self.visualization_runs // 2):
            self.env.reset(j) #don't modify this
            done = False
            k = 0
            
            trajectory_imgs = []

            while(not done):
                k += 1
                _ , _, done, _ = self.env.step(ignore_control_car = True)
                
                if(k % 20 == 0):

                    qvalues = []
                    states = self.env.get_all_lane_states()
                    # print(len(states))
                    
                    #TODO: You can add you code here
                    for st in states:
                        st = torch.tensor(st).to(self.target_dqn.device).float()
                        q_v = self.target_dqn(st)
                        q_v = q_v.detach().numpy()
                        qvalues.append(q_v[ACTION_NO_OP])
                    
                    image = self.env.render_lane_state_values(qvalues)
                    image = Image.fromarray(image)

                    trajectory_imgs.append(image)

                    # Folder for same trajectory
                    trajectory_folder = f"{lane_outputs}/{j}"
                    os.makedirs(trajectory_folder, exist_ok=True)

                    image.save(f"{trajectory_folder}/{k}.png")
            
            
            trajectory_imgs[0].save(
                f"{trajectory_folder}/{j+1}th_trajectory.gif",
                save_all=True,
                append_images=trajectory_imgs[1:],
                duration=200,  # Time per frame in milliseconds
                loop=0,  # Loop forever
                optimize=True  # Optimize GIF for smaller file size
            )

    def visualize_speed_value(self, i:int) -> None:
        '''
        Args:
            i: total iterations done so for
        
        Create image visulizations for no_op actions for particular lane
        '''
        
        speed_outputs = f"{self.log_folder}/visualize_speed"
        os.makedirs(speed_outputs, exist_ok=True)

        for j in range(self.visualization_runs // 2):
            self.env.reset(j) #don't modify this
            done = False
            k = 0

            trajectory_imgs = []

            while(not done):
                k += 1
                _ , _, done, _ = self.env.step(ignore_control_car = True)
                
                if(k % 20 == 0):

                    qvalues = []
                    states = self.env.get_all_speed_states()
                    # print(len(states))

                    #TODO: You can add you code here
                    for st in states:
                        st = torch.tensor(st).to(self.target_dqn.device).float()
                        q_v = self.target_dqn(st)
                        q_v = q_v.detach().numpy()
                        qvalues.append(q_v[ACTION_NO_OP])
                    
                    image = self.env.render_speed_state_values(qvalues)
                    image = Image.fromarray(image)
                    trajectory_imgs.append(image)

                    # Folder for same trajectory
                    trajectory_folder = f"{speed_outputs}/{j}"
                    os.makedirs(trajectory_folder, exist_ok=True)

                    image.save(f"{trajectory_folder}/{k}.png")

            trajectory_imgs[0].save(
                f"{trajectory_folder}/{j+1}th_trajectory.gif",
                save_all=True,
                append_images=trajectory_imgs[1:],
                duration=200,  # Time per frame in milliseconds
                loop=0,  # Loop forever
                optimize=True  # Optimize GIF for smaller file size
            )

    def soft_update_target(self):
        """Soft update of the target network parameters"""
        for target_param, param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def learn(self):
        # If no sufficient data available
        if self.replay_buffer.size() < self.batch_size :
            return
        
        self.optimizer.zero_grad()

        # Resampled Training Batch with Prioritized Experience Replay
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, indices = self.replay_buffer.sample(self.batch_size)
        
        states_batch = torch.tensor(states_batch).to(self.dqn.device).float()
        actions_batch = torch.tensor(actions_batch).to(self.dqn.device).to(torch.int64)
        rewards_batch = torch.tensor(rewards_batch).to(self.dqn.device).float()
        next_states_batch = torch.tensor(next_states_batch).to(self.dqn.device).float()
        dones_batch = torch.tensor(dones_batch).to(self.dqn.device).float()
        
        q_values = self.dqn(states_batch)
        q_value = q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)
        
        next_q_values = self.target_dqn(next_states_batch)
        next_q_value = next_q_values.max(1).values
        
        target = rewards_batch + self.df * next_q_value * (1 - dones_batch)
        
        loss = self.loss(q_value, target).to(self.dqn.device)
        
        loss.backward()
        self.optimizer.step()
        
        # Update Priorities
        td_errors = torch.abs(q_value - target).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

    
    def get_policy(self):
        '''
        Learns the policy
        '''
        #TODO: You can add you code here
        discounted_returns = []  # Store avg discounted return
        max_distances = []  # Store avg max distance
        iterations = []  # Store iteration count

        for episode in range(self.iterations):
            current_state = self.env.reset()
            done = False

            step = 0

            # Rollout phase
            while not done:
                step += 1
                action = self.choose_action(current_state, greedy=True)

                # Perform action
                next_state, reward, done, _ = self.env.step(action)

                # Adding experience
                self.replay_buffer.add_experience(state=current_state,
                                                  action=action,
                                                  reward=reward,
                                                  next_state=next_state,
                                                  done=done)
                
                current_state = next_state
                
            # Learning phase at the end of each episode
            self.learn()
            
            # Hard Update
            if (step + 1) % self.update_network_every == 0:
                self.target_dqn.load_state_dict(self.dqn.state_dict())

            self.eps = max(
                self.min_eps, self.eps_decay * self.eps
            )
            
            if (episode + 1) % self.validate_every == 0:
                avg_return, avg_distance = self.validate_policy()
                discounted_returns.append(avg_return)
                max_distances.append(avg_distance)
                iterations.append(episode + 1)
                print(f"Episode {episode+1}: Avg Return = {avg_return}, Avg Distance = {avg_distance}")
        
        # Visualization of policy at the end training
        self.visualize_policy(self.iterations)

        # Maximum Distance and Value of start state averaged over 100 trajectories
        validation_runs = self.validation_runs
        self.validation_runs = 100
        avg_return, avg_distance = self.validate_policy()
        self.validation_runs = validation_runs
        file_name = f"{self.log_folder}/Maximum_Distance_and_Value.txt"
        with open(file_name, 'w') as outfile:
            s = "Averaged over 100 trajectories\n"
            s += f"Maximum Distance :- {avg_distance}\nValues of start state :- {avg_return}"
            print(s)
            outfile.write(s)

        
        file_name = f"{self.log_folder}/Discounted_returns_avg_max_distance.png"
        self.plot_results(iterations=iterations, 
                          discounted_returns=discounted_returns,
                          max_distances=max_distances,
                          file_name=file_name)

        # Visualize Lane and Speed values with NO_OP
        self.visualize_lane_value(self.iterations)
        self.visualize_speed_value(self.iterations)

        return 

    def plot_results(self, iterations, discounted_returns, max_distances, file_name):
        plt.figure(figsize=(12, 5))

        # Plot Discounted Return
        plt.subplot(1, 2, 1)
        plt.plot(iterations, discounted_returns, marker='o', label="Discounted Return")
        plt.xlabel("Iterations")
        plt.ylabel("Avg Discounted Return")
        plt.title("Discounted Return vs. Iterations")
        plt.legend()

        # Plot Max Distance
        plt.subplot(1, 2, 2)
        plt.plot(iterations, max_distances, marker='o', color='r', label="Max Distance")
        plt.xlabel("Iterations")
        plt.ylabel("Avg Max Distance")
        plt.title("Max Distance vs. Iterations")
        plt.legend()

        plt.savefig(file_name)

        

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations (integer).")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the input file.")
    args = parser.parse_args()
    #for part a
    env = get_highway_env(dist_obs_states = 5, reward_type = 'dist')
    '''
    For part b:
        env = get_highway_env(dist_obs_states = 5, reward_type = 'dist', obs_type='continious')
    '''
    env = HighwayEnv()
    qagent = DQNAgent(env,
                      iterations = 200000,
                      validate_every=1000,
                      validation_runs=1000,
                      log_folder = args.output_folder)
    qagent.get_policy()