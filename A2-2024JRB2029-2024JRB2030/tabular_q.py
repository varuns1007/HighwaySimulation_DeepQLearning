
from env import HighwayEnv, ACTION_NO_OP, get_highway_env
import numpy as np 
from typing import Tuple
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import argparse
import matplotlib.pyplot as plt
import os


'''
This the optional class tempelate for the Tabular Q agent. 
You are free to use your own class template or 
modify this class tempelate 
'''
class TabularQAgent:

    def __init__(self, 
                    env: HighwayEnv, 
                    alpha: float = 0.1, 
                    eps: float = 0.75, 
                    discount_factor: float = 0.9,
                    iterations: int = 100000, 
                    eps_type: str = 'constant',
                    validation_runs: int = 1000,
                    validate_every: int = 1000,
                    visualize_runs: int = 10, 
                    visualize_every: int = 5000,
                    log_folder:str = './',
                    eps_decay_type:str = 'exponential',
                    eps_decay: float = 0.99995,
                    eps_min:float = 0.01
                    ):

        #TODO: You can add you code here
        self.df = discount_factor
        self.alpha = alpha
        self.env = env
        self.iterations = iterations
        self.validation_runs = validation_runs
        self.validate_every = validate_every
        self.visualization_runs = visualize_runs
        self.visualization_every = visualize_every
        self.log_folder = log_folder
        
        # Creating results directory
        os.makedirs(self.log_folder, exist_ok=True)

        self.Q_Mat = defaultdict(float)
        self.eps = eps
        self.eps_initial = self.eps
        self.eps_type = eps_type
        if self.eps_type != 'constant':
            self.eps_decay_type = eps_decay_type
            self.eps_decay = eps_decay
            self.eps_min = eps_min
   
    

    def argmax(self, a):
        # random argmax
        a = np.array(a)
        return np.random.choice(np.arange(len(a), dtype=int)[a == np.max(a)])

    def choose_action(self, state, greedy = False, step=1):

        '''
        Right now returning random action but need to add
        your own logic
        '''
        #TODO: You can add you code 

        action = None

        if greedy:
            q_values = [self.Q_Mat.get((state, act), 0) for act in range(5)]
            action = self.argmax(q_values)

        else:
            if np.random.rand() < self.eps:
                action = np.random.randint(0, 5)
            else:
                q_values = [self.Q_Mat.get((state, act), 0) for act in range(5)]
                action = self.argmax(q_values)

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
            obs = tuple(obs)

            # Starting of highway environment, Control car has position set to 2
            initial_car_pos = self.env.control_car.pos

            total_reward = 0
            discount_factor = 1
            distance = 0

            while not done:
                action = self.choose_action(obs, greedy=True)  # Use greedy action
                next_obs, reward, done, _ = self.env.step(action)
                next_obs = tuple(next_obs)

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
            obs = tuple(obs)

            while not done:
                action = self.choose_action(obs, greedy=True)
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
                        st = tuple(st)
                        q_v = self.Q_Mat.get((st, ACTION_NO_OP), 0)
                        qvalues.append(q_v)
                    
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
                        st = tuple(st)
                        q_v = self.Q_Mat.get((st, ACTION_NO_OP), 0)
                        qvalues.append(q_v)
                    
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

    def get_policy(self):
        '''
        Learns the policy
        '''
        #TODO: You can add you code here

        discounted_returns = []  # Store avg discounted return
        max_distances = []  # Store avg max distance
        iterations = []  # Store iteration count
        eps_list = []
        avg_5_max_distances = []

        for episode in range(self.iterations):
            current_state = self.env.reset()
            current_state = tuple(current_state)
            done = False
            # self.eps = self.eps_initial #reseting eps for each new episode
            eps_list.append(self.eps)

            while not done:
                action = self.choose_action(current_state,step=episode)

                # Perform action
                next_state, reward, done, _ = self.env.step(action)
                next_state = tuple(next_state)

                next_q_value = max([self.Q_Mat.get((next_state, next_action), 0) for next_action in range(5)])

                self.Q_Mat[(current_state), action] = self.Q_Mat.get((current_state, action), 0) + \
                                    self.alpha * (reward + self.df * next_q_value - self.Q_Mat.get((current_state, action), 0))

                current_state = next_state

            if self.eps_type != 'constant':
                if self.eps_decay_type != 'exponential':
                    self.eps = max(self.eps_min,self.eps_initial - (episode/self.iterations)*(self.eps_initial - self.eps_min))
                else:
                    self.eps = max(self.eps_min,self.eps_decay*self.eps)

            
            if (episode+1) % self.validate_every == 0:
                avg_return, avg_distance = self.validate_policy()
                discounted_returns.append(avg_return)
                max_distances.append(avg_distance)
                iterations.append(episode + 1)
                avg_5_max_distances.append(sum(max_distances[-5:])/5)
                print(f"Episode {episode+1}: Avg Return = {avg_return}, Avg Distance = {avg_distance}")

            # if (i_iteration+1) % self.visualization_every == 0:
            #     self.visualize_policy(i_iteration + 1)

        
        
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

        # Visualize Lane and Speed values with NO_OP
        self.visualize_lane_value(self.iterations)
        self.visualize_speed_value(self.iterations)

        file_name = f"{self.log_folder}/Discounted_returns_avg_max_distance.png"
        self.plot_results(iterations=iterations, 
                          discounted_returns=discounted_returns,
                          max_distances=max_distances,
                          file_name=file_name)

        if self.eps_type != 'constant':
            file_name_eps = f"{self.log_folder}/epsilon_trajectory.png"
            self.plot_results_2(iterations=list(range(self.iterations)),data=eps_list,label="Epsilon Trajectory",file_name=file_name_eps)

        file_name_avg_return = f"{self.log_folder}/{self.eps_type}_{self.eps_decay_type if self.eps_type != 'constant' else ''}_Last5_average_return.png"
        self.plot_results_2(iterations=iterations,data=avg_5_max_distances,label="Average Distance",file_name=file_name_avg_return)




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

    def plot_results_2(self, iterations, data, label, file_name):
        plt.figure(figsize=(6, 5))
        plt.plot(iterations, data, marker='o', label=label)
        plt.xlabel("Iterations")
        plt.ylabel(label)
        plt.title(f"{label} vs. Iterations")
        plt.legend()

        plt.savefig(file_name)    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations (integer).")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the input file.")
    args = parser.parse_args()
    
    #For part a, b, c and d:
    env = get_highway_env(dist_obs_states = 5, reward_type = 'dist')

    # For part e and sub part a:
    # env = get_highway_env(dist_obs_states = 5, reward_type = 'overtakes')

    # For part e and sub part b:
    # env = get_highway_env(dist_obs_states = 3, reward_type = 'dist')

    env = HighwayEnv()
    qagent = TabularQAgent(env, iterations=args.iterations,
                           log_folder = args.output_folder,eps_type='variable',eps_decay_type='exponential',eps_decay=(1-1e-4))
    qagent.get_policy()