import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import math
from collections import deque
from gym.wrappers.record_video import RecordVideo
import matplotlib.pyplot as plt

class value_network(nn.Module):
    '''
    takes in state as input and gives value at this state as output
    '''
    def __init__(self,state_dim):
        '''
        state_dim (int): state dimension
        '''
        super(value_network, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

    def forward(self,state):
        v = F.tanh(self.l1(state))
        v = F.tanh(self.l2(v))
        return self.l3(v)
        


class policy_network(nn.Module):
    '''
    Policy Network: Designed for continous action space, where given a 
    state, the network outputs the mean and standard deviation of the action
    '''
    def __init__(self,state_dim,action_dim,log_std = 0.0):
        """
        state_dim (int): The dimensionality of the state vector that the network will receive as input.
        action_dim (int): The dimensionality of the action space
        log_std (float): The logarithm of the standard deviation for action distribution. Initializing log_std allows the network to 
            learn or adjust the spread of the action distribution during training.
        """

        super(policy_network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.l1 = nn.Linear(state_dim,64)
        self.l2 = nn.Linear(64,64)

        self.mean = nn.Linear(64,action_dim)
        """
        A parameter that holds the logarithm of the standard deviations of the action distributions. 
        It is initialized and can be learned during training.
        """
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    
    def forward(self,state):
        '''
        Input: State
        Output: Mean, log_std and std of the distribution of actions
        '''
        a = F.tanh(self.l1(state))
        a = F.tanh(self.l2(a))
        a_mean = self.mean(a)

        #try:
        a_log_std = self.log_std.expand_as(a_mean)  
        #except: 
        #    a_log_std = self.log_std.expand_as(a_mean.reshape([-1,2]))  
        a_std = torch.exp(a_log_std)
        return a_mean, a_log_std, a_std

    def select_action(self, state):
        '''
        Input: State
        Output: Sample drawn from a normal disribution with mean and std
        '''
        a_mean, _, a_std = self.forward(state)
        action = torch.normal(a_mean, a_std)
        return action
    
    def get_log_prob(self, state, action):
        '''
        Input: State, Action
        Output: log probabilities
        '''
        mean, log_std, std = self.forward(state)
        var = std.pow(2)
        log_density = -(action - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std # log_density of all actions
        return log_density.sum(1, keepdim=True) # sum of log likelihood of all actions


class PGAgent():
    '''
    An agent that performs different variants of the PG algorithm
    '''
    def __init__(self,
     state_dim, 
     action_dim,
     discount=0.99,
     lr=1e-3,
     gpu_index=0,
     seed=0,
     env="LunarLander-v2"
     ):
        """
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            discount (float): discount factor
            lr (float): learning rate
            gpu_index (int): GPU used for training
            seed (int): Seed of simulation
            env (str): Name of environment
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.lr = lr
        self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        self.env_name = env
        self.seed = seed
        self.policy = policy_network(state_dim,action_dim) # assign the policy to the agent
        self.value = value_network(state_dim) # assign the value function to the agent
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_value = torch.optim.Adam(self.value.parameters(), lr=self.lr)

    def sample_traj(self,batch_size=2000,evaluate = False):
        '''
        Samples trajectories from the environment by interacting with it according to the policy defined by the agent. 
        This method can operate in either training mode or evaluation mode.

        Input: 
            batch_size: minimum batch size needed for update. In fact it's the length of the whole returned 
            trajectory
        Output:
            states, actions, rewards,not_dones, episodic reward     
        '''
        self.policy.to("cpu") # Move network to CPU for sampling
        env = gym.make(args.env,continuous=True)
        states = []
        actions = []
        rewards = []
        n_dones = [] # not_done
        curr_reward_list = []
        while len(states) < batch_size:
            state, _ = env.reset(seed=self.seed)
            curr_reward = 0
            for t in range(1000):
                state_ten = torch.from_numpy(state).float().unsqueeze(0)
                with torch.no_grad():
                    if evaluate:
                        action = self.policy(state_ten)[0][0].numpy() # Take mean action during evaluation
                    else:
                        action = self.policy.select_action(state_ten)[0].numpy() # Sample from distribution during training
                action = action.astype(np.float64)
                n_state,reward,terminated,truncated,_ = env.step(action) # Execute action in the environment
                done = terminated or truncated
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                n_done = 0 if done else 1
                n_dones.append(n_done)
                state = n_state
                curr_reward += reward
                if done:
                    break
            curr_reward_list.append(curr_reward)
        if evaluate:
            return np.mean(curr_reward_list)
        return states,actions,rewards,n_dones, np.mean(curr_reward_list)
    



    def update(self,states,actions,rewards,n_dones,update_type='Baseline'):
        '''
        Inputs:
            states: list of states
            actions: list of actions
            rewards: list of rewards
            n_dones: list of not dmones
            update_type: type of PG algorithm
        Output: 
            None
        '''
        self.policy.to(self.device) #Move policy to GPU
        if update_type == "Baseline" or update_type == "PPO":
            self.value.to(self.device)  #Move value to GPU
        states_ten = torch.from_numpy(np.stack(states)).to(self.device)   #Convert to tensor and move to GPU
        action_ten = torch.from_numpy(np.stack(actions)).to(self.device)  #Convert to tensor and move to GPU
        rewards_ten = torch.from_numpy(np.stack(rewards)).to(self.device) #Convert to tensor and move to GPU
        n_dones_ten = torch.from_numpy(np.stack(n_dones)).to(self.device) #Convert to tensor and move to GPU

        if update_type == "Rt": ## reinforcement algorithm
            '''
            Peform PG using the cumulative discounted reward of the entire trajectory
            1. Compute the discounted reward of each trajectory (rt)
            2. Compute log probabilities using states_ten and action_ten
            3. Compute policy loss and update the policy
            '''
            # since the input states, actions are actually different trajectories concatenated together,
            # we have to first recover different trajectories by breaking up the big one
            counter = 0 # counts how many actual trajectories make up the concatenated large one
            flag = [0]
            total_r_each = []
            s_pieces  = []
            a_pieces = []
            r_pieces = []
            with torch.no_grad():
                for i in range(len(states)):
                    if n_dones_ten[i] == 0:
                        s_pieces.append(states_ten[flag[-1]:i])
                        a_pieces.append(action_ten[flag[-1]:i])
                        r_pieces.append(rewards_ten[flag[-1]:i])
                        flag.append(i)
                        counter+=1

                    if i == len(states)-1:
                        s_pieces.append(states_ten[flag[-1]:])
                        a_pieces.append(action_ten[flag[-1]:])
                        r_pieces.append(rewards_ten[flag[-1]:])
                        counter+=1
                #with torch.no_grad():
                for i in range(len(s_pieces)):
                    multipliers = (torch.ones(len(s_pieces[i]))*self.discount).pow(torch.arange(len(s_pieces[i])))
                    multipliers[-1] = 0
                    multipliers=multipliers.to(self.device)
                    # 1. Compute the discounted reward of each trajectory (rt)
                    total_r_each.append((r_pieces[i]*multipliers).sum())
                total_r_each = torch.tensor(total_r_each).to(self.device)            
            total_loss = torch.zeros((len(s_pieces), 1), requires_grad=True).to(self.device)
            for i in range(len(s_pieces)):
                # 2. Compute log probabilities using states_ten and action_ten
                sum_log_prob = -self.policy.get_log_prob(s_pieces[i], a_pieces[i]).sum().to(self.device)
                # 3. Compute policy loss and update the policy
                total_loss[i] = total_r_each[i]*sum_log_prob
            total_loss = total_loss.mean()
            # 3. Compute policy loss and update the policy
            self.optimizer_policy.zero_grad()
            total_loss.backward()
            self.optimizer_policy.step()

        if update_type == 'Gt':
            '''
            Perform PG using reward_to_go
            1. Compute reward_to_go (gt) using rewards_ten and n_dones_ten
            2. gt should be of the same length as rewards_ten
            3. Compute log probabilities using states_ten and action_ten
            4. Compute policy loss and update the policy
            '''
            counter = 0
            flag = [0]
            s_pieces  = []
            a_pieces = []
            r_pieces = []
            with torch.no_grad():
                for i in range(len(states)):
                    if n_dones_ten[i] == 0:
                        s_pieces.append(states_ten[flag[-1]:i])
                        a_pieces.append(action_ten[flag[-1]:i])
                        r_pieces.append(rewards_ten[flag[-1]:i])
                        flag.append(i)
                        counter+=1

                    if i == len(states)-1:
                        s_pieces.append(states_ten[flag[-1]:])
                        a_pieces.append(action_ten[flag[-1]:])
                        r_pieces.append(rewards_ten[flag[-1]:])
                        counter+=1
            
            total_loss = torch.zeros(len(s_pieces), requires_grad=True).to(self.device)
            for i in range(len(s_pieces)): # calculate the sum inside braces for each i=1, ..., N
                # 2. gt should be of the same length as the length of the corresponding trajectory
                rewards_ahead = torch.zeros((len(s_pieces[i])), requires_grad=True).to(self.device)
                with torch.no_grad():
                    for j in range(len(s_pieces[i])):
                        tmp = r_pieces[i][j:] # all the running rewards starting from the current state
                        multipliers = (torch.ones(len(tmp))*self.discount).pow(torch.arange(len(tmp))).to(self.device) # discount factors
                        multipliers[-1] = 0 # This should not be needed since the reward of the last state of a trajectory should already be 0. We are just being cautious
                        # 1. Compute reward_to_go (gt) using rewards_ten and n_dones_ten. n_dones_ten not needed here
                        rewards_ahead[j] = (multipliers*tmp).sum()
                # 3. Compute log probabilities using states_ten and action_ten
                log_probs = -self.policy.get_log_prob(s_pieces[i], a_pieces[i])
                # 4. Compute policy loss and update the polic
                total_loss[i] = (log_probs*rewards_ahead).sum()
            total_loss = total_loss.mean()
            # 4. Compute policy loss and update the policy
            self.optimizer_policy.zero_grad()
            total_loss.backward()
            self.optimizer_policy.step()

        if update_type == 'Baseline':
            '''
            Peform PG using reward_to_go and baseline
            1. Compute values of states, this will be used as the baseline 
            2. Compute reward_to_go (gt) using rewards_ten and n_dones_ten
            3. gt should be of the same length as rewards_ten
            4. Compute advantages 
            5. Update the value network to predict gt for each state (L2 norm)
            6. Compute log probabilities using states_ten and action_ten
            7. Compute policy loss (using advantages) and update the policy
            '''

            counter = 0
            flag = [0]
            s_pieces  = []
            a_pieces = []
            r_pieces = []
            with torch.no_grad():
                for i in range(len(states)):
                    if n_dones_ten[i] == 0:
                        s_pieces.append(states_ten[flag[-1]:i])
                        a_pieces.append(action_ten[flag[-1]:i])
                        r_pieces.append(rewards_ten[flag[-1]:i])
                        flag.append(i)
                        counter+=1

                    if i == len(states)-1:
                        s_pieces.append(states_ten[flag[-1]:])
                        a_pieces.append(action_ten[flag[-1]:])
                        r_pieces.append(rewards_ten[flag[-1]:])
                        counter+=1
            
            total_loss = torch.zeros(len(s_pieces), requires_grad=True).to(self.device)
            total_loss_v_prediction = torch.zeros(len(s_pieces), requires_grad=True).to(self.device)
            for i in range(len(s_pieces)): # calculate the sum inside braces for each i=1, ..., N
                # 3. gt should be of the same length as the length of the corresponding trajectory
                rewards_ahead = torch.zeros((len(s_pieces[i])), requires_grad=False).to(self.device) # to store Gt for t = 0, ..., len(s_pieces[i])-1
                v_prediction = torch.zeros((len(s_pieces[i])), requires_grad=True).to(self.device) # to store V(st) for t = 0, ..., len(s_pieces[i])-1
                total_loss_v_prediction_tmp = torch.zeros(len(s_pieces[i]), requires_grad=True).to(self.device) 
                for j in range(len(s_pieces[i])):
                    with torch.no_grad():
                        tmp = r_pieces[i][j:] # all the running rewards starting from the current state
                        multipliers = (torch.ones(len(tmp))*self.discount).pow(torch.arange(len(tmp))).to(self.device) # discount factors
                        multipliers[-1] = 0 # This should not be needed since the reward of the last state of a trajectory should already be 0. We are just being cautious
                        # 2. Compute reward_to_go (gt) using rewards_ten and n_dones_ten. n_dones_ten not needed here
                        rewards_ahead[j] = (multipliers*tmp).sum()
                    # 1. Compute values of states, this will be used as the baseline 
                    v_prediction[j] = self.value(s_pieces[i][j])
                    #rewards_ahead=(rewards_ahead-rewards_ahead.mean())/(rewards_ahead.std())
                with torch.no_grad():
                    # 4. Compute advantages 
                    advantages = rewards_ahead-v_prediction
                    #advantages=(advantages-advantages.mean())/(advantages.std())
                total_loss_v_prediction[i] = (advantages**2).mean()
                # 6. Compute log probabilities using states_ten and action_ten
                log_probs = -self.policy.get_log_prob(s_pieces[i], a_pieces[i])
                # 7. Compute policy loss (using advantages) and update the policy
                total_loss[i] = (log_probs*advantages).sum()
            total_loss_v_prediction = total_loss_v_prediction.mean()
            total_loss = total_loss.mean()
            # 5. Update the value network to predict gt for each state (L2 norm)
            self.optimizer_value.zero_grad()
            total_loss_v_prediction.backward()
            self.optimizer_value.step()
            # 7. Compute policy loss (using advantages) and update the policy
            self.optimizer_policy.zero_grad()
            total_loss.backward()
            self.optimizer_policy.step()

        if update_type == 'PPO':
            counter = 0
            flag = [0]
            s_pieces  = []
            a_pieces = []
            r_pieces = []
            with torch.no_grad():
                for i in range(len(states)):
                    if n_dones_ten[i] == 0:
                        s_pieces.append(states_ten[flag[-1]:i])
                        a_pieces.append(action_ten[flag[-1]:i])
                        r_pieces.append(rewards_ten[flag[-1]:i])
                        flag.append(i)
                        counter+=1

                    if i == len(states)-1:
                        s_pieces.append(states_ten[flag[-1]:])
                        a_pieces.append(action_ten[flag[-1]:])
                        r_pieces.append(rewards_ten[flag[-1]:])
                        counter+=1
            
            total_loss = torch.zeros(len(s_pieces), requires_grad=True).to(self.device)
            total_loss_v_prediction = torch.zeros(len(s_pieces), requires_grad=True).to(self.device)

            for i in range(len(s_pieces)): # calculate the sum inside braces for each i=1, ..., N
                # 3. gt should be of the same length as the length of the corresponding trajectory
                rewards_ahead = torch.zeros((len(s_pieces[i])), requires_grad=False).to(self.device) # to store Gt for t = 0, ..., len(s_pieces[i])-1
                v_prediction = torch.zeros((len(s_pieces[i])), requires_grad=True).to(self.device) # to store V(st) for t = 0, ..., len(s_pieces[i])-1
                total_loss_v_prediction_tmp = torch.zeros(len(s_pieces[i]), requires_grad=True).to(self.device) 
                for j in range(len(s_pieces[i])):
                    with torch.no_grad():
                        tmp = r_pieces[i][j:] # all the running rewards starting from the current state
                        multipliers = (torch.ones(len(tmp))*self.discount).pow(torch.arange(len(tmp))).to(self.device) # discount factors
                        multipliers[-1] = 0 # This should not be needed since the reward of the last state of a trajectory should already be 0. We are just being cautious
                        # 2. Compute reward_to_go (gt) using rewards_ten and n_dones_ten. n_dones_ten not needed here
                        rewards_ahead[j] = (multipliers*tmp).sum()
                    # 1. Compute values of states, this will be used as the baseline 
                    v_prediction[j] = self.value(s_pieces[i][j])
                    #rewards_ahead=(rewards_ahead-rewards_ahead.mean())/(rewards_ahead.std()+1e-8)
                with torch.no_grad():
                    # 4. Compute advantages 
                    advantages = rewards_ahead-v_prediction
                    #advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
                total_loss_v_prediction[i] = (advantages**2).mean()
                # 6. Compute log probabilities using states_ten and action_ten
                log_probs = self.policy.get_log_prob(s_pieces[i], a_pieces[i])
                with torch.no_grad():
                    log_probs_old = self.policy.get_log_prob(s_pieces[i], a_pieces[i])
                ratio = torch.exp(log_probs-log_probs_old)
                min_ = torch.minimum(ratio*advantages, torch.clamp(ratio, min=1-0.2, max=1+0.2)*advantages )
                # 7. Compute policy loss (using advantages) and update the policy
                total_loss[i] = -min_.mean()
            total_loss_v_prediction = total_loss_v_prediction.mean()
            total_loss = total_loss.mean()
            # 5. Update the value network to predict gt for each state (L2 norm)
            self.optimizer_value.zero_grad()
            total_loss_v_prediction.backward()
            self.optimizer_value.step()
            # 7. Compute policy loss (using advantages) and update the policy
            self.optimizer_policy.zero_grad()
            total_loss.backward()
            self.optimizer_policy.step()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v2")           # Gymnasium environment name
    parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--n-iter", default=300, type=int)           # Maximum number of training iterations
    parser.add_argument("--discount", default=0.99)                  # Discount factor
    parser.add_argument("--batch-size", default=5000, type=int)      # Training samples in each batch of training
    parser.add_argument("--lr", default=5e-3,type=float)             # Learning rate
    parser.add_argument("--gpu-index", default=0,type=int)           # GPU index
    parser.add_argument("--algo", default="Baseline",type=str)       # PG algorithm type. Baseline/Gt/Rt
    args = parser.parse_args()

    # Making the environment    
    env = gym.make(args.env,continuous=True)#, render_mode="rgb_array")
    #env = gym.wrappers.RecordVideo(env, video_folder="/ECEN743vid", name_prefix="test-video", episode_trigger=lambda x: True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    kwargs = {
        "state_dim":state_dim,
        "action_dim":action_dim,
        "discount":args.discount,
        "lr":args.lr,
        "gpu_index":args.gpu_index,
        "seed":args.seed,
        "env":args.env
    }   
    learner = PGAgent(**kwargs) # Creating the PG learning agent

    moving_window = deque(maxlen=10)
    all_10_averaged_rewards = []

    for e in range(args.n_iter):
        '''
        Steps of PG algorithm
            1. Sample environment to gather data using a policy
            2. Update the policy using the data
            3. Evaluate the updated policy
            4. Repeat 1-3
        '''
        states,actions,rewards,n_dones,train_reward = learner.sample_traj(batch_size=args.batch_size)
        learner.update(states,actions,rewards,n_dones,args.algo)
        eval_reward= learner.sample_traj(evaluate=True)
        moving_window.append(eval_reward)
        if e % 10 == 0:
            all_10_averaged_rewards.append(np.mean(moving_window))
            print('Training Iteration {} Training Reward: {:.2f} Evaluation Reward: {:.2f} \
            Average Evaluation Reward: {:.2f}'.format(e,train_reward,eval_reward,np.mean(moving_window)))
    
    """
    TODO: Write code for
    1. Logging and plotting
    2. Rendering the trained agent 
    """
    plt.figure(figsize=(10, 5))
    plt.plot(all_10_averaged_rewards)
    plt.title('Evaluation Rewards Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig('training_rewards_plot.png')
    plt.show()
    # 2. Uncomment only when you need to store the final videos 
    """
    learner.policy.eval = True
    # env reset for a fresh start
    state, _ = env.reset()

    # Start the recorder
    env.start_video_recorder()

    for _ in range(1000):
        # action = learner.policy.select_action(torch.from_numpy(state))[0].numpy()
        # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
        action = learner.policy.select_action(torch.from_numpy(state))[0].detach().numpy()
        state, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            state, info = env.reset()

    # close the video recorder before the env!
    env.close_video_recorder()

    # Close the environment
    env.close()
    #################################
    """
