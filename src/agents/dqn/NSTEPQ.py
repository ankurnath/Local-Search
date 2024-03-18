"""
Implements a n-step Q learning agent.
"""

import os
import pickle
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.agents.dqn.utils import ReplayBuffer, Logger, TestMetric, set_global_seed
from src.envs.utils import ExtraAction

class NSTEPQ :
    """
    # Required parameters.
    env:  environment to use.
    network : Choice of neural network.

    
    
    # Q parameters
    gamma : Discount factor
    n_step : Number of steps
    

    # Replay buffer.
    replay_start_size : The capacity of the replay buffer at which training can begin.
    replay_buffer_size : Maximum buffer capacity.
    minibatch_size : Minibatch size.
    

    # Learning rate
    learning_rate : Learning Rate.
    


    # Exploration
    update_exploration : Whether to update the exploration rate.
    

    # Loss function
    adam_epsilon : epsilon for ADAM optimisation.
    loss="mse" : Loss function to use.

    # Saving the agent
    save_network_frequency : Frequency with which the network parameters are saved.
    network_save_path : Folder into which the network parameters are saved.

    # Testing the agent
    evaluate : Whether to test the agent during training.
    test_envs : List of test environments.  None means the training environments (envs) are used.
    test_frequency : Frequency of tests.
    test_save_path : Folder into which the test scores are saved.
    test_metric : The metric used to quantify performance.

    # Other
    
    seed : The global seed to set.  None means randomly selected.
    """
    def __init__(
        self,
        envs,
        network,



        # Q learning parameters
        gamma=0.8,

        # Replay buffer.
        replay_start_size=20,
        replay_buffer_size=50,
        minibatch_size=8,
        update_frequency=1,
        n_step=2,
        

        # Learning rate
        learning_rate=0.0005,

       

        # Exploration
        update_exploration=True,
        final_exploration_rate=0.05,
        
        
        # Loss function
        adam_epsilon=1e-8,
        loss="mse",

        # Saving the agent
        save_network_frequency=10000,
        network_save_path='network',

        # Testing the agent
        evaluate=True,
        test_env=None,
        test_frequency=10000,
        test_save_path='test_scores',

        # Other
        logging=True,
        seed=None
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        
        
        self.minibatch_size = minibatch_size

        self.learning_rate=learning_rate
        self.n_step=n_step
        

        
        self.update_frequency = update_frequency # ????
        self.update_exploration = update_exploration
        self.final_exploration_rate=final_exploration_rate

        self.adam_epsilon = adam_epsilon
        self.logging = logging
        if callable(loss):
            self.loss = loss
        else:
            try:
                self.loss = {'huber': F.smooth_l1_loss, 'mse': F.mse_loss}[loss]
            except KeyError:
                raise ValueError("loss must be 'huber', 'mse' or a callable")

        if type(envs)!=list:
            envs = [envs]
        self.env = env
        

        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size, 
                                          n_step=self.n_step, 
                                          gamma=self.gamma)
        

        self.seed = random.randint(0, 1e6) if seed is None else seed

        for env in self.envs:
            set_global_seed(self.seed, env)

        self.network = network().to(self.device)
        

        self.optimizer = optim.Adam(self.network.parameters(), 
                                    lr=self.learning_rate, 
                                    eps=self.adam_epsilon,
                                    )

        self.evaluate = evaluate
        self.test_env = test_env
        self.test_frequency = test_frequency
        self.test_save_path = test_save_path
        

        self.losses_save_path = os.path.join(os.path.split(self.test_save_path)[0], "losses.pkl")

        self.save_network_frequency = save_network_frequency
        self.network_save_path = network_save_path


    

    

    def learn(self, timesteps, verbose=False):



        # Initialise the state
        # state = torch.as_tensor(self.env.reset())
        state = self.env.reset()
        score = 0
        losses_eps = []
        t1 = time.time()

        test_scores = []
        losses = []

        is_training_ready = False

        for timestep in range(timesteps):

            if not is_training_ready:
                # if all([len(rb)>=self.replay_start_size for rb in self.replay_buffer.values()]):
                if len(self.replay_buffer)>=self.replay_start_size:
                    print('\nThe buffer has {} transitions stored - training is starting!\n'.format(
                        self.replay_start_size))
                    is_training_ready=True

            # Choose action
            action = self.act(state.to(self.device).float(), is_training_ready=is_training_ready)

            # Update epsilon
            if self.update_exploration:
                self.update_epsilon(timestep)

            

            # Perform action in environment
            state_next, reward, done, _ = self.env.step(action)

            score += reward

            # Store transition in replay buffer
            action = torch.as_tensor([action], dtype=torch.long)
            reward = torch.as_tensor([reward], dtype=torch.float)
            state_next = torch.as_tensor(state_next)

            done = torch.as_tensor([done], dtype=torch.float)

            self.replay_buffer.add(state, action, reward, state_next, done)

            if done:
                # Reinitialise the state
                if verbose:
                    loss_str = "{:.2e}".format(np.mean(losses_eps)) if is_training_ready else "N/A"
                    print("timestep : {}, episode time: {}, score : {}, mean loss: {}, time : {} s".format(
                        (timestep+1),
                         self.env.current_step,
                         np.round(score,3),
                         loss_str,
                         round(time.time() - t1, 3)))

                
                
                state=self.env.reset()
                score = 0
                losses_eps = []
                t1 = time.time()

            else:
                state = state_next

            if is_training_ready:

                # Update the main network
                if timestep % self.update_frequency == 0:

                    # Sample a batch of transitions
                    transitions = self.get_random_replay_buffer().sample(self.minibatch_size, self.device)

                    # Train on selected batch
                    loss = self.train_step(transitions)
                    losses.append([timestep,loss])
                    losses_eps.append(loss)


            if (timestep+1) % self.test_frequency == 0 and self.evaluate and is_training_ready:
                test_score = self.evaluate_agent()
                print('\nTest score: {}\n'.format(np.round(test_score,3)))


                best_network = all([test_score > score for t,score in test_scores])

                if best_network:
                    path = self.network_save_path
                    path_main, path_ext = os.path.splitext(path)
                    path_main += "_best"
                    if path_ext == '':
                        path_ext += '.pth'
                    self.save(path_main + path_ext)

                test_scores.append([timestep+1,test_score])

            if (timestep + 1) % self.save_network_frequency == 0 and is_training_ready:
                path = self.network_save_path
                path_main, path_ext = os.path.splitext(path)
                path_main += str(timestep+1)
                if path_ext == '':
                    path_ext += '.pth'
                self.save(path_main+path_ext)

        

        path = self.test_save_path
        if os.path.splitext(path)[-1] == '':
            path += '.pkl'

        with open(path, 'wb+') as output:
            pickle.dump(np.array(test_scores), output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print('test_scores saved to {}'.format(path))

        with open(self.losses_save_path, 'wb+') as output:
            pickle.dump(np.array(losses), output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print('losses saved to {}'.format(self.losses_save_path))


   

    def train_step(self, transitions):

        states, actions, rewards, states_next, dones = transitions

       
        target_preds = self.network(states_next)
        with torch.no_grad():
            q_value_target = target_preds.max(1, True)[0]

        

        # Calculate TD target
        td_target = rewards + (1 - dones) * self.gamma**self.n_step * q_value_target

        # Calculate Q value
        q_value = self.network(states.float()).gather(1, actions)

        # Calculate loss
        loss = self.loss(q_value, td_target, reduction='mean')

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()


        self.optimizer.step()

        return loss.item()

    def act(self, state, is_training_ready=True):
        if is_training_ready and random.uniform(0, 1) >= self.epsilon:
            # Action that maximises Q function
            action = self.predict(state)
        else:
            # Flip random node from that hasn't yet been flipped.
            x = (state[2:, 0] != -10000).nonzero() # double check this part
            action = x[np.random.randint(0, len(x))].item()
        return action

    def update_epsilon(self, timestep):
        eps=0.9**(timestep)
        self.epsilon = max(eps, self.final_exploration_rate)

    


    @torch.no_grad()
    def predict(self, states):
        qs = self.network(states)
        if qs.dim() == 1:
            actions = qs.argmax().item()
        else:
            actions = qs.argmax(1, True).squeeze(1).cpu().numpy()
        return actions
    

    @torch.no_grad()
    def evaluate_agent(self, batch_size=None):
        """
        Evaluates agent's current performance.  Run multiple evaluations at once
        so the network predictions can be done in batches.
        """

        obs = self.test_env.reset()

        while True:
            action = self.predict(obs)
            obs, _, done, _ = self.test_env.step(action)

            if done:
                return self.test_env.objective_value


        

    def save(self, path='network.pth'):
        if os.path.splitext(path)[-1]=='':
            path + '.pth'
        torch.save(self.network.state_dict(), path)

    def load(self,path):
        self.network.load_state_dict(torch.load(path,map_location=self.device))