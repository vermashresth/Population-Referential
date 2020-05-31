from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import gym
import random
from get_dummies import get_dummy_data

class RFGame(MultiAgentEnv):
    def __init__(self, n_agents, n_features, n_clusters, n_samples, n_vocab, sender_type='aware'):
        self.n_agents = n_agents
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.n_vocab = n_vocab
        self.sender_type = sender_type
        self.dones = set()

        if self.sender_type=='aware':
            self.s_obs_space = gym.spaces.Box(low=-0.01, high=1.01, shape=(n_features*n_samples,))
        else:
            self.s_obs_space = gym.spaces.Box(low=-0.01, high=1.01, shape=(n_features,))
        self.s_act_space = gym.spaces.Discrete(n_vocab)

        self.l_obs_space = gym.spaces.Box(low=-0.01, high=1.01, shape=(n_features*n_samples+1,))
        self.l_act_space = gym.spaces.Discrete(n_samples)

        self.X1, self.Y1 = get_dummy_data(self.n_features, self.n_clusters)

    def get_speaker_input(self):
        if self.sender_type=='aware':
            inp = list(self.states)
        else:
            inp = list(self.states[0])
        return np.array(inp).flatten()

    def get_listener_input(self, token):
        shuffled_index, shuffled_states = zip(*sorted(zip(range(self.n_samples), self.states), key=lambda _: random.random()))
        self.target = list(shuffled_index).index(0)
        inp = np.array(shuffled_states).flatten().tolist()
        inp.append(float(token)/self.n_vocab)
        return np.array(inp).flatten()

    def reset(self):
        self.speaker_step = True

        self.nos = np.random.choice(range(self.n_clusters), self.n_samples, replace=False).tolist()
        self.states = [self.X1[self.Y1==i][np.random.choice(self.X1[self.Y1==i].shape[0], 1)][0] for i in self.nos]
        self.target = 0

        obs = {}
        for i in range(self.n_agents):
            if i%2==0:
                id = str(i)
                self.speaker_input = np.array(self.get_speaker_input())
                obs[id] = self.speaker_input
        self.speaker_step = False
        return obs

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        actions = action_dict

        if self.speaker_step:
            for i in range(self.n_agents):
                s_id = i//2*2
                l_id = s_id + 1
                id = str(i)
                identify = int(actions[str(l_id)] == self.target)
                if i%2==0:
                    obs[id], rew[id], done[id], info[id] = self.speaker_input, identify, True, {}
                else:
                    obs[id], rew[id], done[id], info[id] = self.listener_input, identify, True, {}
            done["__all__"] = True
        else:
            for i in range(self.n_agents):
                id = str(i)
                if i%2==1:
                    self.heard_message = actions[str((i//2)*2)]
                    self.listener_input = np.array(self.get_listener_input(self.heard_message))
                    obs[id], rew[id], done[id], info[id] = self.listener_input, 0, False, {}
            self.speaker_step = True
            done["__all__"] = False
        return obs, rew, done, info
