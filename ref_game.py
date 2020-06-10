from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import gym
import random
from get_dummies import get_dummy_data

class RFGame(MultiAgentEnv):
    def __init__(self, n_pairs, n_features, n_clusters, n_samples, n_vocab, sender_type='aware'):
        self.n_pairs = n_pairs
        self.n_agents = 2*self.n_pairs
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

    def get_speaker_input(self, i):
        if self.sender_type=='aware':
            inp = list(self.states[i])
        else:
            inp = list(self.states[i][0])
        return np.array(inp).flatten()

    def get_listener_input(self, i, token):
        shuffled_index, shuffled_states = zip(*sorted(zip(range(self.n_samples), self.states[i]), key=lambda _: random.random()))
        self.target[i] = list(shuffled_index).index(0)
        inp = np.array(shuffled_states).flatten().tolist()
        inp.append(float(token)/self.n_vocab)
        return np.array(inp).flatten()

    def reset(self):
        self.speaker_step = True
        self.mapping()
        self.nos = [np.random.choice(range(self.n_clusters), self.n_samples, replace=False).tolist() for i in range(self.n_pairs)]
        self.states = [[self.X1[self.Y1==i][np.random.choice(self.X1[self.Y1==i].shape[0], 1)][0] for i in self.nos[j]] for j in range(self.n_pairs)]
        self.target = [0 for i in range(self.n_pairs)]

        obs = {}
        self.speaker_input = {}
        self.listener_input = {}
        for i in range(self.n_agents):
            if i<self.n_pairs:
                id = str(i)
                self.speaker_input[i] = np.array(self.get_speaker_input(i))
                obs[id] = self.speaker_input[i]
        self.speaker_step = False
        return obs

    def mapping(self):
        sp = [i for i in range(self.n_pairs)]
        lt = [i for i in range(self.n_pairs, self.n_agents)]
        np.random.shuffle(lt)
        z = list(zip(sp, lt))
        self.sp2lt = {i[0]:i[1] for i in z}
        self.lt2sp = {i[1]:i[0] for i in z}

        self.sp2lt = {i[0]:i[0]+self.n_pairs for i in z}
        self.lt2sp = {i[1]:i[1]-self.n_pairs for i in z}


    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        actions = action_dict

        if self.speaker_step:
            for i in range(self.n_agents):
                id = str(i)
                if i<self.n_pairs:
                    identify = int(actions[str(self.sp2lt[i])] == self.target[i])
                    obs[id], rew[id], done[id], info[id] = self.speaker_input[i], identify, True, {'true_rew':identify}
                else:
                    identify = int(actions[str(i)] == self.target[self.lt2sp[i]])
                    obs[id], rew[id], done[id], info[id] = self.listener_input[self.lt2sp[i]], identify, True, {'true_rew':identify}
            done["__all__"] = True
        else:
            for i in range(self.n_agents):
                id = str(i)
                if i>=self.n_pairs:
                    self.heard_message = actions[str(self.lt2sp[i])]
                    self.listener_input[self.lt2sp[i]] = np.array(self.get_listener_input(self.lt2sp[i], self.heard_message))
                    obs[id], rew[id], done[id], info[id] = self.listener_input[self.lt2sp[i]], 0, False, {}
            self.speaker_step = True
            done["__all__"] = False
        return obs, rew, done, info
