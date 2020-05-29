from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import gym

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

    def get_speaker_input(self):
        if self.sender_type=='aware':
            inp = list(self.states)
        else:
            inp = list(self.states[0])
        return inp

    def get_listener_input(self, token):
        shuffled_index, shuffled_states = zip(*sorted(zip(range(self.n_samples), self.states), key=lambda _: random.random()))
        self.target = list(shuffled_index).index(0)
        inp = np.array(shuffled_states).flatten().tolist()
        inp.append(token)
        return inp

    def reset(self):
        self.speaker_step = True
        self.X1, self.Y1 = get_dummy_data(self.n_features, self.n_clusters)
        self.nos = np.random.choice(range(self.n_clusters), self.n_samples, replace=False).tolist()
        self.states = [self.X1[self.Y1==i][np.random.choice(self.X1[self.Y1==i].shape[0], 1)][0] for i in self.nos]
        self.target = 0

        obs = {}
        for i in range(self.n_agents):
            if i%2==0:
                self.speaker_input = np.array(self.get_speaker_input())
                obs[i] = self.speaker_input
        self.speaker_step = False
        return obs

    # def get_observation_space(self, t=0):
    #     if t==0:
    #         return self.observation_space
    #     else:
    #         return self.observation_space_comm
    #
    # def get_action_space(self, t=0):
    #     if t==0:
    #         return self.action_space
    #     else:
    #         return self.action_space_comm

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        actions = list(action_dict.values())
        if self.speaker_step:
            for i in range(self.n_agents):
                s_id = i//2*2
                l_id = s_id + 1
                rew = actions[l_id] == self.target
                if i%2==0:
                    obs[i], rew[i], done[i], info[i] = self.speaker_input, rew, True, {}
                else:
                    obs[i], rew[i], done[i], info[i] = self.listener_input, rew, True, {}
            done["__all__"] = True
        else:
            for i in range(self.n_agents):
                if i%2==1:
                    self.heard_message = actions[(i//2)*2]
                    self.listener_input = np.array(self.get_listener_input(self.heard_message))
                    obs[i], rew[i], done[i], info[i] = self.listener_input, 0, True, {}
            self.speaker_step = True
            done["__all__"] = False

        return obs, rew, done, info
