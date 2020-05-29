
import ray
import argparse
import gym
import random


# import ray
from ray import tune
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.tests.test_multi_agent_env import MultiCartpole
from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from ray.rllib.utils.schedules import LinearSchedule
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule
import gym
import sys

from ref_game import RFGame
from models import SpeakerModel, ListenerModel

n_agents = 2
n_features = 3
n_clusters = 5
n_samples = 5
n_vocab = 3
max_len = 2

argumentList = sys.argv
tf = try_import_tf()
try:
    ray.init(num_cpus=1)
except:
    print("ray already imported")

RF = RFGame(n_agents, n_features, n_clusters, n_samples, n_vocab)
s_obs_space = RF.s_obs_space
s_act_space = RF.s_act_space

l_obs_space = RF.l_obs_space
l_act_space = RF.l_act_space

# Simple environment with `num_agents` independent cartpole entities
register_env("watershed", lambda _: RF)
ModelCatalog.register_custom_model("s_model", SpeakerModel)
ModelCatalog.register_custom_model("l_model", ListenerModel)

def gen_policy(i):
    if i%2==0:
        config = {
            "model": {
                        "custom_model": "s_model",
                    }
                }
        return (None, s_obs_space, s_act_space, config)
    else:
        config = {
            "model": {
                        "custom_model": "l_model",
                    }
                }
        return (None, l_obs_space, l_act_space, config)

policies = {
        i: gen_policy(i)
        for i in range(n_agents)
    }

policy_ids = list(policies.keys())


trainer = PPOTrainer(config={

            "log_level": "WARN",
            "simple_optimizer": True,
            "num_sgd_iter": 10,
            "num_workers": 0,
            "batch_mode": "complete_episodes",
            "entropy_coeff": 0.01,
            # 'entropy_coeff_schedule' : 200,
            "train_batch_size": 256,
            "sample_batch_size": 128,
            "sgd_minibatch_size": 128,
            # "vf_share_layers":True,
            "vf_loss_coeff": 0.005,
            "lr": 5e-3,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (
                    lambda agent_id: policy_ids[agent_id]),
            }
        }, env="watershed")

exp_dict = {
        'name': 'watershed'
        'run_or_experiment': trainer,
        "stop": {
            "training_iteration": 20000
        },
        'checkpoint_freq': 20,
        # "local_dir": "/content/gdrive/My Drive/watershed_exps"
    }


tune.run(**exp_dict, queue_trials=True)
