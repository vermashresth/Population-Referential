import copy
import sys

import numpy as np
import scipy

# TODO(@evinitsky) put this in alphabetical order

from ray.rllib.agents.ppo.ppo_policy import PPOLoss, BEHAVIOUR_LOGITS, \
    KLCoeffMixin, setup_config, clip_gradients, \
    kl_and_loss_stats, ValueNetworkMixin, vf_preds_and_logits_fetches, postprocess_ppo_gae
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, choose_policy_optimizer, \
    validate_config, update_kl, warn_about_bad_reward_scales
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule, ACTION_LOGP
from ray.rllib.utils import try_import_tf, try_import_tfp
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.agents.trainer_template import build_trainer

# from algorithms.common_funcs import setup_moa_loss, causal_fetches, setup_causal_mixins, get_causal_mixins, \
#     causal_postprocess_trajectory, CAUSAL_CONFIG
import random
from scipy.special import softmax
NUM_AGENTS = 4

tf = try_import_tf()
tfp = try_import_tfp()

POLICY_SCOPE = "func"

# CAUSAL_CONFIG.update(DEFAULT_CONFIG)

def my_vf_preds_and_logits_fetches(policy):
    """Adds value function and logits outputs to experience train_batches."""
    return {
        'VF_PREDS_0': policy.model.value_function(0),
        'VF_PREDS_1': policy.model.value_function(1),
        'VF_PREDS_2': policy.model.value_function(2),
        BEHAVIOUR_LOGITS: policy.model.last_output(),
        'UNCERTAINITY': policy.model.uncertainity(),
        'VF_EST': policy.model.value_function(-1),
        'vf_preds': policy.model.value_function(-1)
    }
def loss_with_moa(policy, model, dist_class, train_batch):
    # you need to override this bit to pull out the right bits from train_batch
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    # if policy.model.causal:
    #     moa_loss = setup_moa_loss(logits, model, policy, train_batch)
    #     policy.moa_loss = moa_loss.total_loss

    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(
            train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool)
    policy.loss_obj = PPOLoss(
        policy.action_space,
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[BEHAVIOUR_LOGITS],
        train_batch[ACTION_LOGP],
        train_batch['VF_PREDS_{}'.format(0)],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
        model_config=policy.config["model"])
    for i in range(1, 3):
        policy.loss_obj.loss += PPOLoss(
            policy.action_space,
            dist_class,
            model,
            train_batch[Postprocessing.VALUE_TARGETS],
            train_batch[Postprocessing.ADVANTAGES],
            train_batch[SampleBatch.ACTIONS],
            train_batch[BEHAVIOUR_LOGITS],
            train_batch[ACTION_LOGP],
            train_batch['VF_PREDS_{}'.format(i)],
            action_dist,
            model.value_function(),
            policy.kl_coeff,
            mask,
            entropy_coeff=policy.entropy_coeff,
            clip_param=policy.config["clip_param"],
            vf_clip_param=policy.config["vf_clip_param"],
            vf_loss_coeff=policy.config["vf_loss_coeff"],
            use_gae=policy.config["use_gae"],
            model_config=policy.config["model"]).mean_vf_loss

    # if policy.model.causal:
    #     policy.loss_obj.loss += moa_loss.total_loss
    return policy.loss_obj.loss


def extra_fetches(policy):
    """Adds value function, logits, moa predictions of counterfactual actions to experience train_batches."""
    ppo_fetches = my_vf_preds_and_logits_fetches(policy)
    # if policy.model.causal:
    #     ppo_fetches.update(causal_fetches(policy))
    return ppo_fetches


def extra_stats(policy, train_batch):
    base_stats = kl_and_loss_stats(policy, train_batch)
    # if policy.model.causal:
    #     base_stats["total_influence"] = train_batch["total_influence"]
    #     base_stats['reward_without_influence'] = train_batch['reward_without_influence']
    #     base_stats['moa_loss'] = policy.moa_loss / policy.moa_weight
    return base_stats

saved_batch_item = {}
random_batch_item = {}
def postprocess_ppo_causal(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    '''
    episode mein sathiyon ki uncertainities ko dekho, unke uncertain walon pe predict karo,
    aur agar meri uncertainity kam hai aur prediction mein v estiate high hai to
    ko naya batch bana ke saathi ke episode mein daal do
    '''
    # mygg = policy.compute_actions(np.array([policy.model.obs_space.sample()]))
    # var = float(np.cov([mygg['VF_PREDS_0'], mygg['VF_PREDS_1'], mygg['VF_PREDS_2']])[0])


    if episode is not None:
      if '1' in other_agent_batches and '0' not in other_agent_batches:
        other_batch = other_agent_batches['1'][1]
        # print(other_agent_batches['1'])
        # print(other_batch)
        unc = np.cov([other_batch['VF_PREDS_0'][0], other_batch['VF_PREDS_1'][0], other_batch['VF_PREDS_2'][0]])
        # if i['UNCERTAINITY']>0.3:

        # print('advicee est')
        # if float(unc)>0.3:
        myggg = policy.compute_actions(other_batch['obs'], full_fetch=True)
        # print('other: ', other_agent_batches['1'] )
        # print('me: ', myggg)
        # print(policy.compute_single_action(other_batch['obs'][0], []))
        mygg = myggg[2]
        # print('myest', mygg)
        my_unc = float(np.cov([mygg['VF_PREDS_0'][0], mygg['VF_PREDS_1'][0], mygg['VF_PREDS_2'][0]]))
        p = np.random.random()*100
        if p<1:
          print('advicor unc', unc)
          print('advicor est', other_batch['VF_EST'][0])
          print('my unc', my_unc)
          print('MY est', mygg['VF_EST'][0])
          # print('vfs', mygg['VF_PREDS_0'][0], mygg['VF_PREDS_1'][0])
          print(mygg)
          print()

        #   print(i)
        #   print(dir(i))


        # builder = episode.new_batch_builder()
        # rollout_id = random.randint(0, 10000)
        # builder.add_values(
        #     agent_id="extra_0",
        #     policy_id="1",  # use p1 so we can easily check it
        #     t=0,
        #     eps_id=rollout_id,  # new id for each rollout
        #     obs=other_batch['obs'][0],
        #     actions=myggg[0],
        #     rewards=mygg['VF_EST'][0],
        #     dones=True,
        #     infos={},
        #     new_obs=other_batch['obs'][0],
        #     prev_actions=np.array([0]),
        #     prev_rewards=np.array([0]))
        # more = ['VF_EST', 'VF_PREDS_0', 'VF_PREDS_1', 'VF_PREDS_2', 'UNCERTAINITY']

        # batch = builder.build_and_reset(episode=None)
        # print('ma batch funcs', dir(batch))
        # new_b = batch['1']
        rew = sample_batch['rewards']
        my_probs = softmax(mygg['behaviour_logits'])[0]
        # print(my_probs)
        act = other_batch['actions'][0]
        my_act_prob = my_probs[act]
        my_act_log_prob = np.log(my_act_prob)
        if unc<my_unc and other_batch['rewards'][0]>0.8:
          more = {'t':np.array([0]), 'eps_id':np.array([random.randint(0,10000000)]), 'agent_index': np.array([0]), 'unroll_id':np.array([0]),
                  'obs':other_batch['obs'],
                  'actions':other_batch['actions'], 'rewards':other_batch['rewards'],
                  'prev_actions':np.array([0]),
                  "prev_rewards":np.array([0]),
                  'dones':np.array([True]),
                  'infos':np.array([{}]),
                  'new_obs':other_batch['obs'],
                  'action_prob':np.array([my_act_prob]),
                  'action_logp': np.array([my_act_log_prob])}
          others = ['action_logp', 'action_prob']
          for i in sample_batch:
              if i in mygg and i not in others:
                sample_batch[i] = np.concatenate([sample_batch[i], other_batch[i]], 0)
              else:
                # print(i)
                sample_batch[i] = np.concatenate([sample_batch[i], more[i]], 0)
          print("I just got adviced")
        # sample_batch['infos'][0]['true_rew'] = sample_batch['rewards'][0]
        # sample_batch['infos'][1]['true_rew'] = -1


        # p = np.random.random()*100
        if p <1:
          # print("new_b", new_b)
          # print(batch)
          print("\n sample \n", sample_batch)
          # print("\n my \n ", batch)

    # episodes[0].add_extra_batch(batch)
    # global saved_batch_item
    # global random_batch_item
    # if policy.model.causal:
    #     sample_batch = causal_postprocess_trajectory(policy, sample_batch)
    # else:
    #     if policy.loss_initialized():
    #       id = policy.model.id
    #       other_agent = "agent-{}".format((id+NUM_AGENTS))
    #       if other_agent in other_agent_batches.keys():
    #
    #         other_batch = other_agent_batches[other_agent][1]
    #         other_with_influence = causal_postprocess_trajectory(policy, other_batch)
    #         # print(id, other_agent_batches[other_agent][1].keys())
    #         if len(sample_batch['rewards'])==len(other_with_influence['total_influence']):
    #           sample_batch['rewards'] +=  other_with_influence['total_influence']*policy.curr_influence_weight
    #
    #         elif len(sample_batch['rewards'])>len(other_with_influence['total_influence']):
    #           #Case 1, reward more than influence
    #           #Lets just remve the last value and save it
    #           # print("case 1")
    #           # print(sample_batch['rewards'], other_with_influence['total_influence'])
    #           for key in sample_batch:
    #             saved_batch_item[key] = sample_batch[key][-1]
    #             sample_batch[key] = sample_batch[key][:-1]
    #
    #           # print(saved_batch_item)
    #           # print(sample_batch['rewards'], other_with_influence['total_influence'])
    #           sample_batch['rewards'] +=  other_with_influence['total_influence']*policy.curr_influence_weight
    #
    #           random_val = np.random.randint(len(sample_batch['rewards']))
    #           for key in sample_batch:
    #             random_batch_item[key] = np.array([sample_batch[key][random_val]])
    #         else:
    #           #Case 2, reward is less than influence
    #           #Push saved reward value in the begginig
    #           # print("case 2")
    #           # print(sample_batch['rewards'], other_with_influence['total_influence'])
    #           # print("saved item: ", saved_batch_item)
    #           for key in sample_batch:
    #             sample_batch[key] = np.concatenate(([saved_batch_item[key]], sample_batch[key]))
    #
    #           # print(sample_batch['rewards'], other_with_influence['total_influence'])
    #           # print()
    #           sample_batch['rewards'] +=  other_with_influence['total_influence']*policy.curr_influence_weight
    #
    #           random_val = np.random.randint(len(sample_batch['rewards']))
    #           for key in sample_batch:
    #             random_batch_item[key] = np.array([sample_batch[key][random_val]])
    #       else:
    #         #Case 3 when only 1 extra reward
    #         # print("case 3")
    #         # print("earlier: ", sample_batch)
    #         # print("random: ",random_batch_item)
    #         # print(sample_batch['rewards'], other_with_influence['total_influence'])
    #         for key in sample_batch:
    #             saved_batch_item[key] = sample_batch[key][-1]
    #         # print(saved_batch_item)
    #         # print(sample_batch['rewards'], other_with_influence['total_influence'])
    #         # print()
    #         sample_batch = random_batch_item.copy()
    #         # print("after: ", sample_batch)
    #
    #

        # batch = postprocess_ppo_gae(policy, new_b)
        # return batch
    # print(sample_batch)
    # if 'true_rewards' not in sample_batch:
    #   sample_batch[2]['infos'][0]['true_rew'] = sample_batch['rewards'][0]
    batch = postprocess_ppo_gae(policy, sample_batch)
    return batch


def build_model(policy, obs_space, action_space, config):
    _, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])

    policy.model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        logit_dim,
        config["model"],
        name=POLICY_SCOPE,
        framework="tf")

    return policy.model


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    # setup_causal_mixins(policy, obs_space, action_space, config)


MyPPOPolicy = build_tf_policy(
    name="MyTFPolicy",
    get_default_config=lambda: DEFAULT_CONFIG,
    loss_fn=loss_with_moa,
    make_model=build_model,
    stats_fn=extra_stats,
    extra_action_fetches_fn=extra_fetches,
    postprocess_fn=postprocess_ppo_causal,
    gradients_fn=clip_gradients,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin])
    # ] + get_causal_mixins())

MyPPOTrainer = build_trainer(
    name="MyPPO",
    default_policy=MyPPOPolicy,
    make_policy_optimizer=choose_policy_optimizer,
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    after_optimizer_step=update_kl,
    after_train_result=warn_about_bad_reward_scales)
