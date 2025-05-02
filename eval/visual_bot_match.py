import time
import numpy as np
from itertools import chain
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, AnyCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.rlviser import RLViserRenderer
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator


env = RLGym(
    state_mutator=MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator()
    ),
    obs_builder=DefaultObs(zero_padding=None),
    action_parser=RepeatAction(LookupTableAction(), repeats=8),
    reward_fn=CombinedReward(
        (GoalReward(), 10.),
        (TouchReward(), 0.1)
    ),
    termination_cond=GoalCondition(),
    truncation_cond=AnyCondition(
        TimeoutCondition(timeout_seconds=300.),
        NoTouchTimeoutCondition(timeout_seconds=30.)
    ),
    transition_engine=RocketSimEngine(),
    renderer=RLViserRenderer()
)

render = True

obs_dim = 92
act_dim = 90 
device = "cpu"

from rlgym_ppo.ppo import PPOLearner
import torch 

def load_policy(checkpoint_folder):
    ppo = PPOLearner(
        obs_dim,
        act_dim,
        device=device,
        batch_size=1, 
        mini_batch_size=1,
        n_epochs=1,
        continuous_var_range=(0.1, 1.0),
        policy_type=env.action_space,
        policy_layer_sizes=[2048, 2048, 1024, 1024],
        critic_layer_sizes=[2048, 2048, 1024, 1024],
        policy_lr=3e-4,
        critic_lr=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
    )

    policy_state = torch.load(f"{checkpoint_folder}/PPO_POLICY.pt", map_location=torch.device("cpu"))
    # value_net_state = torch.load(f"{checkpoint_folder}/PPO_VALUE_NET.pt", map_location=torch.device("cpu"))

    # ppo.load_from(checkpoint_folder)
    ppo.policy.load_state_dict(policy_state)
    # ppo.value_net.load_state_dict(value_net_state)
    
    ppo.policy.eval()
    return ppo.policy

policy1 = load_policy("/Users/rahulraman/Desktop/studies/semester-4/RL/project/inference/999704398/")
policy2 = load_policy("/Users/rahulraman/Desktop/studies/semester-4/RL/project/inference/12014334/")

agents = {'blue-0': policy1, 'orange-0':policy2}

while True:
    obs_dict = env.reset()
    steps = 0
    ep_reward = {agent_id: 0 for agent_id in env.agents}
    t0 = time.time()
    actions = {}
    while True:
        if render:
            env.render()
            time.sleep(6/120)

        for agent_id, action_space in env.action_spaces.items():
            actions[agent_id], _ = agents[agent_id].get_action(obs_dict[agent_id])
            # actions[agent_id] = np.random.randint(action_space[1], size=(1,))

        obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions)

        steps += 1
        for agent_id, reward in reward_dict.items():
            ep_reward[agent_id] += reward

        if any(chain(terminated_dict.values(), truncated_dict.values())):
            break

    ep_time = time.time() - t0
    print("Steps per second: {:.0f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(
        steps / ep_time, ep_time, max(ep_reward.values())))
    