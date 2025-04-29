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

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rewards.rewards import PlayerStateQuality, FaceBallReward
from rewards.rewards_ddp import InAirReward, SpeedTowardBallReward, VelocityBallToGoalReward


def make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])

        return np.array(actions)

lookUpTable = make_lookup_table()

env = RLGym(
    state_mutator=MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator()
    ),
    obs_builder=DefaultObs(zero_padding=None),
    action_parser=RepeatAction(LookupTableAction(), repeats=8),
    reward_fn=CombinedReward(
        (InAirReward(), 0.05),
        (GoalReward(), 20),
        (SpeedTowardBallReward(), 0.05),
        #(FaceBallReward(), 0.05),
        (TouchReward(), 2),
        (PlayerStateQuality(), 2.),
        (VelocityBallToGoalReward(), 1)
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

from rlgym_ppo.util import KBHit
kb = KBHit()
prev_action = 'c'

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

        
        c = kb.getch() if kb.kbhit() else prev_action
        # [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake]
        if c == 'w':
            human_action = [1, 0, 0, 0, 0, 0, 1, 0]
        elif c == 'a':
            human_action = [1, -1, 0, -1, 0, 0, 0, 0]
        elif c == 'd':
            human_action = [1, 1, 0, 1, 0, 0, 0, 0]
        elif c == 's':
            human_action = [-1, 0, 0, 0, 0, 0, 0, 0]
        elif c == 'c':
            human_action = [0, 0, 0, 0, 0, 0, 0, 0]
        
        # print("ACTION: ", c)
        #prev_action = c

        human_action = np.array(human_action)
        matches = np.all(lookUpTable == human_action, axis=1)
        index = np.where(matches)[0]
        # print("index:", index, index.shape)
        actions["blue-0"] = index
        actions['orange-0'] = np.random.randint(90, size=(1,))
        # print(actions["blue-0"].shape, actions['orange-0'].shape)

        obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions)

        steps += 1
        # print("reward dict: ", reward_dict)
        for agent_id, reward in reward_dict.items():
            ep_reward[agent_id] += reward

        if any(chain(terminated_dict.values(), truncated_dict.values())):
            break

    ep_time = time.time() - t0
    print("Steps per second: {:.0f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(
        steps / ep_time, ep_time, max(ep_reward.values())))
    print(ep_reward)
    