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

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rewards.rewards import PlayerStateQuality, FaceBallReward, AdvancedInAirReward, CarDestroyer
from rewards.rewards_ddp import InAirReward, SpeedTowardBallReward, VelocityBallToGoalReward

from pynput import keyboard

pressed_keys = set()

def on_press(key):
    try:
        pressed_keys.add(key.char)
    except AttributeError:
        pressed_keys.add(key)  

def on_release(key):
    try:
        pressed_keys.discard(key.char)
    except AttributeError:
        pressed_keys.discard(key)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()



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
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
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
        (AdvancedInAirReward(), 5),
        (CarDestroyer(), 20),
    ),
    termination_cond=GoalCondition(),
    truncation_cond=AnyCondition(
        TimeoutCondition(timeout_seconds=300.),
        NoTouchTimeoutCondition(timeout_seconds=30.)
    ),
    transition_engine=RocketSimEngine(),
    renderer=RLViserRenderer()
)

from rlgym_ppo.ppo import PPOLearner
import torch 

obs_dim = 92
act_dim = 90 
device = "cpu"

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
        policy_layer_sizes=[1024, 512],
        critic_layer_sizes=[1024, 512],
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

policy1 = load_policy("/Users/rahulraman/Desktop/studies/semester-4/RL/project/inference/checkpoints/train_ppo_4-1745981774458124084/1625428828")

render = True

control_state = {
    "throttle": 0,
    "steer": 0,
    "boost": 0,
    "jump": 0,
    "yaw": 0,
    "pitch": 0,
    "roll": 0,
    "handbrake": 0
}

def update_control_state():
    """Read pressed_keys and update control_state for this frame."""
    control_state["throttle"] = int(keyboard.Key.up in pressed_keys or 'w' in pressed_keys) - int(keyboard.Key.down in pressed_keys or 's' in pressed_keys)
    control_state["steer"]   = int(keyboard.Key.right in pressed_keys or 'd' in pressed_keys) - int(keyboard.Key.left in pressed_keys or 'a' in pressed_keys)
    control_state["boost"]   = int(keyboard.Key.shift in pressed_keys)
    control_state["jump"]    = int(keyboard.Key.space in pressed_keys)
    control_state["roll"]    = int('e' in pressed_keys) - int('q' in pressed_keys)
    control_state["yaw"]     = int(keyboard.Key.right in pressed_keys) - int(keyboard.Key.left in pressed_keys)
    control_state["pitch"]   = int(keyboard.Key.up in pressed_keys) - int(keyboard.Key.down in pressed_keys)
    control_state["handbrake"]= int('x' in pressed_keys)


idle_idx = np.where(
    np.all(lookUpTable == np.zeros(8, dtype=int), axis=1)
)[0][0]

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

        # Update our control_state from all keys currently down
        update_control_state()
        is_aerial = any([
            control_state["jump"],
            control_state["pitch"] != 0,
            control_state["roll"]  != 0,
            control_state["yaw"]   != 0
        ])
        if not is_aerial:
            human_action = [
                control_state["throttle"] or control_state["boost"],
                control_state["steer"],
                0,
                control_state["steer"],
                0,
                0,
                control_state["boost"],
                control_state["handbrake"]
            ]
        else:
            human_action = [
                control_state["boost"],
                control_state["yaw"],
                control_state["pitch"],
                control_state["yaw"],
                control_state["roll"],
                control_state["jump"],
                control_state["boost"],
                control_state["handbrake"]
            ]

        human_action = np.array(human_action)

        matches = np.all(lookUpTable == human_action, axis=1)
        index = np.where(matches)[0]

        actions["blue-0"]   = index
        actions["orange-0"], _ = policy1.get_action(obs_dict["orange-0"])

        obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions)
        steps += 1
        for aid, r in reward_dict.items():
            ep_reward[aid] += r

        if any(chain(terminated_dict.values(), truncated_dict.values())):
            break

    ep_time = time.time() - t0
    print(f"Steps/sec: {steps/ep_time:.0f} | Time: {ep_time:.2f}s | Reward: {max(ep_reward.values()):.2f}")
    print(ep_reward)
