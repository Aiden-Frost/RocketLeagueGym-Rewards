import os
os.environ["WANDB_API_KEY"] = "ENTER HERE"

def build_rlgym_v2_env():
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league import common_values
    from rlgym_ppo.util import RLGymV2GymWrapper
    import numpy as np

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds), TimeoutCondition(timeout_seconds=game_timeout_seconds))

    
    reward_fn = CombinedReward(
        # Format is (func, weight)
        (InAirReward(), 0.15),
        (SpeedTowardBallReward(), 5),
       #(VelocityBallToGoalReward(), 1),
        (GoalReward(), 25),
        #(EventReward(touch=1), 25),
        (TouchReward(), 10),
    )

    obs_builder = DefaultObs(zero_padding=None,
                             pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
                             ang_coef=1 / np.pi,
                             lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                             ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
                             boost_coef=1 / 100.0,)

    state_mutator = MutatorSequence(FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
                                    KickoffMutator())
    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine())

    return RLGymV2GymWrapper(rlgym_env)


import numpy as np # Import numpy, the python math library
from rlgym.rocket_league.api import GameState # Import game state stuff
from rlgym.api import RewardFunction, AgentID
from typing import List, Dict, Any


class InAirReward(RewardFunction): # We extend the class "RewardFunction"
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass # Don't do anything when the game resets

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        return 1. if not state.cars[agent].on_ground else 0.


# Import CAR_MAX_SPEED from common game values
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

class SpeedTowardBallReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}
    
    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        player_vel = state.cars[agent].physics.linear_velocity
        pos_diff = (state.ball.position - state.cars[agent].physics.position)
        dist_to_ball = np.linalg.norm(pos_diff)
        dir_to_ball = pos_diff / dist_to_ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)
        if speed_toward_ball > 0:
            # We are moving toward the ball at a speed of "speed_toward_ball"
            # The maximum speed we can move toward the ball is the maximum car speed
            # We want to return a reward from 0 to 1, so we need to divide our "speed_toward_ball" by the max player speed
            reward = speed_toward_ball / CAR_MAX_SPEED
            return reward
        else:
            # We are not moving toward the ball
            # Many good behaviors require moving away from the ball, so I highly recommend you don't punish moving away
            # We'll just not give any reward
            return 0

if __name__ == "__main__":
    from rlgym_ppo import Learner

    # 8 processes
    n_proc = 1

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_rlgym_v2_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=None,
                      ppo_batch_size=100_000, # batch size - set this number to as large as your GPU can handle
                      policy_layer_sizes=[2048, 2048, 1024, 1024], # policy network
                      critic_layer_sizes=[2048, 2048, 1024, 1024], # value network
                      ts_per_iteration=100_000, # timesteps per training iteration - set this equal to the batch size
                      exp_buffer_size=200_000, # size of experience buffer - keep this 2 - 3x the batch size
                      ppo_minibatch_size=50000, # minibatch size - set this less than or equal to the batch size
                      ppo_ent_coef=0.01, # entropy coefficient - this determines the impact of exploration on the policy
                      policy_lr=2e-4, # policy learning rate
                      critic_lr=2e-4, # value function learning rate
                      ppo_epochs=2,   # number of PPO epochs
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=1_000_000, # save every 1M steps
                      timestep_limit=1_000_000_000, # Train for 1B steps
                      log_to_wandb=True,
                      checkpoints_save_folder="train_ppo_1",
                      checkpoint_load_folder="/scratch/rr4549/RL/train_ppo_1-1745437873454411429/9001946",
                      wandb_run_name="train_ppo_1",
                      device="cuda:0")
    learner.learn()