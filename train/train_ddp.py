import os
os.environ["WANDB_API_KEY"] = "ENTER HERE"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rewards.rewards_ddp import SpeedTowardBallReward, InAirReward, VelocityBallToGoalReward

def build_rlgym_v2_env():
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league import common_values
    from rlgym_ppo.util import RLGymV2GymWrapper
    from rlgym.rocket_league.rlviser import RLViserRenderer

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds)
    )

    reward_fn = CombinedReward(
        (InAirReward(), 0.002),
        (SpeedTowardBallReward(), 0.01),
        (VelocityBallToGoalReward(), 0.1),
        (GoalReward(), 10.0)
    )

    obs_builder = DefaultObs(zero_padding=None,
                           pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 
                                              1 / common_values.BACK_NET_Y, 
                                              1 / common_values.CEILING_Z]),
                           ang_coef=1 / np.pi,
                           lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                           ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
                           boost_coef=1 / 100.0)

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator()
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        # renderer=RLViserRenderer()
    )

    return RLGymV2GymWrapper(rlgym_env)

def train(rank, world_size, n_gpus, M):
    from rlgym_ppo import Learner
    
    local_gpu = rank #% n_gpus
    # torch.cuda.set_device(local_gpu)
    # init_method = "file:///tmp/ddp_shared_init"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)#, init_method=init_method)#, init_method="tcp://127.0.0.1:29500")
    
    # 32 processes
    n_proc = 1

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_rlgym_v2_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=None, # Leave this empty for now.
                      ppo_batch_size=100_000,  # batch size - much higher than 300K doesn't seem to help most people
                      policy_layer_sizes=[2048, 2048, 1024, 1024],  # policy network
                      critic_layer_sizes=[2048, 2048, 1024, 1024],  # critic network
                      ts_per_iteration=100_000,  # timesteps per training iteration - set this equal to the batch size
                      exp_buffer_size=300_000,  # size of experience buffer - keep this 2 - 3x the batch size
                      ppo_minibatch_size=50_000,  # minibatch size - set this as high as your GPU can handle
                      ppo_ent_coef=0.01,  # entropy coefficient - this determines the impact of exploration
                      policy_lr=1e-4,  # policy learning rate
                      critic_lr=1e-4,  # critic learning rate
                      ppo_epochs=2,   # number of PPO epochs
                      standardize_returns=True, # Don't touch these.
                      standardize_obs=False, # Don't touch these.
                      save_every_ts=1_000_000,  # save every 1M steps
                      checkpoint_load_folder=None,
                      timestep_limit=1_000_000_000,  # Train for 1B steps
                      log_to_wandb=True, # Set this to True if you want to use Weights & Biases for logging.
                      device=f'cuda:{local_gpu}',
                      rank=rank,
                      local_gpu=local_gpu,
                    ) 
    learner.learn()
    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR",  "127.0.0.1")
    os.environ.setdefault("MASTER_PORT",  "29500")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_P2P_DISABLE"] = "1"
    # os.environ["NCCL_SHM_DISABLE"] = "1"
    # os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    
    n_gpus = torch.cuda.device_count()
    M = 1   # <-- how many DDP ranks you want *per* GPU
    world_size = n_gpus * M
    mp.spawn(
        train, 
        args=(world_size, n_gpus, M), 
        nprocs=world_size, 
        join=True
    )
