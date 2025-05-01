from typing import List, Dict, Any
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
import numpy as np
from rlgym.rocket_league.math import cosine_similarity


class PlayerStateQuality(RewardFunction[AgentID, GameState, float]):
    """reward the agent based on the state of the player and game."""

    def __init__(self):
        self.previous_player_qualities = {'blue-0': 0, 'orange-0':0}
        self.previos_state_quality = 0

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.previous_player_qualities = {'blue-0': 0, 'orange-0':0}
        self.previos_state_quality = 0
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        
        BLUE_GOAL = (np.array(common_values.BLUE_GOAL_BACK) + np.array(common_values.BLUE_GOAL_CENTER)) / 2
        ORANGE_GOAL = (np.array(common_values.ORANGE_GOAL_BACK) + np.array(common_values.ORANGE_GOAL_CENTER)) / 2
        rewards = {}
        ball_pos = state.ball.position
        state_quality = 10 * (np.exp(-np.linalg.norm(ORANGE_GOAL - ball_pos) / common_values.CAR_MAX_SPEED)
                                                  - np.exp(-np.linalg.norm(BLUE_GOAL - ball_pos) / common_values.CAR_MAX_SPEED))

        player_quality = dict()

        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics

            pos = car_physics.position
            alignment = 0.5 * (cosine_similarity(ball_pos - pos, common_values.ORANGE_GOAL_BACK - pos)
                               - cosine_similarity(ball_pos - pos, common_values.BLUE_GOAL_BACK - pos))
            if car.team_num == common_values.ORANGE_TEAM:
                alignment *= -1

            liu_dist = np.exp(-np.linalg.norm(ball_pos - pos) / common_values.CAR_MAX_SPEED)
            player_quality[agent] = (0.25 * liu_dist + 0.25 * alignment)


            rewards[agent] = player_quality[agent] - self.previous_player_qualities[agent]
            if car.team_num == common_values.BLUE_TEAM:
                rewards[agent] += state_quality - self.previos_state_quality
            else:
                rewards[agent] -= state_quality - self.previos_state_quality

        self.previous_player_qualities = player_quality
        self.previos_state_quality = state_quality

        return rewards


class FaceBallReward(RewardFunction):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        
        rewards = {}
        for agent in agents:
            car_physics = state.cars[agent].physics
            pos_diff = state.ball.position - car_physics.position
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            rewards[agent] = float(np.dot(car_physics.forward, norm_pos_diff))
        
        return rewards
    


class AdvancedInAirReward(RewardFunction[AgentID, GameState, float]):
    """Advanced Rewards the agent for being in the air and shooting"""
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        
        rewards = {agent: 0 for agent in agents}

        for agent in agents:
            car = state.cars[agent]
            car_height = car.physics.position[2]
            ball_height = state.ball.position[2]
            if car.ball_touches > 0:
                avg_height = 0.5 * (car_height + ball_height)
                h0 = np.cbrt((float(0) - 150) / common_values.CEILING_Z)
                h1 = np.cbrt((float(common_values.CEILING_Z) - 150) / common_values.CEILING_Z)
                hx = np.cbrt((float(avg_height) - 150) / common_values.CEILING_Z)
                height_factor = ((hx - h0) / (h1 - h0)) ** 2
                rewards[agent] += 10 * height_factor
            
            if car.on_ground and car_height < common_values.BALL_RADIUS:
                rewards[agent] -= 0.0005
        
        return rewards
    
class BoostReward(RewardFunction[AgentID, GameState, float]):
    """reward for saving boost when on ground and and using boost when aerial."""
    
    def __init__(self):
        self.previous_boost_amount = {'blue-0': 0, 'orange-0':0}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.previous_boost_amount = {'blue-0': 0, 'orange-0':0}
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        
        rewards = {}

        for agent in agents:
            car = state.cars[agent]
            car_height = car.physics.position[2]

            boost_diff = (np.sqrt(np.clip(car.boost_amount, 0, 1))
                          - np.sqrt(np.clip(self.previous_boost_amount[agent], 0, 1)))
            
            self.previous_boost_amount[agent] = car.boost_amount

            if boost_diff >= 0:
                rewards[agent] = boost_diff
            elif car_height < common_values.GOAL_HEIGHT:
                rewards[agent] = 2 * boost_diff * (1 - car_height / common_values.GOAL_HEIGHT)
        
        return rewards 
            
class CarDestroyer(RewardFunction[AgentID, GameState, float]):
    
    def __init__(self):
        self.previous_demoed = {'blue-0': False, 'orange-0':False}
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.previous_demoed = {'blue-0': False, 'orange-0':False}
    
    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            if car.is_demoed and not self.previous_demoed[agent]:
                rewards[agent] = -1
                self.previous_demoed[agent] = True
                rewards['blue-0' if agent=='orange-0' else 'orange-0'] = 1
        
        return rewards


class TouchRewardPenalize(RewardFunction[AgentID, GameState, float]):
    """
    penalize when enemy has the ball. For more ball possession.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        
        rewards = {'blue-0': 0, 'orange-0':0}
        if state.cars['blue-0'].ball_touches > 0:
            rewards['orange-0'] = -1
        if state.cars['blue-0'].ball_touches > 0:
            rewards['blue-0'] = -1
        
        return