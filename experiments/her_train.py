import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_env import PokerEnvironment
from texasholdem.game.player_state import PlayerState
from texasholdem.game.action_type import ActionType
from texasholdem.game.game import GameState
from agents.her_agent import HERAgent
from buffer.her_replay_buffer import HERBuffer
from llm_agent import LLMAgent
from poker_utils import calculate_hand_strength, calculate_potential

import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def train(
    num_episodes: int = 10000,
    num_players: int = 6,
    opponent_personalities: Optional[List[str]] = None,
    batch_size: int = 32,
    save_interval: int = 100,
    target_update_interval: int = 10,
    k_goals: int = 4,
    optimization_steps: int = 40  # Number of optimization steps per episode
):
    """
    Train HER agent against LLM opponents following the HER algorithm
    """
    # Initialize environment and directories
    env, dirs = _initialize_training(num_players, opponent_personalities)
    
    # Initialize agents and buffer
    agent = HERAgent(
        state_dim=env.reset().shape[0],
        action_dim=5,  # FOLD, CHECK, CALL, RAISE, ALL_IN
        k_goals=k_goals,
        model_dir=dirs['models'],
        metrics_dir=dirs['metrics'],
        plots_dir=dirs['plots']
    )
    
    her_buffer = HERBuffer(100000)
    llm_opponents = _initialize_opponents(opponent_personalities, dirs)
    
    # Training loop (following Algorithm 1)
    for episode in tqdm(range(num_episodes)):
        episode_transitions = []
        episode_achieved_goals = []
        
        # Sample initial goal and state
        state = env.reset()
        goal = _sample_initial_goal(env)
        
        # Run episode and collect experience
        done = False
        episode_reward = 0
        
        while not done:
            # Get current player
            current_player = env.game.current_player
            
            # DQN agent's turn
            if current_player == 0:
                if env.game.players[0].state == PlayerState.OUT:
                    continue
                    
                # Get action from HER agent
                valid_actions = env._get_valid_actions()
                stack_info = _get_stack_info(env) if valid_actions.get(ActionType.RAISE) else None
                action = agent.select_action(state, goal)
                
            # LLM opponents' turns
            else:
                if env.game.players[current_player].state == PlayerState.OUT:
                    continue
                    
                opponent_idx = current_player - 1
                action = llm_opponents[opponent_idx].get_action(env.game, current_player)
            
            # Execute action in environment
            next_state, reward, done, info = env.step(action)
            
            # Record achieved goal for current state
            achieved_goal = _compute_achieved_goal(env, info)
            episode_achieved_goals.append(achieved_goal)
            
            # Store transition if it's agent's turn
            if current_player == 0:
                transition = (state, action, reward, next_state, done, goal, achieved_goal)
                episode_transitions.append(transition)
                episode_reward += reward
            
            state = next_state
            
        # Store original transitions in replay buffer
        for transition in episode_transitions:
            her_buffer.push(*transition)
        
        # Sample additional goals for HER (Strategy S)
        additional_goals = agent.sample_goals(episode_achieved_goals, episode_transitions)
        
        # Store additional transitions with new goals
        for new_goal in additional_goals:
            for t in episode_transitions:
                state, action, _, next_state, done, _, achieved = t
                
                # Compute reward for new goal
                new_reward = agent.compute_reward(achieved, new_goal, info)
                
                # Store new transition
                her_buffer.push(state, action, new_reward, next_state, done, new_goal, achieved)
        
        # Perform optimization steps
        if len(her_buffer) >= batch_size:
            for _ in range(optimization_steps):
                loss = agent.update(her_buffer, batch_size)
                if loss is not None:
                    agent.losses.append(loss)
        
        # Update target network
        if episode % target_update_interval == 0:
            agent.update_target_network()
        
        # Log and save progress
        agent.log_episode(episode_reward, episode_reward > 0)
        if episode % save_interval == 0:
            agent.save_model(f"episode_{episode}")
            agent.save_metrics()
            _log_progress(agent, episode)
    
    # Final save
    agent.save_metrics()
    agent.save_model("final")
    logger.info(f"Training completed. All files saved in {dirs['root']}")

def _get_stack_info(env):
    """Get stack information for raise actions"""
    current_player = env.game.current_player
    return {
        'player_stack': env.game.players[current_player].chips,
        'current_bet': env.game.player_bet_amount(current_player),
        'chips_to_call': env.game.chips_to_call(current_player),
        'min_raise': env.game.min_raise(),
        'max_raise': env.game.player_bet_amount(current_player) + env.game.players[current_player].chips
    }

def _initialize_training(num_players: int, opponent_personalities: List[str]) -> Tuple[PokerEnvironment, Dict[str, str]]:
    """Initialize environment and directories"""
    # Create environment
    env = PokerEnvironment(num_players=num_players)
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirs = {
        'root': f"results/her_poker_training/run_{timestamp}",
        'models': f"results/her_poker_training/run_{timestamp}/models",
        'metrics': f"results/her_poker_training/run_{timestamp}/metrics",
        'plots': f"results/her_poker_training/run_{timestamp}/plots",
        'logs': f"results/her_poker_training/run_{timestamp}/logs"
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    # Save configuration
    config = {
        "num_players": num_players,
        "opponent_personalities": opponent_personalities,
        "timestamp": timestamp
    }
    with open(os.path.join(dirs['root'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
        
    return env, dirs

def _initialize_opponents(personalities: List[str], dirs: Dict[str, str]) -> List[LLMAgent]:
    """Initialize LLM opponents"""
    with open('apikeys.json') as f:
        apikeys = json.load(f)
        
    return [
        LLMAgent(
            api_key=apikeys['openai'],
            personality=p,
            log_dir=os.path.join(dirs['logs'], f'opponent_{i}')
        )
        for i, p in enumerate(personalities)
    ]

def _sample_initial_goal(env: PokerEnvironment) -> Dict:
    """Sample initial goal for episode"""
    return {
        'pot_size': np.random.uniform(1, 20) * env.big_blind,  # Target pot size in BB
        'opponent_folded': bool(np.random.randint(2)),  # Target opponent fold
        'hand_strength': np.random.uniform(0, 1),  # Target hand strength
        'stack_size': np.random.uniform(20, 100) * env.big_blind  # Target stack size in BB
    }

def _compute_achieved_goal(env: PokerEnvironment, info: Dict) -> Dict:
    """Compute achieved goal from current state"""
    return {
        'pot_size': sum(pot.get_total_amount() for pot in env.game.pots) / env.big_blind,
        'opponent_folded': info.get('opponent_folded', False),
        'hand_strength': info.get('hand_strength', 0.0),
        'stack_size': env.game.players[env.game.current_player].chips / env.big_blind
    }

def _log_progress(agent: HERAgent, episode: int):
    """Log training progress"""
    avg_reward = np.mean(agent.episode_rewards[-100:])
    win_rate = np.mean(agent.win_history[-100:])
    logger.info(f"Episode {episode}")
    logger.info(f"Average Reward (last 100): {avg_reward:.2f}")
    logger.info(f"Win Rate (last 100): {win_rate:.2f}")
    logger.info(f"Epsilon: {agent.epsilon:.3f}")
    logger.info("-" * 50) 

def _run_episode(env, agent, llm_opponents, her_buffer, batch_size, transitions_dir, episode):
    """Run a single episode and collect transitions"""
    # Reset all players' states before starting new episode
    for player in env.game.players:
        player.state = PlayerState.IN
        player.chips = env.game.buyin
        player.last_pot = 0
    
    # Reset game state to RUNNING
    env.game.game_state = GameState.RUNNING
    
    state = env.reset()
    goal = _sample_initial_goal(env)  # Sample goal at start of episode
    episode_reward = 0
    done = False
    
    episode_transitions = []
    episode_achieved_goals = []
    
    while not done:
        current_player = env.game.current_player
        
        if current_player == 0:  # Agent's turn
            if env.game.players[0].state == PlayerState.OUT:
                continue
                
            # Get valid actions and stack info
            valid_actions = env._get_valid_actions()
            stack_info = _get_stack_info(env) if valid_actions.get(ActionType.RAISE) else None
            
            # Get action from HER agent
            action = agent.select_action(state, goal, valid_actions)
            
        else:  # Opponent's turn
            if env.game.players[current_player].state == PlayerState.OUT:
                continue
                
            opponent_idx = current_player - 1
            action = llm_opponents[opponent_idx].get_action(env.game, current_player)
        
        # Take action in environment
        next_state, reward, done, info = env.step(action)
        
        # Compute achieved goal
        achieved_goal = _compute_achieved_goal(env, info)
        episode_achieved_goals.append(achieved_goal)
        
        # Store transition if agent's turn
        if current_player == 0:
            transition = (state, action, reward, next_state, done, goal, achieved_goal)
            episode_transitions.append(transition)
            episode_reward += reward
            
            # Save transition to file
            transition_data = {
                "state": state.tolist(),
                "action_type": int(action[0].value),
                "total_amount": action[1],
                "reward": float(reward),
                "next_state": next_state.tolist(),
                "done": done,
                "goal": goal,
                "achieved_goal": achieved_goal,
                "player": current_player
            }
            
            with open(os.path.join(transitions_dir, f'episode_{episode}.jsonl'), 'a') as f:
                json.dump(transition_data, f)
                f.write('\n')
        
        state = next_state
    
    return episode_transitions, episode_achieved_goals, episode_reward 

if __name__ == "__main__":
    # used for testing
    opponent_types = ['a']  # For 2 players (1 opponent)
    train(num_episodes=400, num_players=2, opponent_personalities=opponent_types)
    
    # # 6 players, 5 opponents
    # opponent_types = ['b', 'b', 'b', 'b', 'b']
    # train(num_players=6, opponent_personalities=opponent_types)