import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_env import PokerEnvironment
from texasholdem.game.player_state import PlayerState
from texasholdem.game.action_type import ActionType
from texasholdem.game import GameState
from agents.dqn_agent import DQNAgent
from buffer.dqn_replay_buffer import ReplayBuffer
from llm_agent import LLMAgent
from buffer.dqn_replay_buffer import RaiseReplayBuffer


import numpy as np
from datetime import datetime
import logging
from typing import List, Union
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)
# Optionally, also suppress openai logging
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Load API keys
with open('apikeys.json') as f:
    apikeys = json.load(f)

# Define personality mapping
PERSONALITY_MAP = {
    # Short codes to full names
    'a': "aggressive",
    'c': "conservative",
    'b': "balanced",
    # Full names to themselves for convenience
    "aggressive": "aggressive",
    "conservative": "conservative",
    "balanced": "balanced"
}

def validate_personalities(personalities: List[str]) -> List[str]:
    """
    Validate and convert personality codes/names to full names
    Args:
        personalities: List of personality identifiers 
                      (can be either codes ['a', 'c', 'b'] or 
                      full names ['aggressive', 'conservative', 'balanced'])
    Returns:
        List of full personality names
    Raises:
        ValueError if invalid personality identifier found
    """
    valid_identifiers = set(PERSONALITY_MAP.keys())
    
    for p in personalities:
        if p not in valid_identifiers:
            raise ValueError(
                f"Invalid personality '{p}'. Must be one of: "
                f"codes {['a', 'c', 'b']} or "
                f"full names {['aggressive', 'conservative', 'balanced']}"
            )
    
    return [PERSONALITY_MAP[p] for p in personalities]

def train(
    num_episodes: int = 10000,
    num_players: int = 6,
    opponent_personalities: Union[List[str], None] = None,
    batch_size: int = 64,
    save_interval: int = 100,
    target_network_update_interval: int = 50,
    save_model_interval: int = 1000
):
    """
    Train DQN agent against LLM opponents
    Args:
        num_episodes: Number of episodes to train
        num_players: Total number of players (including DQN agent)
        opponent_personalities: List of personality codes for opponents ('a', 'c', 'b').
                              If None, will use balanced mix of personalities.
                              Must have length of num_players - 1
        batch_size: Batch size for DQN updates
        save_interval: Interval for saving metrics and plots
        target_network_update_interval: Interval for updating target network
    """
    # Initialize environment and directories
    env, dirs = _initialize_training(num_players, opponent_personalities)
    
    # Initialize agents and buffers
    agent, llm_opponents, replay_buffer, raise_buffer = _initialize_agents(
        env, opponent_personalities, dirs
    )
    
    # Training loop
    total_steps = 0
    for episode in tqdm(range(num_episodes)):
        episode_reward = _run_episode(
            env, agent, llm_opponents, 
            replay_buffer, raise_buffer,
            batch_size, dirs['transitions'], episode
        )
        
        # Update networks and save progress
        _update_and_save(
            agent, episode, episode_reward,
            total_steps, save_interval,
            save_model_interval,
            target_network_update_interval,
            dirs
        )
        total_steps += 1
    
    # Final save
    agent.save_metrics()
    agent.save_model("final")
    logger.info(f"Training completed. All files saved in {dirs['root']}")

def _initialize_training(num_players, opponent_personalities):
    """Initialize environment and create directories"""
    # Validate inputs
    if num_players < 2:
        raise ValueError("Must have at least 2 players")
        
    # Set up opponent personalities
    num_opponents = num_players - 1
    if opponent_personalities is None:
        opponent_personalities = ['b'] * num_opponents
    elif len(opponent_personalities) != num_opponents:
        raise ValueError(f"Must specify {num_opponents} opponent personalities")
    
    opponent_personalities = validate_personalities(opponent_personalities)
    
    # Log configuration
    logger.info("Training Configuration:")
    logger.info(f"Number of players: {num_players}")
    logger.info(f"Opponent personalities: {opponent_personalities}")
    logger.info("-" * 50)
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirs = _create_directories(timestamp)
    
    # Initialize environment
    env = PokerEnvironment(num_players=num_players)
    
    # Save configuration
    _save_config(dirs['root'], num_players, opponent_personalities)
    
    return env, dirs

def _initialize_agents(env, opponent_personalities, dirs):
    """Initialize DQN agent, LLM opponents, and replay buffers"""
    # Initialize DQN agent
    state_dim = env.reset().shape[0]
    action_dim = 5  # FOLD, CHECK, CALL, RAISE, ALL_IN
    agent = DQNAgent(
        state_dim, 
        action_dim,
        model_dir=dirs['models'],
        metrics_dir=dirs['metrics'],
        plots_dir=dirs['plots']
    )
    
    # Initialize LLM opponents
    llm_opponents = [
        LLMAgent(
            api_key=apikeys["openai_api_key"], 
            personality=personality,
            log_dir=os.path.join(dirs['logs'], 'llm_decisions')
        ) for personality in opponent_personalities
    ]
    
    # Initialize replay buffers
    replay_buffer = ReplayBuffer(100000)
    raise_buffer = RaiseReplayBuffer()
    
    return agent, llm_opponents, replay_buffer, raise_buffer

def _run_episode(env, agent, llm_opponents, replay_buffer, raise_buffer, batch_size, transitions_dir, episode):
    """Run a single episode and collect transitions"""
    # Reset all players' states before starting new episode
    for player in env.game.players:
        player.state = PlayerState.IN
        player.chips = env.game.buyin  # Use buyin instead of starting_stack
        player.last_pot = 0  # Reset last pot tracking
    
    # Reset game state to RUNNING
    env.game.game_state = GameState.RUNNING
    
    state = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        # Get current player
        current_player = env.game.current_player
        
        # Get valid actions and stack info before any action
        valid_actions = env._get_valid_actions()
        stack_info = _get_stack_info(env) if valid_actions.get(ActionType.RAISE) else None
        
        # DQN agent's turn
        if current_player == 0:
            if env.game.players[0].state == PlayerState.OUT:
                continue
                
            action = agent.select_action(state, valid_actions, stack_info)
            
        # LLM opponents' turns
        else:
            if env.game.players[current_player].state == PlayerState.OUT:
                continue
                
            opponent_idx = current_player - 1
            action = llm_opponents[opponent_idx].get_action(env.game, current_player)
        
        # Validate and adjust raise amounts for both agents
        if action[0] == ActionType.RAISE:
            min_raise = stack_info['min_raise']
            max_raise = stack_info['max_raise']
            current_bet = stack_info['current_bet']
            
            if action[1] is None or action[1] < min_raise:
                # If raise amount is invalid, convert to call
                chips_to_call = stack_info['chips_to_call']
                action = (ActionType.CALL, chips_to_call)
            else:
                # Ensure raise amount is within valid range
                total_amount = min(max_raise, max(min_raise, action[1]))
                action = (action[0], total_amount)
        
        # Take action in environment
        next_state, reward, done, _ = env.step(action)
        
        # Store transition
        transition = {
            "state": state.tolist(),
            "action_type": int(action[0].value),
            "total_amount": action[1],
            "reward": float(reward),
            "next_state": next_state.tolist(),
            "done": done,
            "player": current_player  # Track which player made the move
        }
        
        # Save transition to file
        transition_file = os.path.join(transitions_dir, f'episode_{episode}.jsonl')
        with open(transition_file, 'a') as f:
            json.dump(transition, f)
            f.write('\n')
        
        # If DQN agent's turn, update networks
        if current_player == 0:
            # Store in replay buffer
            replay_buffer.push(state, action[0], reward, next_state, done)
            episode_reward += reward
            
            # Update DQN
            loss = agent.update(replay_buffer, batch_size)
            if loss > 0:
                agent.losses.append(loss)
        
        # Store raise data if applicable (from any player)
        if action[0] == ActionType.RAISE:
            # Get stack info for the current player making the raise
            current_stack_info = _get_stack_info(env)
            raise_buffer.push(
                state, 
                action[1], 
                reward,
                current_stack_info['min_raise'],  # Store min_raise
                current_stack_info['max_raise']   # Store max_raise
            )
            
            # Train raise network periodically
            if len(raise_buffer.states) >= batch_size:
                states, raises, rewards, min_raises, max_raises = raise_buffer.sample(batch_size)
                raise_loss = agent.update_raise_network(states, raises, rewards, min_raises, max_raises)
        
        state = next_state
    
    return episode_reward

def _update_and_save(agent, episode, episode_reward, total_steps, save_interval, save_model_interval, target_update_interval, dirs):
    """Update networks and save progress"""
    # Log episode results
    won = episode_reward > 0
    agent.log_episode(episode_reward, won)
    
    # Update target network
    if episode % target_update_interval == 0:
        agent.update_target_network()
    
    # Save metrics and plot progress
    if episode % save_interval == 0:
        agent.save_metrics()
        _log_progress(agent, episode, total_steps)
        
        # Plot raise network loss if available
        if agent.raise_losses:
            _plot_raise_loss(agent, dirs['plots'])
            
    # Save model periodically
    if episode % save_model_interval == 0:
        agent.save_model(f"episode_{episode}")

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

# Helper functions for directory creation, config saving, and logging
def _create_directories(timestamp):
    """Create directory structure for saving training data"""
    base_dir = "results/dqn_poker_training"
    run_dir = f"run_{timestamp}"
    
    dirs = {
        'root': os.path.join(base_dir, run_dir),
        'models': os.path.join(base_dir, run_dir, 'models'),
        'transitions': os.path.join(base_dir, run_dir, 'transitions'),
        'metrics': os.path.join(base_dir, run_dir, 'metrics'),
        'plots': os.path.join(base_dir, run_dir, 'plots'),
        'logs': os.path.join(base_dir, run_dir, 'logs')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    # Set up file logging
    file_handler = logging.FileHandler(os.path.join(dirs['logs'], 'training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return dirs

def _save_config(root_dir, num_players, opponent_personalities):
    """Save training configuration"""
    config = {
        "num_players": num_players,
        "opponent_personalities": opponent_personalities,
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(root_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

def _log_progress(agent, episode, total_steps):
    """Log training progress"""
    avg_reward = np.mean(agent.episode_rewards[-100:])
    win_rate = np.mean(agent.win_history[-100:])
    logger.info(f"Episode {episode}")
    logger.info(f"Average Reward (last 100): {avg_reward:.2f}")
    logger.info(f"Win Rate (last 100): {win_rate:.2f}")
    logger.info(f"Epsilon: {agent.epsilon:.3f}")
    logger.info(f"Total Steps: {total_steps}")
    logger.info("-" * 50)

def _plot_raise_loss(agent, plots_dir):
    """Plot raise network loss"""
    plt.figure(figsize=(10, 5))
    plt.plot(agent.raise_losses, label='Raise Network Loss')
    plt.title("Raise Network Training Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'raise_network_loss.png'))
    plt.close()

if __name__ == "__main__":
    # # used for testing
    # opponent_types = ['a']  # For 2 players (1 opponent)
    # train(num_episodes=400, num_players=2, opponent_personalities=opponent_types)
    
    # 7 players, 6 balanced opponents
    opponent_types = ['b', 'b', 'b']
    train(num_players=4, opponent_personalities=opponent_types)