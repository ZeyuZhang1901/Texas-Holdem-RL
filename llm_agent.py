from openai import OpenAI
import json
import numpy as np
from texasholdem.game.action_type import ActionType
from texasholdem.game.hand_phase import HandPhase
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LLMAgent:
    def __init__(self, api_key, personality="aggressive", log_dir="llm_decisions"):
        """
        Initialize LLM agent with different playing styles and logging
        Args:
            api_key: OpenAI API key
            personality: Playing style - "aggressive", "conservative", or "balanced" (or 'a', 'c', 'b')
            log_dir: Directory to save LLM decisions
        """
        self.client = OpenAI(api_key=api_key)
        
        # Convert personality code to full name if needed
        PERSONALITY_MAP = {
            'a': "aggressive",
            'c': "conservative",
            'b': "balanced",
            "aggressive": "aggressive",
            "conservative": "conservative",
            "balanced": "balanced"
        }
        
        if personality not in PERSONALITY_MAP:
            raise ValueError(
                f"Invalid personality '{personality}'. Must be one of: "
                f"codes {['a', 'c', 'b']} or "
                f"full names {['aggressive', 'conservative', 'balanced']}"
            )
        
        self.personality = PERSONALITY_MAP[personality]
        
        # Set up logging directory
        self.log_dir = os.path.join(log_dir, self.personality, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize decision log
        self.decision_log_path = os.path.join(self.log_dir, "decisions.jsonl")
        
        # Define personality prompts using full names
        self.personality_prompts = {
            "aggressive": "You are an aggressive poker expert who likes to put pressure on opponents and isn't afraid to make big bets.",
            "conservative": "You are a tight-conservative poker expert who waits for strong hands and makes careful, calculated decisions.",
            "balanced": "You are a balanced poker expert who adapts their strategy based on the situation and combines aggression with careful play."
        }
        
    def format_prompt(self, game_state):
        """
        Format game state into a clear, structured prompt for the LLM
        
        Args:
            game_state: Dictionary containing the formatted game state
            
        Returns:
            str: Formatted prompt
        """
        prompt = f"""{self.personality_prompts[self.personality]}

Below are all the information you have about the game.

Current Game State:
Hole Cards: {', '.join(game_state['hand'])}
Community Cards: {', '.join(game_state['board']) if game_state['board'] else 'None'}
Stage: {game_state['hand_phase']}
Position: {game_state['position']}

Stack Info (in BB):
- Your Stack: {game_state['effective_stack']:.1f}
- Pot Size: {game_state['pot_size']:.1f}
- To Call: {game_state['chips_to_call']:.1f}
- Pot Odds: {game_state['pot_odds']}

Table Info:
- Active Players: {game_state['num_active_players']}
- Raises This Round: {game_state['num_raises']}
- Current Bets: {game_state['current_bets']}

Valid Actions: {', '.join(game_state['available_actions'])}
{f"Raise Range: {game_state['min_raise']:.1f} BB to {game_state['max_raise']:.1f} BB" if game_state['min_raise'] is not None else ""}

As an expert poker player with {self.personality} style, assign probabilities to each available action that sum to exactly 1.0.
Output format must be exactly:
ACTION_PROBABILITIES:
<ACTION>: <PROBABILITY>
<ACTION>: <PROBABILITY>
...

Some examples are provided below:

Example 1:
ACTION_PROBABILITIES:
FOLD: 0.2
CALL: 0.6
RAISE: 0.2

Example 2:
ACTION_PROBABILITIES:
FOLD: 0.3
CHECK: 0.6
RAISE: 0.1

Example 3:
ACTION_PROBABILITIES:
FOLD: 0.1
CALL: 0.1
RAISE: 0.2
ALL_IN: 0.6

Only include valid actions from the list above, Otherwise, we can't extract the probabilities properly. Probabilities must sum to 1.0."""

        return prompt

    def parse_llm_response(self, response_text):
        """
        Parse LLM response to get action probabilities
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            dict: Parsed probabilities for each action or None if parsing fails
        """
        try:
            # Extract the probability section
            if "ACTION_PROBABILITIES:" not in response_text:
                logger.error("Missing ACTION_PROBABILITIES section")
                return None
                
            prob_section = response_text.split("ACTION_PROBABILITIES:")[1].strip()
            
            # Parse probabilities
            action_probs = {}
            total_prob = 0.0
            
            for line in prob_section.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    action, prob = [x.strip() for x in line.split(':')]
                    prob = float(prob)
                    
                    # Validate probability
                    if prob < 0 or prob > 1:
                        logger.error(f"Invalid probability value: {prob}")
                        return None
                        
                    action_probs[action] = prob
                    total_prob += prob
                    
                except ValueError as e:
                    logger.error(f"Error parsing line '{line}': {e}")
                    continue
            
            # Validate total probability
            if not 0.99 <= total_prob <= 1.01:  # Allow small floating point errors
                logger.error(f"Probabilities sum to {total_prob}, not 1.0")
                return None
                
            # Normalize probabilities to exactly 1.0
            action_probs = {k: v/total_prob for k, v in action_probs.items()}
            
            return action_probs
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Response text: {response_text}")
            return None

    def get_action(self, game, player_id):
        """Get action distribution from LLM and sample action"""
        # Get formatted game state
        game_state = self.format_state(game, player_id)
        
        # Generate prompt
        prompt = self.format_prompt(game_state)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            parsed_response = self.parse_llm_response(response_text)
            
            if parsed_response is None:
                # Fallback to conservative play if parsing fails
                logger.warning("Failed to parse LLM response, using fallback action")
                chosen_action = ActionType.CHECK if ActionType.CHECK in game.get_available_moves().action_types else ActionType.FOLD
                self.log_decision(game_state, response_text, chosen_action)
                return chosen_action
            
            # Convert probabilities to valid actions
            action_probs = {}
            for action_name, prob in parsed_response.items():
                try:
                    action = ActionType[action_name.upper()]
                    if action in game.get_available_moves().action_types:
                        action_probs[action] = prob
                except KeyError:
                    continue
            
            # Normalize probabilities
            total_prob = sum(action_probs.values())
            action_probs = {k: v/total_prob for k, v in action_probs.items()}
            
            # Sample action
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            chosen_action = np.random.choice(actions, p=probs)
            
            # Calculate total amount if raise
            total_amount = None
            if chosen_action == ActionType.RAISE:
                moves = game.get_available_moves()
                current_bet = game.player_bet_amount(player_id)
                chips_to_call = game.chips_to_call(player_id)
                available_chips = game.players[player_id].chips
                min_raise = moves.raise_range.start
                max_raise = moves.raise_range.stop - 1
                
                # Calculate raise amount based on personality
                if self.personality == "aggressive":
                    raise_amount = min(max_raise - current_bet - chips_to_call, 
                                     int(available_chips * 0.75))
                elif self.personality == "conservative":
                    raise_amount = min_raise
                else:  # balanced
                    raise_amount = min(max_raise - current_bet - chips_to_call,
                                     int(available_chips * 0.5))
                    
                # Convert to total amount (current_bet + chips_to_call + raise_amount)
                total_amount = current_bet + chips_to_call + raise_amount
                
                # Ensure we don't exceed max raise
                total_amount = min(total_amount, max_raise)
            
            # Log the decision
            self.log_decision(
                game_state,
                {
                    "raw_response": response_text,
                    "parsed_response": parsed_response,
                    "action_probabilities": {str(k): v for k, v in action_probs.items()}
                },
                chosen_action,
                total_amount
            )
            
            return (chosen_action, total_amount)
            
        except Exception as e:
            logger.error(f"Error in get_action: {e}")
            # Fallback to conservative play
            chosen_action = ActionType.CHECK if ActionType.CHECK in game.get_available_moves().action_types else ActionType.FOLD
            self.log_decision(game_state, f"Error: {str(e)}", chosen_action)
            return chosen_action

    def log_decision(self, game_state, llm_response, chosen_action, raise_amount=None):
        """Log LLM decision and context"""
        decision_log = {
            "timestamp": datetime.now().isoformat(),
            "personality": self.personality,
            "game_state": game_state,
            "llm_response": llm_response,
            "chosen_action": str(chosen_action),
            "raise_amount": raise_amount,
            "hand_phase": game_state["hand_phase"]
        }
        
        with open(self.decision_log_path, 'a') as f:
            json.dump(decision_log, f)
            f.write('\n')

    def format_state(self, game, player_id):
        """
        Format the game state into a dictionary containing relevant information
        
        Args:
            game: TexasHoldEm instance
            player_id: ID of the current player
            
        Returns:
            dict: Formatted game state
        """
        # Get player's hand and convert to string representation
        player = game.players[player_id]
        hand = [str(card) for card in game.get_hand(player_id)]
        
        # Get community cards
        board = [str(card) for card in game.board] if game.board else []
        
        # Get available moves
        moves = game.get_available_moves()
        
        # Calculate chips to call
        chips_to_call = game.chips_to_call(player_id)
        
        # Get available actions
        available_actions = [str(action_type).split('.')[1] for action_type in moves.action_types]
        
        # Calculate pot size and pot odds
        total_pot = sum(pot.get_total_amount() for pot in game.pots)
        pot_odds = f"{(chips_to_call / (total_pot + chips_to_call)):.2f}" if chips_to_call > 0 else "N/A"
        
        # Get position relative to button
        position = (player_id - game.btn_loc) % len(game.players)
        position_names = ["BUTTON", "SMALL_BLIND", "BIG_BLIND"] + ["MIDDLE"] * (len(game.players) - 3)
        
        # Update the num_raises calculation to handle PrehandHistory
        num_raises = 0
        if game.hand_history and game.hand_phase in game.hand_history:
            history = game.hand_history[game.hand_phase]
            if hasattr(history, 'actions'):  # Check if history has actions attribute
                num_raises = len([a for a in history.actions if a.action_type == ActionType.RAISE])
        
        return {
            "hand": hand,
            "board": board,
            "position": position_names[position],
            "hand_phase": str(game.hand_phase).split('.')[1],
            "effective_stack": player.chips / game.big_blind,  # Convert to BB
            "pot_size": total_pot / game.big_blind,  # Convert to BB
            "chips_to_call": chips_to_call / game.big_blind,  # Convert to BB
            "pot_odds": pot_odds,
            "num_active_players": sum(1 for p in game.in_pot_iter()),
            "num_raises": num_raises,
            "current_bets": {i: game.pots[-1].get_player_amount(i)/game.big_blind 
                            for i in game.pots[-1].players_in_pot()},
            "available_actions": available_actions,
            "min_raise": moves.raise_range.start / game.big_blind if hasattr(moves, 'raise_range') and moves.raise_range else None,
            "max_raise": (moves.raise_range.stop - 1) / game.big_blind if hasattr(moves, 'raise_range') and moves.raise_range else None
        }