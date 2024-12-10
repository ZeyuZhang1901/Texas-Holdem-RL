import numpy as np
from texasholdem.game.game import TexasHoldEm
from texasholdem.game.action_type import ActionType
from texasholdem.game import GameState
from poker_utils import calculate_hand_strength, calculate_potential

class PokerEnvironment:
    def __init__(self, num_players=6, buyin=500, big_blind=5, small_blind=2, gamma=0.99):
        self.num_players = num_players
        self.buyin = buyin
        self.big_blind = big_blind
        self.small_blind = small_blind
        self.gamma = gamma
        
        # Initialize game with correct parameters
        self.game = TexasHoldEm(
            buyin=self.buyin,
            big_blind=self.big_blind,
            small_blind=self.small_blind,
            max_players=self.num_players
        )
        
        # Initialize reward tracking
        self._action = (None, None)
        self._last_potential = None
        
    def encode_cards(self, cards):
        """
        Encode cards into a compact 17-dimensional binary vector representation where:
        - First 13 bits: rank bits (2-A)
        - Last 4 bits: suit bits (spades=1, hearts=2, diamonds=4, clubs=8)
        
        Args:
            cards: List of Card objects or None
            
        Returns:
            numpy array: Binary vector of length 17 (13 rank bits + 4 suit bits)
        """
        encoded = np.zeros(17)
        
        if not cards:
            return encoded
            
        for card in cards:
            # Encode rank (0-12 for 2-A)
            encoded[card.rank] = 1
            
            # Encode suit (last 4 positions)
            # Convert suit bits to position (1->0, 2->1, 4->2, 8->3)
            suit_position = {1: 13, 2: 14, 4: 15, 8: 16}
            encoded[suit_position[card.suit]] = 1
            
        return encoded
        
    def _encode_position(self, player_id):
        """
        Encode player position relative to button:
        - One-hot encoding of position (Early, Middle, Late, Blinds)
        """
        positions = np.zeros(4)  # [Early, Middle, Late, Blinds]
        
        if player_id == self.game.sb_loc or player_id == self.game.bb_loc:
            positions[3] = 1  # Blinds
        else:
            # Calculate relative position to button
            distance = (player_id - self.game.btn_loc) % self.num_players
            if distance <= self.num_players // 3:
                positions[2] = 1  # Late
            elif distance <= 2 * self.num_players // 3:
                positions[1] = 1  # Middle
            else:
                positions[0] = 1  # Early
                
        return positions
        
    def _encode_betting(self):
        """
        Encode betting information:
        - Pot odds
        - Current bets
        - Number of raises
        """
        current_player = self.game.current_player
        # Get total pot amount from all pots
        pot_total = sum(pot.get_total_amount() for pot in self.game.pots)
        
        # Calculate pot odds (ensure no division by zero)
        chips_to_call = self.game.chips_to_call(current_player)
        pot_odds = np.array([chips_to_call / (pot_total + chips_to_call)]) if pot_total > 0 and chips_to_call > 0 else np.array([0])
        
        # Current bets normalized by big blind (use player_bet_amount for total bet across all pots)
        current_bets = np.array([
            self.game.player_bet_amount(i) / self.big_blind 
            for i in range(self.num_players)
        ])
        
        # Count raises in current round (ensure hand_history exists for current phase)
        num_raises = 0
        if (self.game.hand_history and 
            self.game.hand_phase in self.game.hand_history and 
            self.game.hand_history[self.game.hand_phase] and
            hasattr(self.game.hand_history[self.game.hand_phase], 'actions')):
            num_raises = len([
                action for action in self.game.hand_history[self.game.hand_phase].actions 
                if action.action_type == ActionType.RAISE
            ])
        
        return np.concatenate([
            pot_odds,                # float[1]: pot odds
            [num_raises / 4.0],      # float[1]: normalized number of raises
            current_bets            # float[num_players]: normalized bet amounts
        ])
        
    def _encode_state(self):
        """
        Encode complete game state:
        1. Hand cards (17 dims)
        2. Board cards (17 dims)
        3. Position (4 dims)
        4. Betting info (num_players + 2 dims)
        5. Stack sizes (num_players dims)
        """
        current_player = self.game.current_player
        
        # Encode cards (ensure proper handling of None)
        hand = self.encode_cards(self.game.get_hand(current_player) or [])
        board = self.encode_cards(self.game.board or [])
        
        # Encode position
        position = self._encode_position(current_player)
        
        # Encode betting information
        betting = self._encode_betting()
        
        # Encode stack sizes as ratios of initial stack (ensure no division by zero)
        stacks = np.array([
            max(0.0, self.game.players[i].chips / self.buyin) 
            for i in range(self.num_players)
        ])
        
        # Verify dimensions
        assert hand.shape == (17,), f"Hand shape {hand.shape} != (17,)"
        assert board.shape == (17,), f"Board shape {board.shape} != (17,)"
        assert position.shape == (4,), f"Position shape {position.shape} != (4,)"
        assert betting.shape == (self.num_players + 2,), f"Betting shape {betting.shape} != ({self.num_players + 2},)"
        assert stacks.shape == (self.num_players,), f"Stacks shape {stacks.shape} != ({self.num_players},)"
        
        return np.concatenate([
            hand,      # float[17]: card encodings
            board,     # float[17]: card encodings
            position,  # float[4]: position one-hot
            betting,   # float[num_players + 2]: betting info
            stacks     # float[num_players]: stack sizes
        ])
        
    def _get_valid_actions(self):
        """Get mask of valid actions"""
        moves = self.game.get_available_moves()
        chips_to_call = self.game.chips_to_call(self.game.current_player)
        
        return {
            ActionType.FOLD: True,
            ActionType.CHECK: ActionType.CHECK in moves.action_types or chips_to_call == 0,
            ActionType.CALL: ActionType.CALL in moves.action_types and chips_to_call > 0,
            ActionType.RAISE: ActionType.RAISE in moves.action_types,
            ActionType.ALL_IN: True
        }
        
    def calculate_reward(self, done):
        """
        Calculate reward with multiple components:
        1. Normalized profit/loss (if hand is done)
        2. Potential-based shaping reward (each action)
        3. Action-based immediate rewards
        4. Risk-adjusted returns
        
        Returns:
            float: Combined reward signal
        """
        reward = 0
        current_player = self.game.current_player
        
        # 1. Terminal reward (normalized by big blind)
        if done:
            profit = self.game.players[current_player].chips - self.game.buyin
            reward += profit / self.game.big_blind
            
        # 2. Potential-based shaping
        if self.game.is_hand_running():
            hand = self.game.get_hand(current_player)
            new_potential = calculate_potential(hand, self.game.board, self.num_players)
            if self._last_potential is not None:
                reward += self.gamma * new_potential - self._last_potential
            self._last_potential = new_potential
            
        # 3. Action-based immediate rewards
        if self._action[0] is not None:  # If an action was taken
            action_type, bet_amount = self._action
            
            # Reward for good folds
            if action_type == ActionType.FOLD:
                pot_odds = self.game.chips_to_call(current_player) / sum(pot.get_total_amount() for pot in self.game.pots)
                hand_strength = calculate_hand_strength(self.game.get_hand(current_player), self.game.board)
                if hand_strength < pot_odds:
                    reward += 0.1  # Small reward for good fold
                    
            # Reward for value betting
            elif action_type in (ActionType.RAISE, ActionType.ALL_IN):
                hand_strength = calculate_hand_strength(self.game.get_hand(current_player), self.game.board)
                if hand_strength > 0.8:  # Strong hand
                    reward += 0.2  # Reward for value betting
                    
        # 4. Risk adjustment
        if done:
            # Penalize high variance strategies
            std_dev = np.std([self.game.players[i].chips - self.game.buyin for i in range(self.num_players)])
            risk_penalty = -0.1 * (std_dev / self.game.big_blind)
            reward += risk_penalty
            
        return reward
        
    def step(self, action_tuple):
        """
        Execute action and return next state, reward, done
        """
        # Unpack action tuple
        action_type, total_amount = action_tuple
        
        # Add check for number of active players
        active_players = sum(1 for _ in self.game.in_pot_iter())
        if active_players <= 1:
            done = True
            next_state = self._encode_state()
            reward = self.calculate_reward(done)
            return next_state, reward, done, {}

        # Store action for reward calculation
        self._action = (action_type, total_amount)
        
        try:
            if not self.game.is_hand_running():
                print(f"Warning: Hand not running before action. Game state: {self.game.game_state}")
                print(f"Active players: {active_players}")
                print(f"Current action: {action_type}, amount: {total_amount}")
                
                # Try to recover the game state
                if self.game.game_state == GameState.STOPPED:
                    print("Recreating game...")
                    self.game = TexasHoldEm(
                        buyin=self.buyin,
                        big_blind=self.big_blind,
                        small_blind=self.small_blind,
                        max_players=self.num_players
                    )
                
                print("Starting new hand...")
                self.game.start_hand()
                
                # Verify hand started successfully
                if not self.game.is_hand_running():
                    raise RuntimeError("Failed to start new hand")
            
            # Add validation before taking action
            current_player = self.game.current_player
            chips_to_call = self.game.chips_to_call(current_player)
            
            # If player has already called (chips_to_call == 0) and tries to CALL,
            # convert it to CHECK
            if action_type == ActionType.CALL and chips_to_call == 0:
                action_type = ActionType.CHECK
                print(f"Converting CALL to CHECK for player {current_player} (no chips to call)")
            
            # Take the action
            self.game.take_action(action_type, total=total_amount)
            
        except Exception as e:
            print(f"Error during step: {str(e)}")
            print(f"Game state: {self.game.game_state}")
            print(f"Hand running: {self.game.is_hand_running()}")
            print(f"Active players: {active_players}")
            print(f"Current player: {self.game.current_player}")
            print(f"Chips to call: {self.game.chips_to_call(self.game.current_player)}")
            print(f"Available moves: {self.game.get_available_moves()}")
            raise
        
        next_state = self._encode_state()
        done = not self.game.is_hand_running()
        reward = self.calculate_reward(done)
        return next_state, reward, done, {}
        
    def reset(self):
        """Reset environment and return initial state"""
        self._action = (None, None)
        self._last_potential = None
        
        # More comprehensive reset logic
        try:
            active_players = sum(1 for _ in self.game.in_pot_iter())
            if self.game.game_state == GameState.STOPPED or active_players <= 1:
                print("Recreating game during reset...")
                self.game = TexasHoldEm(
                    buyin=self.buyin,
                    big_blind=self.big_blind,
                    small_blind=self.small_blind,
                    max_players=self.num_players
                )
            
            if not self.game.is_hand_running():
                print("Starting new hand during reset...")
                self.game.start_hand()
                
                # Verify hand started successfully
                if not self.game.is_hand_running():
                    raise RuntimeError("Failed to start new hand during reset")
                
        except Exception as e:
            print(f"Error during reset: {str(e)}")
            print(f"Game state: {self.game.game_state}")
            print(f"Active players: {sum(1 for _ in self.game.in_pot_iter())}")
            raise
        
        return self._encode_state()