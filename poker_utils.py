import numpy as np
from typing import List, Optional
from texasholdem.card.card import Card
from texasholdem.evaluator import evaluator
import random

def calculate_hand_strength(hand: List[Card], board: List[Card]) -> float:
    """
    Calculate relative hand strength using Monte Carlo simulation
    
    Args:
        hand: List of hole cards
        board: List of community cards
        
    Returns:
        float: Estimated hand strength [0,1]
    """
    if not hand:
        return 0.0
        
    # If board is complete, use exact evaluation
    if len(board) == 5:
        hand_value = evaluator.evaluate(hand, board)
        # Convert to normalized strength (lower rank is better in evaluator)
        return (7462 - hand_value) / 7462
        
    # Otherwise use Monte Carlo simulation
    wins = 0
    trials = 100
    
    for _ in range(trials):
        # Create copy of remaining deck
        all_cards = []
        suits = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        for suit in suits:
            for rank in ranks:
                card = Card(rank + suit)  # Assuming Card constructor takes string like "Ah" for Ace of hearts
                if card not in hand and card not in board:
                    all_cards.append(card)
        remaining_cards = all_cards
        
        # Complete the board randomly
        sim_board = board.copy()
        while len(sim_board) < 5:
            card = random.choice(remaining_cards)
            sim_board.append(card)
            remaining_cards.remove(card)
            
        # Evaluate hand
        hand_value = evaluator.evaluate(hand, sim_board)
        
        # Compare against random opponent hand
        opp_hand = random.sample(remaining_cards, 2)
        opp_value = evaluator.evaluate(opp_hand, sim_board)
        
        if hand_value < opp_value:  # Lower is better in evaluator
            wins += 1
            
    return wins / trials

def calculate_potential(hand: List[Card], board: List[Card], num_players: int) -> float:
    """
    Calculate hand potential (probability of winning) using Monte Carlo simulation
    
    Args:
        hand: List of hole cards
        board: List of community cards
        num_players: Number of opponents
        
    Returns:
        float: Estimated winning probability [0,1]
    """
    if not hand:
        return 0.0
        
    wins = 0
    trials = 100
    
    for _ in range(trials):
        # Create copy of remaining deck
        all_cards = []
        suits = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        for suit in suits:
            for rank in ranks:
                card = Card(rank + suit)  # Assuming Card constructor takes string like "Ah" for Ace of hearts
                if card not in hand and card not in board:
                    all_cards.append(card)
        remaining_cards = all_cards
        
        # Complete the board randomly
        sim_board = board.copy()
        while len(sim_board) < 5:
            card = random.choice(remaining_cards)
            sim_board.append(card)
            remaining_cards.remove(card)
            
        # Evaluate our hand
        our_value = evaluator.evaluate(hand, sim_board)
        
        # Compare against all opponents
        hand_wins = True
        for _ in range(num_players - 1):
            opp_hand = random.sample(remaining_cards, 2)
            opp_value = evaluator.evaluate(opp_hand, sim_board)
            if opp_value <= our_value:  # Lower is better in evaluator
                hand_wins = False
                break
                
        if hand_wins:
            wins += 1
            
    return wins / trials 