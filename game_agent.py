"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

from abc import ABC, abstractmethod

NO_MOVE = (-1, -1)
PHI = 1.618033988749895


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : isolation.Board
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : IsolationPlayer
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`).

    Returns
    -------
    value: float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    elif game.is_winner(player):
        return float("inf")

    else:
        agent_score = float(len(game.get_legal_moves(player)))
        enemy_score = float(len(game.get_legal_moves(game.get_opponent(player))))
        return agent_score - 2*enemy_score


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : isolation.Board
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : IsolationPlayer
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`).

    Returns
    -------
    value: float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    elif game.is_winner(player):
        return float("inf")

    else:
        # Aggressive strategy
        agent_score = float(len(game.get_legal_moves(player)))
        enemy_score = float(len(game.get_legal_moves(game.get_opponent(player))))
        return agent_score - PHI*enemy_score


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`).

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    elif game.is_winner(player):
        return float("inf")

    else:
        # Dummy strategy
        agent_score = float(len(game.get_legal_moves(player)))
        enemy_score = float(len(game.get_legal_moves(game.get_opponent(player))))
        return agent_score - enemy_score


class IsolationPlayer(ABC):
    """Base class for minimax and alpha-beta agents.

    Attributes
    ----------
    search_depth : int
        Depth limit of player agent search

    score : callable
        Function to calculate score over remaining legal moves

    time_left : int
        Remaining time to player search for a solution

    TIMER_THRESHOLD: float
        Time threshold when the player should stop
    """

    @staticmethod
    def is_terminal(board):
        """
        Parameters
        ----------
        board : isolation.Board
            Current state of isolation game board

        Returns
        -------
        state: bool
            False if there are not further legal moves, True otherwise
        """
        return not bool(board.get_legal_moves())

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        """IsolationPlayer abstract constructor

        Abstract construct ensuring class is never constructed/tested directly

        Parameters
        ----------
        search_depth : int, optional
            A strictly positive integer (i.e., 1, 2, 3,...) for the number of
            layers in the game tree to explore for fixed-depth search. (i.e., a
            depth of one (1) would only explore the immediate successors of the
            current state.)

        score_fn : callable, optional
            A function to use for heuristic evaluation of game states.

        timeout : float, optional
            Time remaining (in milliseconds) when search is aborted. Should be a
            positive value large enough to allow the function to return before the
            timer expires.
        """
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    @abstractmethod
    def get_move(self, game, time_left):
        """Get next best legal move before timeout expires

        Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Parameters
        ----------
        game: isolation.Board
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left: callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        move : (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.

        """
        raise NotImplementedError


class MinimaxPlayer(IsolationPlayer):
    """Agent powered by depth-limited minimax search

    Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Get next best legal move before timeout expires

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : isolation.Board
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        move : (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.

        """
        self.time_left = time_left

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            # Handle any actions required after timeout as needed
            return NO_MOVE

    def max_value(self, board, depth=100):
        """
        Parameters
        ----------
        board : isolation.Board
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        depth : int
            Depth search limit

        Returns
        -------
            Maximum value over all legal child nodes, otherwise float('-inf')
        """

        if IsolationPlayer.is_terminal(board) or depth <= 0:
            return self.score(board, self)

        local_max = float("-inf")
        for move in board.get_legal_moves():
            local_max = max(local_max, self.min_value(board.forecast_move(move), depth - 1))

        return local_max

    def min_value(self, board, depth=100):
        """
        Parameters
        ----------
        board : isolation.Board
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        depth : int
            Depth search limit

        Returns
        -------
            Minimum value over all legal child nodes, otherwise float('inf')
        """

        if IsolationPlayer.is_terminal(board) or depth <= 0:
            return self.score(board, self)

        local_min = float("inf")
        for move in board.get_legal_moves():
            local_min = max(local_min, self.min_value(board.forecast_move(move), depth - 1))

        return local_min

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        move : (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if IsolationPlayer.is_terminal(game):
            return NO_MOVE

        best_score = float("-inf")
        best_move = None

        for move in game.get_legal_moves():
            score = self.min_value(game.forecast_move(move), depth - 1)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Agent powered by iterative deepening and alpha-beta pruning search

    Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Get next best legal move before timeout expires

        Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Parameters
        ----------
        game : isolation.Board
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        move : (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.

        """
        self.time_left = time_left

        if IsolationPlayer.is_terminal(game):
            return NO_MOVE

        return self.iterative_deepening(game)

    def alpha_beta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        move : (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        _, best_move = self.max_value(game, depth, alpha, beta)

        return best_move

    def iterative_deepening(self, board):
        best_move = random.choice(board.get_legal_moves())
        depth = 1
        try:
            while True:
                best_move = self.alpha_beta(board, depth=depth)
                depth += 1
        except SearchTimeout:
            pass
        finally:
            return best_move

    def max_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if IsolationPlayer.is_terminal(game) or depth <= 0:
            return self.score(game, self), NO_MOVE

        best_move = random.choice(game.get_legal_moves())
        best_score = float('-inf')

        for move in game.get_legal_moves():
            score, _ = self.min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if best_score < score:
                best_score = score
                best_move = move

            if best_score >= beta:
                return best_score, best_move

            alpha = max(best_score, alpha)

        return best_score, best_move

    def min_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if IsolationPlayer.is_terminal(game) or depth <= 0:
            return self.score(game, self), NO_MOVE

        best_move = random.choice(game.get_legal_moves())
        best_score = float('inf')
        legal_moves = game.get_legal_moves()

        for move in legal_moves:
            score, _ = self.max_value(game.forecast_move(move), depth - 1, alpha, beta)

            if best_score > score:
                best_score = score
                best_move = move

            if best_score <= alpha:
                return best_score, best_move

            beta = min(best_score, beta)

        return best_score, best_move
