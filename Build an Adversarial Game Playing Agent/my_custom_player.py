from sample_players import DataPlayer
from isolation import isolation
import random, time, math, copy

"""
MCTreeSearchNode Class implementation

Below are given implementations for:
update_node
expand_node
optimal_child_node
backpropagate
playout
is_terminal_node
is_explored

"""
class MCTreeSearchNode():

    def __init__(self, state: isolation, action = None, parent_node = None):
        
        #init children nodes
        self.children_nodes = []
        #set parent node, init to None if root node
        self.parent_node = parent_node
        
        #init state
        self.state  = state
        #init action, none for root node
        self.action = action 

        #init all state actions as unexplored actions
        self.unexplored_actions = self.state.actions()

        #init reward to 0.0
        self.reward        = 0.0
        #init number of visits to 1.0
        self.num_of_visits = 1.0

    """
    Update the node with reward used in backpropagation function
    """
    def update_node(self, reward):
        #increment current reward with new value
        self.reward        += reward
        #increment number of visits for this node
        self.num_of_visits += 1

    """
    Expand current node with random chosen action from set of unexplored actions
    """
    def expand_node(self):
        #make a random choice from set of unexplored actions
        expand_action = random.choice(self.unexplored_actions)
        #calculate new state based on chosen action
        expand_state  = self.state.result(expand_action)
        #generate new expanded child node based on given state and action
        expanded_node = MCTreeSearchNode(parent_node = self, state = expand_state, action = expand_action)

        #remove random chosen action from set of unexplored actions 
        self.unexplored_actions.remove(expand_action)
        #append expanded node to children nodes
        self.children_nodes.append(expanded_node)

        return expanded_node
    
    """
    Calculate optimal child node by scoring all child nodes using function
    score = exploitation_part(child.reward/child.num_of_visits) + exploration_part(factor * SQRT(2.0 * LOG(parent.num_of_visits/child.num_of_visits))
    The child that gets the maximum score will be selected as optimal node.
    """
    def optimal_child_node(self, factor):
        #calculate score for each child node
        scored_child_nodes = [(child_node.reward / (child_node.num_of_visits)) + factor * math.sqrt(2.0 * math.log(self.num_of_visits) / (child_node.num_of_visits)) for child_node in self.children_nodes]
        #return child node with maximum score
        return self.children_nodes[scored_child_nodes.index(max(scored_child_nodes))]

    """
    Backpropagation will update each node with reward calculated.
    It starts from current node backing up to root node.
    """
    def backpropagate(self, reward):
        #update node with calculated reward
        self.update_node(reward)
        #make a negativa value from curent reward and backpropagate to previous visited nodes using recursion
        reward *= -1
        if self.parent_node:
            self.parent_node.backpropagate(reward)
   
    """
    This function makes a random search of actions from current state and makes decision 
    If current_state player wins over init_state player then (reward=1)
    """
    def playout(self):
        
        #select init state
        init_state    = copy.deepcopy(self.state)
        #select current state
        current_state = copy.deepcopy(self.state)
        
        #make random selection of curent state actions and update the current state while not terminal
        while not current_state.terminal_test():
            action = random.choice(current_state.actions())
            current_state = current_state.result(action)
        
        #return 1 if current_state player wins over init_state player
        return -1 if current_state._has_liberties(init_state.player()) else 1

    """
    Check if state passes terminal test
    """
    def is_terminal_node(self):       
        return self.state.terminal_test()

    """
    Check if there are no unexplored actions remaining
    """
    def is_explored(self):
        return len(self.unexplored_actions) == 0

class MCTreeSearch():

    def __init__(self, node: MCTreeSearchNode):
        self.root_node = node

    """
    Return expanded node if node is not explored.
    Otherwise, find optimal child node with applied factor 
    """
    def tree_policy(self, factor):
        #start from root node
        node = self.root_node
        #iterate while node is not terminal
        while not node.is_terminal_node():
            #expand the node if not explored
            if not node.is_explored():
                return node.expand_node()
            #find the optimal child node with applied factor
            node = node.optimal_child_node(factor)
        return node

    """
    Function that executes Monte Carlo Tree Search Simulation
    The search follows MCTS steps:
    selection
    expansion
    simulation (playout)
    backpropagation
    
    Each search is limited to a maximum time (ms) in order to perform all the steps
    
    """
    def run_search(self, factor, max_time):  
        #define start time
        start_time = time.time()
        #run loop until time duration reaches max allowed time
        while (time.time() - start_time) <= max_time/1000.:
            #run selection and expansion step
            node   = self.tree_policy(factor)
            #run playout step and determine reward
            reward = node.playout()
            #backpropagate reward to parent node recursively
            node.backpropagate(reward)

        #find optimal node by searching the optimal node from the root of the tree
        optimal_node   = self.root_node.optimal_child_node(factor)
        #select optimal action
        optimal_action = optimal_node.action

        return optimal_action

"""
Code combined from alpha beta pruning lessons and sample players (minimax)
"""
class Alpha_Beta_Search():

    def __init__(self, state: isolation, player_id, depth = 5):
        self.state = state
        self.player_id = player_id
        self.depth = depth

    def min_value(self, state, alpha, beta, player_id, depth):
        if state.terminal_test():
            return state.utility(player_id)
        if depth <= 0:
            return self.score()
        value = float("inf")
        for action in state.actions():
            value = min(value, self.max_value(state.result(action), alpha, beta, player_id, depth-1))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def max_value(self, state, alpha, beta, player_id, depth):
        if state.terminal_test():
            return state.utility(player_id)
        if depth <= 0: 
            return self.score()
        value = float("-inf")
        for action in state.actions():
            value = max(value, self.min_value(state.result(action), alpha, beta, player_id, depth-1))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value
    
    def run_search(self):
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for action in self.state.actions():
            value = self.min_value(self.state.result(action), alpha, beta, self.player_id, self.depth-1)
            alpha = max(alpha, value)
            if value >= best_score:
                best_score = value
                best_move = action
        return best_move


    def score(self):
        own_loc = self.state.locs[self.player_id]
        opp_loc = self.state.locs[1 - self.player_id]
        own_liberties = self.state.liberties(own_loc)
        opp_liberties = self.state.liberties(opp_loc)

        return len(own_liberties) - len(opp_liberties)
        
        
class CustomPlayer_Alpha_Beta_Itt(DataPlayer):
    """ Implement customized agent to play knight's Isolation """
    
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least
        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired.
        See RandomPlayer and GreedyPlayer in sample_players for more examples.
        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        if state.terminal_test() or state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            #perform itterative deepening with max depth limit set to 5
            depth_limit = 5
            for depth in range(1, depth_limit + 1):
                best_move = Alpha_Beta_Search(state, self.player_id, depth).run_search()
            self.queue.put(best_move)

class CustomPlayer_MCTS(DataPlayer):
    """
    Implement an agent to play knight's Isolation with Monte Carlo Tree Search
    """

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least
        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired.
        See RandomPlayer and GreedyPlayer in sample_players for more examples.
        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        if state.terminal_test() or state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            result = MCTreeSearch(MCTreeSearchNode(state)).run_search(factor = 1.0, max_time = 100)
      
            if result:
                self.queue.put(result)
            elif state.actions():
                 self.queue.put(random.choice(state.actions()))
            else:
                 self.queue.put(None)

CustomPlayer = CustomPlayer_MCTS