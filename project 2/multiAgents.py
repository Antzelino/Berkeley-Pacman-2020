# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"

        movesScoresPairs = sorted(zip(scores,legalMoves))
        bestScore = max(movesScoresPairs)[0]
        bestMoves = [x[1] for x in movesScoresPairs if x[0] == bestScore] # only the moves with best score

        if Directions.STOP in bestMoves and len(bestMoves) > 1: # if we have other choices than STOP that are equally good, choose among them
            bestMoves.remove(Directions.STOP)

        return random.choice(bestMoves)

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"

        newGhostPositions = successorGameState.getGhostPositions()
        evaluationPoints = currentGameState.getScore()

        # if too close to ghost, it's not a good state. -50 for very close, -100 for game over
        for i,newGhostPos in enumerate(newGhostPositions):
            if min(newScaredTimes) == 0: # if there is at least 1 ghost not scared, pacman avoids them (not worth the code effort to eat scared ghosts)
                if manhattanDistance(newPos,newGhostPos) < 2:
                    evaluationPoints -= 50
                if manhattanDistance(newPos,newGhostPos) < 1:
                    evaluationPoints -= 100

        # if this state eats a food, +20 evaluation points
        if newFood.count() < currentGameState.getFood().count():
            evaluationPoints += 20

        position = currentGameState.getPacmanPosition()
        closestFood = (-1,-1)

        closestFoodDistance = float('inf')
        for food in currentGameState.getFood().asList():
            thisDistance = manhattanDistance(position, food)
            if thisDistance < closestFoodDistance:
                closestFoodDistance = thisDistance
                closestFood = food

        # if pacman goes towards the nearest food, +10 evaluation points
        if manhattanDistance(newPos, closestFood) < closestFoodDistance:
            evaluationPoints += 10

        # if pacman keeps the same direction (instead of moving randomly around), +10 evaluation points
        currentDirection = currentGameState.getPacmanState().getDirection()
        if currentDirection != Directions.STOP and currentDirection == action:
            evaluationPoints += 10

        return evaluationPoints

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimaxMax(gameState, index, depth, returnValuesActions=False):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            value = -float('inf')
            valuesActionsPairs = []
            for action in gameState.getLegalActions(index):
                successorValue = minimaxMin(gameState.generateSuccessor(index, action), index+1, depth)
                value = max(value, successorValue)
                if returnValuesActions:
                    valuesActionsPairs.append((successorValue,action))

            if returnValuesActions:
                return valuesActionsPairs

            return value

        def minimaxMin(gameState, index, depth):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            val = float('inf')
            for action in gameState.getLegalActions(index):
                if index == gameState.getNumAgents()-1: # successor is max node (pacman)
                    val = min([val, minimaxMax(gameState.generateSuccessor(index, action), 0, depth-1)])
                else: # successor is another min node (another ghost)
                    val = min([val, minimaxMin(gameState.generateSuccessor(index, action), index+1, depth)])
            
            return val

        def minimax(gameState):
            valuesActionsPairs = minimaxMax(gameState, self.index, self.depth, True)
            bestValue = max(valuesActionsPairs)[0] # 0 is index of values, 1 is index of actions
            bestActions = [x[1] for x in valuesActionsPairs if x[0] == bestValue] # bestActions: only the legal actions with max value

            if hasattr(gameState, "getPacmanState"):
                pacmanDirection = gameState.getPacmanState().getDirection()
                if pacmanDirection in bestActions:
                    bestActions.append(pacmanDirection)
                    bestActions.append(pacmanDirection) # triple the chances for pacman to continue same direction
                bestActions = bestActions+bestActions
                if Directions.REVERSE[pacmanDirection] in bestActions:
                    # we already duplicated all the elements in the list, so removing this one would still leave the other in the list
                    bestActions.remove(Directions.REVERSE[pacmanDirection]) # halve the chances that pacman reverses direction he was going
                if Directions.STOP in bestActions: # same thing for stopping
                    bestActions.remove(Directions.STOP)

            return random.choice(bestActions)

        minimaxAction = minimax(gameState)

        return minimaxAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBetaMax(gameState, depth, index, a, b, returnValuesActions=False):
            '''
            When returnValuesActions is True, a list of (value,action) tuples is returned,
            otherwise -as expected- a value is returned. (useful for the call on the root node)
            '''
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            value = -float('inf')
            abValuesActions = [] # the list of (value,action) tuples that's returned if this is a root node
            for action in gameState.getLegalActions(index):
                successor = gameState.generateSuccessor(index, action)
                successorValue = alphaBetaMin(successor, depth, index+1, a, b)
                abValuesActions.append((successorValue,action))

                value = max(value, successorValue)

                a = max(a,value)
                if a > b:
                    break

            if returnValuesActions:
                return abValuesActions

            return value

        def alphaBetaMin(gameState, depth, index, a, b):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            value = float('inf')
            for action in gameState.getLegalActions(index):
                successor = gameState.generateSuccessor(index, action)
                
                if index == gameState.getNumAgents()-1: # successor is max node (pacman)
                    value = min(value, alphaBetaMax(successor, depth-1, 0, a, b))
                else:
                    value = min(value, alphaBetaMin(successor, depth, index+1, a, b))

                b = min(value, b)
                if a > b:
                    break

            return value

        def alphaBetaPruning(gameState):
            valuesActionsPairs = alphaBetaMax(gameState, self.depth, self.index, -float('inf'), float('inf'), True)
            bestValue = max(valuesActionsPairs)[0] # 0 is index of values, 1 is index of actions
            bestActions = [x[1] for x in valuesActionsPairs if x[0] == bestValue] # bestActions: only the legal actions with max value
            
            if hasattr(gameState, "getPacmanState"):
                pacmanDirection = gameState.getPacmanState().getDirection()
                if pacmanDirection in bestActions:
                    bestActions.append(pacmanDirection)
                    bestActions.append(pacmanDirection) # triple the chances for pacman to continue same direction
                bestActions = bestActions+bestActions
                if Directions.REVERSE[pacmanDirection] in bestActions:
                    # we already duplicated all the elements in the list, so removing this one would still leave the other in the list
                    bestActions.remove(Directions.REVERSE[pacmanDirection]) # halve the chances that pacman reverses direction he was going
                if Directions.STOP in bestActions: # same thing for stopping
                    bestActions.remove(Directions.STOP)

            return random.choice(bestActions)

        alphabetaAction = alphaBetaPruning(gameState)

        return alphabetaAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimaxMax(gameState, index, depth, returnValuesActions=False): # exactly like minimax maxValue
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            value = -float('inf')
            valuesActionsPairs = []
            for action in gameState.getLegalActions(index):
                successorValue = expectimaxRandom(gameState.generateSuccessor(index, action), index+1, depth)
                value = max(value, successorValue)
                if returnValuesActions:
                    valuesActionsPairs.append((successorValue,action))

            if returnValuesActions:
                return valuesActionsPairs

            return value

        def expectimaxRandom(gameState, index, depth):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            value = 0.0
            legalActions = gameState.getLegalActions(index)
            numActions = len(legalActions)
            probability = 1.0/numActions # every legal move has the same probability

            for action in legalActions:
                if index == gameState.getNumAgents()-1: # successor is max node (pacman)
                    value += probability * expectimaxMax(gameState.generateSuccessor(index, action), 0, depth-1)
                else: # successor is another chance node (another ghost)
                    value += probability * expectimaxRandom(gameState.generateSuccessor(index, action), index+1, depth)
            
            return value

        def expectimax(gameState): # returns the expectimax action
            valuesActionsPairs = expectimaxMax(gameState, self.index, self.depth, True)
            bestValue = max(valuesActionsPairs)[0] # 0 is index of values, 1 is index of actions
            bestActions = [x[1] for x in valuesActionsPairs if x[0] == bestValue] # bestActions: only the legal actions with max value
            return random.choice(bestActions)

        expectimaxAction = expectimax(gameState)

        return expectimaxAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This is quite a simple evaluation function, which could definitely
      be improved by a lot (see list of available functions below), but it's also effective!
      The evaluation function takes into consideration the score of the game currently,
      the average distance of pacman to all foods and the distance to the nearest food.
      (Or actually the reciprocal of each distance). The result is a linear combination
      of these values.
    """
    "*** YOUR CODE HERE ***"

    # gameState:   Functions of the gameState class that I might want to use
    #   getLegalActions(agentIndex)               # ACTIONS
    #   generateSuccessor(agentIndex, action)     # SUCCESSOR
    #   getLegalPacmanActions()
    #   generatePacmanSuccessor(action)
    #   getPacmanState()                          # STATE
    #   getPacmanPosition()                       # POSITION
    #   getGhostStates()
    #   getGhostPositions()
    #   getGhostState(agentIndex)
    #   getGhostPosition(agentIndex)
    #   getNumAgents()                            # NUM AGENTS
    #   getScore()                                # SCORE
    #   getCapsules() # list of (x,y) of remaining capsules                         # CAPSULES
    #   getNumFood()                                                                # FOOD
    #   getFood() # grid of boolean food. example: if getFood()[x][y] == True: ...
    #   hasFood(x,y)
    #   getWalls()                                                                  # WALLS
    #   hasWall(x,y)
    #   isLose() , isWin()                                                          # WIN / LOSE

    pacmanPosition = currentGameState.getPacmanPosition()
    numFood = currentGameState.getNumFood()

    avgDistanceFromFoods = 1e-8 # avoid division by zero
    closestFoodDistance = float('inf')
    for foodPosition in currentGameState.getFood().asList():
        thisDistance = manhattanDistance(pacmanPosition,foodPosition)
        avgDistanceFromFoods += float(thisDistance)/float(numFood) # compute average distance
        if thisDistance < closestFoodDistance:
            closestFoodDistance = thisDistance # save distance to nearest food

    w1 = 0.33
    w2 = 0.33
    w3 = 0.33
    return currentGameState.getScore() * w1 + (1.0/avgDistanceFromFoods) * w2 + (1.0/closestFoodDistance) * w3

# Abbreviation
better = betterEvaluationFunction
