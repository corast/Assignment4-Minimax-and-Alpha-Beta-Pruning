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
import random
import util

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
		some Directions.X for some X in the set {North, South, West, East, Stop}
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(
			gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(
			len(scores)) if scores[index] == bestScore]
		# Pick randomly among the best
		chosenIndex = random.choice(bestIndices)

		"Add more of your code here if you want to"

		return legalMoves[chosenIndex]

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
		newScaredTimes = [
			ghostState.scaredTimer for ghostState in newGhostStates]

		"*** YOUR CODE HERE ***"
		return successorGameState.getScore()


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

	def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
		self.index = 0  # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)


import sys

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

										gameState.isWin() #True if game is won
										gameState.isLose() #True if game is lost

										self.evaluationFunction(gameState)

										each ghost generate a state
										Depth is number of times we run max

		"""
		def maxPlayer(gameState, depth):
			""" max function for pacman """
			#CHeck if we are at the bottom of the tree(where we get a value), or we have encountered an winning or losing move.
			if (gameState.isWin() or gameState.isLose() or depth == 0):
    			#Return value the value of this move.
				return self.evaluationFunction(gameState)
			#Store a value as - infinite ( large value what we know will be overwritten)
			value = -sys.maxint - 1

			# Check for every action, from legelActions pacman(index 0)
			for action in gameState.getLegalActions(pac_index):
				# Get next state with this action from pacman's index.
				nextState = gameState.generateSuccessor(pac_index, action)
				value = max(value, minPlayer(nextState, depth, pac_index + 1))
			return value

		def minPlayer(gameState, depth, ghost_index):
			""" min function for ghosts """
			if (gameState.isWin() or gameState.isLose() or depth == 0):
				return self.evaluationFunction(gameState)
			value = sys.maxint
			legalActions = gameState.getLegalActions(ghost_index)

			# We need to check if we are at the last ghost.
			if (ghost_index == gameState.getNumAgents() - 1):
    			#Check every action of this ghost.
				for action in legalActions:
					nextState = gameState.generateSuccessor(
						ghost_index, action)
					#recursivly check the next level in the tree (MaxPlayer in this case)
					value = min(value, maxPlayer(nextState, depth - 1))
			else:
				for action in legalActions:
					nextState = gameState.generateSuccessor(
						ghost_index, action)
					#Recursivly check the next level in the tree(Next is a Ghost layer)
					value = min(value, minPlayer(
						nextState, depth, ghost_index + 1))
			return value

		# Generate minmax tree of depth 2.
		pac_index = 0
		#Set bestAction to STOP, because we know this is allwasy a valid move. 
		#Incase we cant find a better move with the miniMax tree, we will just stand still
		bestAction = Directions.STOP
		#Generate all the legal actions for pacman.
		legalActions = gameState.getLegalActions(pac_index)
		#set current score to a very small number, to be overwritten later.
		score = -sys.maxint - 1

		for action in legalActions:
			next_state = gameState.generateSuccessor(pac_index, action)
			# Check which action is the best action with minmax algorithm
			prev_score = score
			#Get a new score from the tree.
			score = max(score, minPlayer(next_state, self.depth, 1))
			#Check if the new score is better than the previous one, 
			#if it is we change the bestAction to our current action itteration.
			if(score > prev_score):
				bestAction = action
		#Return the best Action we got from checking every step in the minimax tree.
		return bestAction

		util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
									Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
			Returns the minimax action using self.depth and self.evaluationFunction
		"""
		def maxPlayer(gameState, a, b, depth):
			""" max function for pacman """
			if (gameState.isWin() or gameState.isLose() or depth == 0):
				return self.evaluationFunction(gameState)
			# Store system.max negative integer
			value = -sys.maxint - 1
			# Check for every action, from legelActions pacman(index 0)
			for action in gameState.getLegalActions(pac_index):
				# Get next state with this action from pacman's index.
				nextState = gameState.generateSuccessor(pac_index, action)
				# Get the max value of the next min layers.
				value = max(value, minPlayer(
					nextState, a, b, pac_index + 1, depth))
				#Check if the value that is returned is smaller than the beta value
				if(value > b):
					#Return value now, because we know it is the best we can get.
					return value
				a = max(a, value)
			return value

		def minPlayer(gameState, a, b, ghost_index, depth):
			""" min function for ghost """
			if (gameState.isWin() or gameState.isLose() or depth == 0):
				return self.evaluationFunction(gameState)
			# Store system.max integer.
			value = sys.maxint
			legalActions = gameState.getLegalActions(ghost_index)
				# We need to check if we are at the last ghost.
			if (ghost_index == gameState.getNumAgents()-1):
				for action in legalActions:
					nextState = gameState.generateSuccessor(ghost_index, action)
					#get value from next dept in the tree. Where next is a max.
					value = min(value, maxPlayer(nextState, a, b, depth-1))
					if(value < a):
    					#Return value now, since we can't get a better value
						return value
					b = min(b,value)
			else:
				#if we are not at the last ghost(min layer), we got n-ghosts min layer
				for action in legalActions:
					nextState = gameState.generateSuccessor(ghost_index, action)
					#Get next value from the next ghost in a min layer.
					value = min(value, minPlayer(nextState,a,b,ghost_index+1,depth))
					if(value < a):
    					#Return value now, since we can't get a better value
						return value
					b = min(b,value)
			return value

		"""
			For the alpha-beta tree, alpha a is a ceiling value, whilst beta b is a floor value. 
			We know we can atleast expect either an alpha value or a beta value, depending on our current layer.
			We alwasy compare the current beta and alpha values with previous layers, if we encounter a value smaller than alpha,
			or greater than beta, we know we don't have to check the rest of that tree.
		"""
		# Generate minmax tree of depth 2.
		pac_index = 0
		bestAction = Directions.STOP
		legalActions = gameState.getLegalActions(pac_index)
		score = -sys.maxint - 1
		#Alpha init value
		a = -sys.maxint - 1
		#Beta init value
		b = sys.maxint
		#itterate tru every action pacman can do. Return the one with best value
		for action in legalActions:
			next_state = gameState.generateSuccessor(pac_index, action)
			# Check which action is the best action with minmax algorithm
			prev_score = score
			#Get score of this action with the minimax alg above.
			score = max(score, minPlayer(
				next_state, a, b, pac_index + 1, self.depth))
			#Check our score from the tree generated from this action.
			#And set new action if it is better than the pevious one.
			if(score > prev_score):
				bestAction = action
			if(score >= b):
				# if score is higher than beta, we can stop. Because know it will never get any lower than beta b.
				return bestAction
			# Update alpha
			a = max(a, score)
		return bestAction

		util.raiseNotDefined()


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
		util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
	"""
									Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
									evaluation function (question 5).

									DESCRIPTION: <write something here so we know what you did>
	"""
	"*** YOUR CODE HERE ***"
	util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
