import torch
import multiprocessing

# CONFIG
# CUDA variable from Torch
CUDA = torch.cuda.is_available()
# Dtype of the tensors depending on CUDA
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")
# MCTS parallel
MCTS_PARALLEL = 4


# GLOBAL

# Size of the Go board
GOBANG_SIZE = 6
# Number of move to end a game
MOVE_LIMIT = GOBANG_SIZE ** 2
# Maximum ratio that can be replaced in the rotation buffer
MAX_REPLACEMENT =1.0
# Number of last states to keep
HISTORY = 1
# Learning rate
LR = 1e-2
# Number of MCTS simulation
MCTS_SIM = 400
# Exploration constant
C_PUCT = 5
# L2 Regularization
L2_REG = 1e-4
# Momentum
MOMENTUM = 0.9
# Activate MCTS
MCTS_FLAG = True
# Epsilon for Dirichlet noise
EPS = 0.25
# Alpha for Dirichlet noise
ALPHA = 0.3
# Batch size for evaluation during MCTS
BATCH_SIZE_EVAL = 4
# Number of moves before changing temperature to stop exploration
TEMPERATURE_MOVE = 3


# TRAINING
EPOCH = 5
# Number of simulated game generated per round
SIMULATION_PER_ROUND = 100
# Number of moves to consider when creating the batch
MOVES = 2048
# Number of mini-batch before evaluation during training
BATCH_SIZE = 64
# Number of channels of the output feature maps
OUTPLANES_MAP = 10
# Shape of the input state
INPLANES = (HISTORY + 1) * 2 + 1
# Probabilities for all moves
OUTPLANES = (GOBANG_SIZE ** 2)
# Number of residual blocks
BLOCKS = 3
# Optimizer
ADAM = True
# Learning rate annealing factor
LR_DECAY = 0.1
# Learning rate annnealing interval
LR_DECAY_ITE = 1000
# Data augment
ROTATION_NUM = 4
# EVALUATION

# Number of matches against its old version to evaluate the newly trained network
EVAL_MATCHS = 10
# Threshold to keep the new neural net
EVAL_THRESH = 0.55