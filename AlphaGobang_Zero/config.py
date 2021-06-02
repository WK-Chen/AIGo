# length of the board size
BOARD_WIDTH = 9
# n in a row to win
N_IN_ROW = 5

# Train parameters
SELF_PLAY_PARALLEL = 4
PLAY_BATCH_SIZE = SELF_PLAY_PARALLEL
MCTS_PARALLEL = 1
LEARN_RATE = 2e-3
LR_MULTIPLIER = 1.0
TEMP = 1.0
N_PLAYOUT = 400
C_PUCT = 5
BUFFER_SIZE = 10000
BATCH_SIZE = 1024
EPOCHS = 5
KL_TARG = 2e-2
CHECK_FREQ = PLAY_BATCH_SIZE * 50
GAME_BATCH_NUM = 4000
BEST_WIN_RATIO = 0.2
PURE_MCTS_PLAYOUT_NUM = 1000