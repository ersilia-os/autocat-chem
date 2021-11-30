TRAIN_TEST_SPLIT = 0.2
TRY_GPU = True
MAX_TREE_DEPTH = 12  # Constrained by GPU memory (for single task)
BATCH_SIZE = 100000
BATCH_ITERATIONS = 3
HIST_BINS = 100  # Weights
CLIP_BOUNDS = [-10, 10]  # Scaler clipping
NBITS = 2048  # Morgan fps
RADIUS = 3