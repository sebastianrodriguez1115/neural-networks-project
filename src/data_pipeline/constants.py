"""Shared constants for the data_pipeline package."""

KMER_SIZES = [3, 4, 5]
KMER_DIMS = [4**k for k in KMER_SIZES]  # [64, 256, 1024]
TOTAL_KMER_DIM = sum(KMER_DIMS)  # 1344
BIGRU_PAD_DIM = 1024
MIN_GENOME_LENGTH = 500_000
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
