"""Constantes compartidas del paquete data_pipeline."""

KMER_SIZES = [3, 4, 5]
KMER_DIMS = [4**k for k in KMER_SIZES]  # [64, 256, 1024]
TOTAL_KMER_DIM = sum(KMER_DIMS)  # 1344
BIGRU_PAD_DIM = 1024
ANTIBIOTIC_EMBEDDING_DIM = 49  # min(50, (96 // 2) + 1) para 96 antibióticos
# Offsets para segmentar k=3,4,5 [Lugo21]
KMER_OFFSETS = [sum(KMER_DIMS[:i]) for i in range(len(KMER_DIMS) + 1)]
MIN_GENOME_LENGTH = 500_000
MIN_RECORDS_PER_ANTIBIOTIC = 20
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}

# Tokenización de k-meros [Mikolov13; Compeau14]
TOKEN_KMER_K = 4                 # tamaño del k-mero para tokenización
TOKEN_VOCAB_SIZE = 4 ** TOKEN_KMER_K  # 256 tokens válidos
TOKEN_PAD_ID = TOKEN_VOCAB_SIZE  # 256 — token de padding (fuera del vocab)
TOKEN_MAX_LEN = 4096             # longitud máxima de la secuencia
TOKEN_EMBED_DIM = 64             # dimensión del embedding de k-meros
