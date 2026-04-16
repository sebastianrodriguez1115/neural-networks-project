import logging
from pathlib import Path

import numpy
import pandas
from Bio import SeqIO
from sklearn.model_selection import train_test_split

from .constants import (
    BASE_TO_INDEX,
    BIGRU_PAD_DIM,
    HIER_KMER_K,
    HIER_N_SEGMENTS,
    KMER_DIMS,
    KMER_SIZES,
    RANDOM_SEED,
    TOKEN_EMBED_DIM,
    TOKEN_KMER_K,
    TOKEN_MAX_LEN,
    TOKEN_PAD_ID,
    TOKEN_VOCAB_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)


class KmerExtractor:
    """Extrae histogramas de frecuencias de k-meros a partir de un archivo FASTA de genoma."""

    def __init__(self, fasta_path: Path):
        self._fasta_path = Path(fasta_path)
        self._histograms: dict[int, numpy.ndarray] = {}

    def extract(self) -> dict[int, numpy.ndarray]:
        sequences = self._read_sequences()
        for k, dim in zip(KMER_SIZES, KMER_DIMS, strict=True):
            histogram = numpy.zeros(dim, dtype=numpy.float64)
            for sequence in sequences:
                self._count_kmers(sequence, k, histogram)
            self._histograms[k] = histogram
        return self._histograms

    def to_mlp_vector(self) -> numpy.ndarray:
        """Concatena los histogramas (k=3,4,5) en un vector de 1344 dimensiones."""
        if not self._histograms:
            raise RuntimeError("Call extract() before to_mlp_vector()")
        return numpy.concatenate([self._histograms[k] for k in KMER_SIZES])

    def to_token_sequence(
        self, k: int = TOKEN_KMER_K, max_len: int = TOKEN_MAX_LEN
    ) -> numpy.ndarray:
        """Extrae la secuencia de k-meros como tokens (IDs enteros).

        En lugar de contar frecuencias, emite la secuencia ordenada de k-meros
        tal como aparecen en el genoma [Mikolov13; Haykin, Cap. 7]. Cada k-mero
        se codifica con el rolling hash de 2 bits existente [Compeau14].

        Si la secuencia total excede max_len, se submuestrea uniformemente
        para mantener cobertura del genoma completo [Haykin, Cap. 1.2].

        Parámetros:
            k: tamaño del k-mero (default 4, vocab = 256)
            max_len: longitud máxima de la secuencia de salida (default 4096)

        Retorna:
            numpy.ndarray de shape (max_len,) con dtype int64.
            Valores en [0, 4^k - 1]. Si el genoma produce menos de max_len
            tokens, se rellena con un token de padding (4^k).
        """
        # 1. Extraer TODOS los tokens del genoma
        all_tokens = []
        sequences = self._read_sequences()
        mask = (4**k) - 1
        for sequence in sequences:
            current = 0
            valid_count = 0
            for base in sequence:
                base_index = BASE_TO_INDEX.get(base)
                if base_index is None:
                    valid_count = 0
                    current = 0
                else:
                    current = ((current << 2) | base_index) & mask
                    valid_count += 1
                    if valid_count >= k:
                        all_tokens.append(current)

        total = len(all_tokens)

        # 2. Subsampling uniforme o padding
        if total >= max_len:
            # Seleccionar max_len posiciones uniformemente distribuidas.
            # numpy.linspace genera índices equidistantes que cubren todo el genoma.
            # Esto preserva la cobertura global [Haykin, Cap. 1.2].
            indices = numpy.linspace(0, total - 1, max_len, dtype=int)
            tokens = numpy.array(all_tokens, dtype=numpy.int64)[indices]
        else:
            # Padding con token especial (4^k) para secuencias cortas.
            pad_token = 4**k
            tokens = numpy.full(max_len, pad_token, dtype=numpy.int64)
            tokens[:total] = all_tokens

        return tokens

    def to_tiled_histogram_matrix(
        self, k: int = HIER_KMER_K, n_segments: int = HIER_N_SEGMENTS
    ) -> numpy.ndarray:
        """Divide el genoma en n_segments y calcula el histograma de k-meros por segmento.

        Garantiza cobertura del 100% del genoma sin el sesgo de subsampling
        del modelo Token BiGRU. Cada segmento es una 'ventana' contigua de bases
        que preserva la localidad geográfica de los genes de resistencia.

        Parámetros:
            k: tamaño del k-mero (default 4, dim=256)
            n_segments: número de divisiones geográficas (default 64)

        Retorna:
            numpy.ndarray de shape (n_segments, 4^k) con dtype float32.
            Cada fila es un histograma de frecuencias relativas (suma 1).
        """
        # 1. Concatenar todos los contigs separados por k-1 N's.
        # Las N's reinician el rolling hash (_count_kmers ignora bases ambiguas),
        # evitando k-meros artificiales en los límites entre contigs de ensamblajes draft.
        separator = "N" * (k - 1)
        full_seq = separator.join(self._read_sequences())
        total_len = len(full_seq)

        if total_len == 0:
            return numpy.zeros((n_segments, 4**k), dtype=numpy.float32)

        # 2. Calcular tamaño base del segmento
        segment_size = total_len // n_segments
        matrix = numpy.zeros((n_segments, 4**k), dtype=numpy.float32)

        for i in range(n_segments):
            start = i * segment_size
            # El último segmento absorbe el remanente (si total_len % n_segments != 0)
            end = (i + 1) * segment_size if i < n_segments - 1 else total_len

            segment = full_seq[start:end]
            if not segment:
                continue

            # 3. Calcular histograma para este segmento geográfico
            hist = numpy.zeros(4**k, dtype=numpy.float64)
            self._count_kmers(segment, k, hist)

            # 4. Normalizar a frecuencia relativa (Haykin, Cap. 1.2)
            # Esto hace que la representación sea independiente de la longitud del segmento.
            s = hist.sum()
            if s > 0:
                matrix[i] = (hist / s).astype(numpy.float32)

        return matrix

    def _read_sequences(self) -> list[str]:
        return [
            str(record.seq).upper()
            for record in SeqIO.parse(self._fasta_path, "fasta")
        ]

    @staticmethod
    def _count_kmers(sequence: str, k: int, histogram: numpy.ndarray) -> None:
        """Cuenta k-meros usando un rolling hash de 2 bits. O(n) por secuencia.

        Cada base se codifica en 2 bits (A=00, C=01, G=10, T=11). El hash
        avanza una base a la vez mediante desplazamiento izquierdo, OR y AND con máscara.
        Las rachas de bases ambiguas (p. ej. 'N') reinician el hash.

        Técnica: codificación 2-bit estándar para conteo de k-meros en ADN.
        Ver: Compeau & Pevzner, Bioinformatics Algorithms (2014), Cap. 9.
        """
        mask = (4**k) - 1
        current = 0
        valid_count = 0
        for base in sequence:
            base_index = BASE_TO_INDEX.get(base)
            if base_index is None:
                valid_count = 0
                current = 0
            else:
                current = ((current << 2) | base_index) & mask
                valid_count += 1
                if valid_count >= k:
                    histogram[current] += 1


def build_antibiotic_index(antibiotics: pandas.Series) -> pandas.DataFrame:
    """Crea un mapeo ordenado de nombre de antibiótico a índice entero."""
    unique = sorted(antibiotics.unique())
    return pandas.DataFrame({"antibiotic": unique, "index": range(len(unique))})


def split_genomes(
    labels: pandas.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    random_seed: int = RANDOM_SEED,
) -> pandas.DataFrame:
    """Split estratificado train/val/test por genome_id.

    Estratifica sobre el fenotipo mayoritario por genoma para preservar
    la distribución de clases en cada split. La partición en subconjuntos
    de estimación (train) y validación sigue el marco de validación cruzada
    descrito en Haykin (2009), Sección 4.13.
    """
    genome_phenotype = (
        labels.groupby("genome_id")["resistant_phenotype"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"resistant_phenotype": "majority_phenotype"})
    )

    val_test_ratio = 1.0 - train_ratio
    train_ids, val_test_ids, _, val_test_labels = train_test_split(
        genome_phenotype["genome_id"],
        genome_phenotype["majority_phenotype"],
        test_size=val_test_ratio,
        stratify=genome_phenotype["majority_phenotype"],
        random_state=random_seed,
    )

    relative_val = val_ratio / val_test_ratio
    val_ids, test_ids = train_test_split(
        val_test_ids,
        test_size=1.0 - relative_val,
        stratify=val_test_labels,
        random_state=random_seed,
    )

    splits = pandas.concat(
        [
            pandas.DataFrame({"genome_id": train_ids, "split": "train"}),
            pandas.DataFrame({"genome_id": val_ids, "split": "val"}),
            pandas.DataFrame({"genome_id": test_ids, "split": "test"}),
        ],
        ignore_index=True,
    )
    logger.info(
        f"Split complete: "
        f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )
    return splits


def normalize_features(
    vectors: dict[str, numpy.ndarray],
    train_ids: set[str],
) -> tuple[dict[str, numpy.ndarray], numpy.ndarray, numpy.ndarray]:
    """Normalización z-score usando media y std del train set. Devuelve (normalizado, media, std).

    Cada feature se transforma como x' = (x - media) / std, resultando en media cero
    y varianza unitaria (LeCun et al., 1998, Sección 4.3). Las estadísticas se calculan
    solo sobre el train set para evitar data leakage (Haykin, 2009, Sección 4.6,
    Heurístico 5).
    """
    train_matrix = numpy.stack([vectors[gid] for gid in sorted(train_ids)])
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    std[std == 0] = 1.0
    normalized = {gid: (vec - mean) / std for gid, vec in vectors.items()}
    return normalized, mean, std


def mlp_vector_to_bigru_matrix(mlp_vector: numpy.ndarray) -> numpy.ndarray:
    """Convierte un vector MLP de 1344 dims en una matriz de entrada BiGRU de [1024, 3].

    Separa el vector en sus histogramas originales (64, 256, 1024), rellena cada uno
    con ceros hasta 1024 y los apila como columnas.
    """
    columns = []
    offset = 0
    for dim in KMER_DIMS:
        histogram = mlp_vector[offset : offset + dim]
        padded = numpy.pad(histogram, (0, BIGRU_PAD_DIM - dim))
        columns.append(padded)
        offset += dim
    return numpy.stack(columns, axis=1)
