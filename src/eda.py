"""
eda.py

Análisis exploratorio del dataset de etiquetas AMR (Exploratory Data Analysis).

El EDA examina el CSV de etiquetas descargado de BV-BRC e imprime en consola:
    - Resumen general del dataset
    - Distribución de registros y genomas por especie
    - Balance de clases global (Resistant / Susceptible) y pos_weight sugerido
    - Ranking de antibióticos por número de registros con su balance de clases
    - Calidad de datos: valores nulos y registros duplicados
    - Outliers: genomas extremos, antibióticos muy desbalanceados, etiquetas contradictorias
    - Baseline benchmark: majority class global y por antibiótico
    - Análisis genómico (opcional): longitud, contigs y GC content de archivos FASTA

Los hallazgos del EDA informan decisiones del pipeline:
    - Dim del embedding de antibiótico: min(50, (n_antibióticos // 2) + 1)
    - pos_weight para BCEWithLogitsLoss
    - Estrategia para duplicados (genome_id + antibiotic)
"""

from pathlib import Path

import numpy
import pandas
from Bio import SeqIO

from bvbrc import ESKAPE_TAXON_IDS


# Mapeo inverso: taxon_id → nombre de especie, para mostrar nombres legibles en el reporte
TAXON_ID_TO_SPECIES_NAME = {taxon_id: name for name, taxon_id in ESKAPE_TAXON_IDS.items()}


def export_contradictions(labels_path: Path, output_path: Path) -> int:
    """
    Finds (genome_id, antibiotic) pairs with contradictory labels and exports them to CSV.

    A contradictory pair is one where the same genome was tested against the same
    antibiotic and produced both Resistant and Susceptible results in different records.

    Args:
        labels_path: Path to the AMR labels CSV.
        output_path: Path where the contradictions CSV will be saved.

    Returns:
        Number of contradictory pairs found.
    """
    dataframe = pandas.read_csv(labels_path, dtype={"genome_id": str})

    contradictory_pairs = (
        dataframe.groupby(["genome_id", "antibiotic"])["resistant_phenotype"]
        .nunique()
        .gt(1)
    )
    contradictory_indices = contradictory_pairs[contradictory_pairs].index

    result = (
        dataframe.set_index(["genome_id", "antibiotic"])
        .loc[contradictory_indices]
        .reset_index()
        .sort_values(["genome_id", "antibiotic"])
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    return len(contradictory_indices)


def run_eda(labels_path: Path, top_n: int = 20, genomes_dir: Path | None = None) -> None:
    """
    Carga el CSV de etiquetas AMR y ejecuta el análisis exploratorio completo.

    Args:
        labels_path:  Ruta al CSV de etiquetas AMR generado por fetch_amr_labels().
        top_n:        Número de antibióticos a mostrar en el ranking.
        genomes_dir:  Directorio con archivos .fna. Si se provee, incluye análisis genómico.
    """
    dataframe = pandas.read_csv(labels_path, dtype={"genome_id": str})

    _print_section("RESUMEN GENERAL")
    _print_overview(dataframe)

    _print_section("REGISTROS POR ESPECIE")
    _print_per_species(dataframe)

    _print_section("BALANCE DE CLASES GLOBAL")
    _print_class_balance(dataframe)

    _print_section(f"TOP {top_n} ANTIBIÓTICOS POR NÚMERO DE REGISTROS")
    _print_top_antibiotics(dataframe, top_n)

    _print_section("CALIDAD DE DATOS")
    _print_data_quality(dataframe)

    _print_section("OUTLIERS")
    _print_outliers(dataframe)

    _print_section("BASELINE BENCHMARK")
    _print_baseline_benchmark(dataframe)

    if genomes_dir is not None:
        _print_section("ANÁLISIS GENÓMICO")
        _print_genome_analysis(genomes_dir, dataframe)


# ── Secciones del reporte ──────────────────────────────────────────────────────

def _print_overview(dataframe: pandas.DataFrame) -> None:
    total_records = len(dataframe)
    unique_genomes = dataframe["genome_id"].nunique()
    unique_antibiotics = dataframe["antibiotic"].nunique()
    unique_species = dataframe["taxon_id"].nunique()

    print(f"  Total de registros:       {total_records:>10,}")
    print(f"  Genome IDs únicos:        {unique_genomes:>10,}")
    print(f"  Antibióticos distintos:   {unique_antibiotics:>10,}")
    print(f"  Especies en el dataset:   {unique_species:>10,}")

    # Regla empírica definida en docs/4_models.md
    embedding_dim = min(50, (unique_antibiotics // 2) + 1)
    print(
        f"\n  → Dim embedding antibiótico sugerida: {embedding_dim}"
        f"  [min(50, ({unique_antibiotics} // 2) + 1)]"
    )


def _print_per_species(dataframe: pandas.DataFrame) -> None:
    header = f"  {'Especie':<35} {'Registros':>10} {'Genomas':>8} {'R%':>7} {'S%':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for taxon_id, group in dataframe.groupby("taxon_id"):
        species_name = TAXON_ID_TO_SPECIES_NAME.get(taxon_id, f"taxon_id={taxon_id}")
        total = len(group)
        unique_genomes = group["genome_id"].nunique()
        resistant_pct = (group["resistant_phenotype"] == "Resistant").mean() * 100
        susceptible_pct = 100 - resistant_pct

        print(
            f"  {species_name:<35} {total:>10,} {unique_genomes:>8,}"
            f" {resistant_pct:>6.1f}% {susceptible_pct:>6.1f}%"
        )


def _print_class_balance(dataframe: pandas.DataFrame) -> None:
    counts = dataframe["resistant_phenotype"].value_counts()
    total = len(dataframe)

    resistant_count = counts.get("Resistant", 0)
    susceptible_count = counts.get("Susceptible", 0)

    print(f"  Resistant:    {resistant_count:>10,}  ({resistant_count / total * 100:.1f}%)")
    print(f"  Susceptible:  {susceptible_count:>10,}  ({susceptible_count / total * 100:.1f}%)")

    if resistant_count > 0 and susceptible_count > 0:
        # pos_weight = negativos / positivos; "positivo" = Resistant (clase de interés)
        pos_weight = susceptible_count / resistant_count
        print(f"\n  → pos_weight sugerido (Susceptible / Resistant): {pos_weight:.4f}")


def _print_top_antibiotics(dataframe: pandas.DataFrame, top_n: int) -> None:
    antibiotic_counts = dataframe.groupby("antibiotic").size().sort_values(ascending=False)
    top_antibiotics = antibiotic_counts.head(top_n)

    header = f"  {'Antibiótico':<35} {'Registros':>10} {'R%':>7} {'S%':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for antibiotic, count in top_antibiotics.items():
        group = dataframe[dataframe["antibiotic"] == antibiotic]
        resistant_pct = (group["resistant_phenotype"] == "Resistant").mean() * 100
        susceptible_pct = 100 - resistant_pct

        print(
            f"  {antibiotic:<35} {count:>10,}"
            f" {resistant_pct:>6.1f}% {susceptible_pct:>6.1f}%"
        )

    remaining = len(antibiotic_counts) - top_n
    if remaining > 0:
        print(f"\n  ... y {remaining} antibióticos más.")


def _print_data_quality(dataframe: pandas.DataFrame) -> None:
    total = len(dataframe)

    print("  Valores nulos por columna:")
    has_nulls = False
    for column in dataframe.columns:
        null_count = dataframe[column].isna().sum()
        if null_count > 0:
            print(f"    {column:<35} {null_count:>8,}  ({null_count / total * 100:.1f}%)")
            has_nulls = True
    if not has_nulls:
        print("    Ninguno.")

    duplicate_count = dataframe.duplicated(subset=["genome_id", "antibiotic"]).sum()
    print(f"\n  Registros duplicados (genome_id + antibiotic): {duplicate_count:,}")


def _print_outliers(dataframe: pandas.DataFrame) -> None:
    # Genomas con número extremo de registros
    genome_counts = dataframe.groupby("genome_id").size()
    mean_records = genome_counts.mean()
    std_records = genome_counts.std()
    threshold = mean_records + 3 * std_records
    outlier_genomes = genome_counts[genome_counts > threshold]

    print(f"  Registros por genoma — media: {mean_records:.1f}, std: {std_records:.1f}, umbral (mean+3σ): {threshold:.1f}")
    print(f"  Genomas con registros extremos (>{threshold:.0f}): {len(outlier_genomes)}")
    if not outlier_genomes.empty:
        for genome_id, count in outlier_genomes.sort_values(ascending=False).head(5).items():
            print(f"    {genome_id}  →  {count} registros")

    # Antibióticos con desbalance extremo de clases (>90% una clase)
    print()
    imbalanced = []
    for antibiotic, group in dataframe.groupby("antibiotic"):
        resistant_pct = (group["resistant_phenotype"] == "Resistant").mean() * 100
        if resistant_pct >= 90 or resistant_pct <= 10:
            imbalanced.append((antibiotic, len(group), resistant_pct))

    imbalanced.sort(key=lambda x: abs(x[2] - 50), reverse=True)
    print(f"  Antibióticos con desbalance extremo (R%≥90 o R%≤10): {len(imbalanced)}")
    header = f"  {'Antibiótico':<35} {'Registros':>10} {'R%':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for antibiotic, count, resistant_pct in imbalanced[:10]:
        print(f"  {antibiotic:<35} {count:>10,} {resistant_pct:>6.1f}%")

    # Etiquetas contradictorias: mismo genome_id + antibiótico con fenotipos distintos
    conflicts = (
        dataframe.groupby(["genome_id", "antibiotic"])["resistant_phenotype"]
        .nunique()
        .gt(1)
        .sum()
    )
    print(f"\n  Pares (genome_id, antibiotic) con etiquetas contradictorias: {conflicts:,}")


def _print_baseline_benchmark(dataframe: pandas.DataFrame) -> None:

    # Baseline de clase mayoritaria global: siempre predecir "Resistant" (clase mayoritaria)
    total = len(dataframe)
    resistant_count = (dataframe["resistant_phenotype"] == "Resistant").sum()
    susceptible_count = total - resistant_count
    majority_class = "Resistant" if resistant_count >= susceptible_count else "Susceptible"
    majority_accuracy = max(resistant_count, susceptible_count) / total * 100
    print(f"  Majority class global: '{majority_class}' — accuracy: {majority_accuracy:.1f}%")

    # Baseline de clase mayoritaria por antibiótico
    y_true = []
    y_pred = []
    for antibiotic, group in dataframe.groupby("antibiotic"):
        majority = group["resistant_phenotype"].mode()[0]
        y_true.extend(group["resistant_phenotype"].tolist())
        y_pred.extend([majority] * len(group))

    y_true_bin = numpy.array([1 if y == "Resistant" else 0 for y in y_true])
    y_pred_bin = numpy.array([1 if y == "Resistant" else 0 for y in y_pred])

    tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
    fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum()
    fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()
    tn = ((y_true_bin == 0) & (y_pred_bin == 0)).sum()

    accuracy = (tp + tn) / len(y_true_bin) * 100
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n  Baseline por antibiótico (majority class de cada antibiótico):")
    print(f"    Accuracy:             {accuracy:.1f}%")
    print(f"    Precision (Resistant): {precision:.4f}")
    print(f"    Recall    (Resistant): {recall:.4f}")
    print(f"    F1        (Resistant): {f1:.4f}")
    print(f"\n  → Este es el piso mínimo que deben superar MLP y BiGRU.")


def _print_genome_analysis(genomes_dir: Path, dataframe: pandas.DataFrame) -> None:
    fasta_files = sorted(genomes_dir.glob("*.fna"))
    if not fasta_files:
        print(f"  No se encontraron archivos .fna en: {genomes_dir}")
        return

    print(f"  Archivos .fna encontrados: {len(fasta_files)}\n")

    genome_stats = []
    for fasta_path in fasta_files:
        genome_id = fasta_path.stem
        records = list(SeqIO.parse(fasta_path, "fasta"))
        if not records:
            continue

        total_length = sum(len(r.seq) for r in records)
        num_contigs = len(records)
        gc_count = sum(r.seq.upper().count("G") + r.seq.upper().count("C") for r in records)
        gc_content = gc_count / total_length * 100 if total_length > 0 else 0.0
        n_content = sum(r.seq.upper().count("N") for r in records)
        n_pct = n_content / total_length * 100 if total_length > 0 else 0.0

        matches = dataframe.loc[dataframe["genome_id"] == genome_id, "taxon_id"]
        taxon_id = matches.iloc[0] if not matches.empty else None
        species = TAXON_ID_TO_SPECIES_NAME.get(taxon_id, "desconocida") if taxon_id else "desconocida"

        genome_stats.append({
            "genome_id": genome_id,
            "species": species,
            "total_length_mb": total_length / 1e6,
            "num_contigs": num_contigs,
            "gc_content": gc_content,
            "n_pct": n_pct,
        })

    stats_df = pandas.DataFrame(genome_stats)

    # Resumen global
    header = f"  {'Métrica':<30} {'Media':>10} {'Std':>10} {'Min':>10} {'Max':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for col, label in [
        ("total_length_mb", "Longitud total (Mb)"),
        ("num_contigs",     "Número de contigs"),
        ("gc_content",      "Contenido GC (%)"),
        ("n_pct",           "Bases N (%)"),
    ]:
        vals = stats_df[col]
        print(f"  {label:<30} {vals.mean():>10.2f} {vals.std():>10.2f} {vals.min():>10.2f} {vals.max():>10.2f}")

    # Resumen por especie
    print()
    header2 = f"  {'Especie':<35} {'N':>4} {'Long. media (Mb)':>17} {'Contigs med.':>13} {'GC% med.':>9}"
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for species, grp in stats_df.groupby("species"):
        print(
            f"  {species:<35} {len(grp):>4}"
            f" {grp['total_length_mb'].mean():>17.2f}"
            f" {grp['num_contigs'].mean():>13.1f}"
            f" {grp['gc_content'].mean():>9.1f}%"
        )

    # Alertas
    short_genomes = stats_df[stats_df["total_length_mb"] < 0.5]
    fragmented = stats_df[stats_df["num_contigs"] > 500]
    high_n = stats_df[stats_df["n_pct"] > 5]

    print()
    print(f"  Genomas cortos (<0.5 Mb):          {len(short_genomes)}")
    print(f"  Genomas muy fragmentados (>500 contigs): {len(fragmented)}")
    print(f"  Genomas con >5% bases N:           {len(high_n)}")


# ── Helpers de formato ─────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
