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
from data_pipeline.constants import MIN_RECORDS_PER_ANTIBIOTIC


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

    _print_section("MÉTODOS DE LABORATORIO (TYPING METHOD)")
    _print_typing_method_analysis(dataframe)

    _print_section("BASELINE BENCHMARK")
    _print_baseline_benchmark(dataframe)

    if genomes_dir is not None:
        _print_section("ANÁLISIS GENÓMICO")
        _print_genome_analysis(genomes_dir, dataframe)


def run_expanded_eda(
    raw_labels_path: Path,
    processed_dir: Path,
    non_eskape_labels_path: Path | None = None,
    top_n_taxa: int = 10,
    top_n_antibiotics: int = 10,
    min_records_per_antibiotic: int = MIN_RECORDS_PER_ANTIBIOTIC,
) -> None:
    """Ejecuta EDA reproducible del dataset expandido all-taxa.

    Usa los artefactos generados por `prepare-labels-for-download` y
    `prepare-data` para distinguir el raw combinado, los labels limpios y el
    subconjunto efectivo usado por los modelos después del filtro genómico.
    """
    raw = pandas.read_csv(raw_labels_path, dtype={"genome_id": str})
    non_eskape = None
    if non_eskape_labels_path is not None and non_eskape_labels_path.exists():
        non_eskape = pandas.read_csv(non_eskape_labels_path, dtype={"genome_id": str})

    processed_dir = Path(processed_dir)
    labels_for_download = pandas.read_csv(
        processed_dir / "labels_for_download.csv",
        dtype={"genome_id": str},
    )
    cleaned = pandas.read_csv(
        processed_dir / "cleaned_labels.csv",
        dtype={"genome_id": str},
    )
    splits = pandas.read_csv(processed_dir / "splits.csv", dtype={"genome_id": str})
    discarded_path = processed_dir / "discarded_genomes.csv"
    discarded = (
        pandas.read_csv(discarded_path, dtype={"genome_id": str})
        if discarded_path.exists()
        else pandas.DataFrame(columns=["genome_id", "reason"])
    )
    effective = _build_effective_expanded_labels(cleaned, splits, labels_for_download)
    cleaning = _summarize_expanded_cleaning(
        raw,
        min_records_per_antibiotic=min_records_per_antibiotic,
    )

    _print_section("EDA EXPANDIDO — ALCANCE")
    _print_expanded_scope(raw, non_eskape, labels_for_download, effective, discarded)

    _print_section("EDA EXPANDIDO — LIMPIEZA")
    _print_expanded_cleaning(cleaning)

    _print_section("EDA EXPANDIDO — BALANCE")
    _print_expanded_balance(effective, splits)

    _print_section("EDA EXPANDIDO — TAXONES")
    _print_expanded_taxa(effective, top_n_taxa)

    _print_section("EDA EXPANDIDO — ANTIBIÓTICOS")
    _print_expanded_antibiotics(effective, top_n_antibiotics)

    _print_section("EDA EXPANDIDO — CONFOUNDS")
    _print_expanded_confounds()


# ── Secciones del reporte ──────────────────────────────────────────────────────

def _build_effective_expanded_labels(
    cleaned: pandas.DataFrame,
    splits: pandas.DataFrame,
    labels_for_download: pandas.DataFrame,
) -> pandas.DataFrame:
    """Intersecta labels limpios con splits y agrega taxon_id para EDA."""
    split_columns = ["genome_id", "split"]
    if "split_source" in splits.columns:
        split_columns.append("split_source")
    effective = cleaned.merge(splits[split_columns], on="genome_id", how="inner")
    if "split_source" not in effective.columns:
        effective["split_source"] = "all"

    if "taxon_id" in labels_for_download.columns:
        taxon_meta = labels_for_download[["genome_id", "taxon_id"]].drop_duplicates(
            subset=["genome_id"],
        )
        effective = effective.merge(taxon_meta, on="genome_id", how="left")

    return effective


def _summarize_expanded_cleaning(
    raw: pandas.DataFrame,
    min_records_per_antibiotic: int,
) -> dict[str, int]:
    """Replica los conteos principales de limpieza usados en LabelCleaner."""
    broth = raw[raw["laboratory_typing_method"] == "Broth dilution"].reset_index(drop=True)
    phenotype_counts = broth.groupby(["genome_id", "antibiotic"])[
        "resistant_phenotype"
    ].nunique()
    contradictory_indices = phenotype_counts[phenotype_counts > 1].index
    contradictory_mask = broth.set_index(["genome_id", "antibiotic"]).index.isin(
        contradictory_indices,
    )
    without_contradictions = broth[~contradictory_mask].reset_index(drop=True)
    duplicates_removed = int(
        without_contradictions.duplicated(subset=["genome_id", "antibiotic"]).sum()
    )
    deduplicated = without_contradictions.drop_duplicates(
        subset=["genome_id", "antibiotic"],
        keep="first",
    ).reset_index(drop=True)
    counts = deduplicated["antibiotic"].value_counts()
    to_keep = counts[counts >= min_records_per_antibiotic].index

    return {
        "initial_records": len(raw),
        "typing_method_removed": len(raw) - len(broth),
        "contradictory_pairs": len(contradictory_indices),
        "contradictory_rows": int(contradictory_mask.sum()),
        "duplicates_removed": duplicates_removed,
        "low_frequency_antibiotics_removed": len(counts) - len(to_keep),
        "low_frequency_rows_removed": len(deduplicated)
        - len(deduplicated[deduplicated["antibiotic"].isin(to_keep)]),
    }


def _print_expanded_scope(
    raw: pandas.DataFrame,
    non_eskape: pandas.DataFrame | None,
    labels_for_download: pandas.DataFrame,
    effective: pandas.DataFrame,
    discarded: pandas.DataFrame,
) -> None:
    print(
        f"  Raw combinado all-taxa: {len(raw):,} registros, "
        f"{raw['genome_id'].nunique():,} genomas, "
        f"{raw['taxon_id'].nunique():,} taxones, "
        f"{raw['antibiotic'].nunique():,} antibióticos"
    )
    if non_eskape is not None:
        print(
            f"  Raw no-ESKAPE:         {len(non_eskape):,} registros, "
            f"{non_eskape['genome_id'].nunique():,} genomas, "
            f"{non_eskape['taxon_id'].nunique():,} taxones, "
            f"{non_eskape['antibiotic'].nunique():,} antibióticos"
        )
    print(
        f"  Labels limpios:        {len(labels_for_download):,} registros, "
        f"{labels_for_download['genome_id'].nunique():,} genomas, "
        f"{labels_for_download['taxon_id'].nunique():,} taxones, "
        f"{labels_for_download['antibiotic'].nunique():,} antibióticos"
    )
    print(
        f"  Dataset efectivo:      {len(effective):,} registros, "
        f"{effective['genome_id'].nunique():,} genomas"
    )
    if discarded.empty:
        print("  Genomas descartados:   0")
    else:
        print(f"  Genomas descartados:   {len(discarded):,}")
        for reason, count in discarded["reason"].value_counts().items():
            print(f"    {reason}: {count:,}")


def _print_expanded_cleaning(cleaning: dict[str, int]) -> None:
    print(f"  Registros iniciales:                         {cleaning['initial_records']:>10,}")
    print(f"  Removidos por método distinto a Broth:       {cleaning['typing_method_removed']:>10,}")
    print(f"  Pares contradictorios removidos:             {cleaning['contradictory_pairs']:>10,}")
    print(f"  Filas contradictorias removidas:             {cleaning['contradictory_rows']:>10,}")
    print(f"  Duplicados consistentes removidos:           {cleaning['duplicates_removed']:>10,}")
    print(f"  Antibióticos removidos por baja frecuencia:  {cleaning['low_frequency_antibiotics_removed']:>10,}")
    print(f"  Filas removidas por baja frecuencia:         {cleaning['low_frequency_rows_removed']:>10,}")


def _print_expanded_balance(effective: pandas.DataFrame, splits: pandas.DataFrame) -> None:
    _print_class_balance(effective)
    train = effective[effective["split"] == "train"]
    resistant = int((train["resistant_phenotype"] == "Resistant").sum())
    susceptible = int((train["resistant_phenotype"] == "Susceptible").sum())
    if resistant > 0:
        print(f"\n  → pos_weight efectivo en train: {susceptible / resistant:.4f}")

    print("\n  Balance por split_source:")
    crosstab = pandas.crosstab(
        effective["split_source"],
        effective["resistant_phenotype"],
    )
    for source, row in crosstab.iterrows():
        total = int(row.sum())
        r = int(row.get("Resistant", 0))
        s = int(row.get("Susceptible", 0))
        print(
            f"    {source:<8} {total:>8,} registros  "
            f"R={r / total * 100:>5.1f}%  S={s / total * 100:>5.1f}%"
        )

    print("\n  Genomas por split:")
    for split, count in splits["split"].value_counts().sort_index().items():
        print(f"    {split:<5} {count:>8,}")
    print("\n  Genomas por split_source:")
    for source, count in splits["split_source"].value_counts().items():
        print(f"    {source:<8} {count:>8,}")


def _print_expanded_taxa(effective: pandas.DataFrame, top_n: int) -> None:
    if "taxon_id" not in effective.columns:
        print("  No hay columna taxon_id disponible en los artefactos procesados.")
        return
    summary = (
        effective.groupby("taxon_id")
        .agg(
            records=("genome_id", "size"),
            genomes=("genome_id", "nunique"),
            resistant=("resistant_phenotype", lambda s: int((s == "Resistant").sum())),
        )
        .reset_index()
    )
    summary["r_pct"] = summary["resistant"] / summary["records"] * 100
    records = summary["records"]
    print(
        f"  Taxones: {len(summary):,}  mediana={records.median():.0f}  "
        f"p75={records.quantile(0.75):.0f}  max={records.max():,} registros"
    )
    header = f"  {'taxon_id':>10} {'Registros':>10} {'Genomas':>8} {'R%':>7}"
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for _, row in summary.sort_values("records", ascending=False).head(top_n).iterrows():
        print(
            f"  {int(row['taxon_id']):>10} {int(row['records']):>10,} "
            f"{int(row['genomes']):>8,} {row['r_pct']:>6.1f}%"
        )


def _print_expanded_antibiotics(effective: pandas.DataFrame, top_n: int) -> None:
    summary = (
        effective.groupby("antibiotic")
        .agg(
            records=("genome_id", "size"),
            genomes=("genome_id", "nunique"),
            resistant=("resistant_phenotype", lambda s: int((s == "Resistant").sum())),
        )
        .reset_index()
    )
    summary["r_pct"] = summary["resistant"] / summary["records"] * 100
    records = summary["records"]
    print(
        f"  Antibióticos: {len(summary):,}  media={records.mean():.1f}  "
        f"mediana={records.median():.1f}  p75={records.quantile(0.75):.1f}  "
        f"min={records.min():,}  max={records.max():,} registros"
    )
    header = f"  {'Antibiótico':<35} {'Registros':>10} {'Genomas':>8} {'R%':>7}"
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for _, row in summary.sort_values("records", ascending=False).head(top_n).iterrows():
        print(
            f"  {row['antibiotic']:<35} {int(row['records']):>10,} "
            f"{int(row['genomes']):>8,} {row['r_pct']:>6.1f}%"
        )


def _print_expanded_confounds() -> None:
    print("  Confounds principales del dataset expandido:")
    print("    Taxonomía: los k-meros codifican especie/género y pueden inducir atajos.")
    print("    Shift de clase: locked y new tienen priors Resistant/Susceptible muy distintos.")
    print("    Antibióticos dominio-específicos: algunos aparecen ligados a taxones concretos.")
    print("    Cola larga: muchos taxones tienen soporte demasiado bajo para métricas estables.")
    print("\n  Decisión: reportar métricas por split_source, taxon_id y antibiótico.")

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


def _print_typing_method_analysis(dataframe: pandas.DataFrame) -> None:
    total = len(dataframe)
    counts = dataframe["laboratory_typing_method"].value_counts(dropna=False)

    header = f"  {'Método de laboratorio':<35} {'Registros':>10} {'%':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for method, count in counts.items():
        # Representación legible para nulos
        method_str = "Nulo (No especificado)" if pandas.isna(method) else str(method)
        print(f"  {method_str:<35} {count:>10,} {count / total * 100:>6.1f}%")

    print("\n  → El equipo recomienda filtrar y concentrar el análisis en 'Broth dilution'")
    print("    por ser el estándar de oro (gold standard) para MIC.")


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
