import pandas

from eda import run_expanded_eda


def test_run_expanded_eda_reports_processed_artifacts(tmp_path, capsys):
    raw_path = tmp_path / "raw.csv"
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    pandas.DataFrame(
        [
            ("g1", 111, "ampicillin", "Resistant", "Broth dilution"),
            ("g1", 111, "ampicillin", "Susceptible", "Broth dilution"),
            ("g2", 111, "ampicillin", "Resistant", "Broth dilution"),
            ("g2", 111, "ampicillin", "Resistant", "Broth dilution"),
            ("g3", 222, "tetracycline", "Susceptible", "Broth dilution"),
            ("g4", 222, "tetracycline", "Resistant", "MIC"),
        ],
        columns=[
            "genome_id",
            "taxon_id",
            "antibiotic",
            "resistant_phenotype",
            "laboratory_typing_method",
        ],
    ).to_csv(raw_path, index=False)
    pandas.DataFrame(
        [
            ("g2", 111, "ampicillin", "Resistant", "Broth dilution"),
            ("g3", 222, "tetracycline", "Susceptible", "Broth dilution"),
        ],
        columns=[
            "genome_id",
            "taxon_id",
            "antibiotic",
            "resistant_phenotype",
            "laboratory_typing_method",
        ],
    ).to_csv(processed_dir / "labels_for_download.csv", index=False)
    pandas.DataFrame(
        [
            ("g2", "ampicillin", "Resistant"),
            ("g3", "tetracycline", "Susceptible"),
        ],
        columns=["genome_id", "antibiotic", "resistant_phenotype"],
    ).to_csv(processed_dir / "cleaned_labels.csv", index=False)
    pandas.DataFrame(
        [
            ("g2", "train", "locked"),
            ("g3", "test", "new"),
        ],
        columns=["genome_id", "split", "split_source"],
    ).to_csv(processed_dir / "splits.csv", index=False)
    pandas.DataFrame(
        [("g4", "below_min_length")],
        columns=["genome_id", "reason"],
    ).to_csv(processed_dir / "discarded_genomes.csv", index=False)

    run_expanded_eda(
        raw_labels_path=raw_path,
        processed_dir=processed_dir,
        non_eskape_labels_path=None,
        top_n_taxa=2,
        top_n_antibiotics=2,
        min_records_per_antibiotic=1,
    )

    output = capsys.readouterr().out
    assert "EDA EXPANDIDO" in output
    assert "Dataset efectivo" in output
    assert "Pares contradictorios removidos" in output
    assert "locked" in output
    assert "new" in output
    assert "ampicillin" in output
