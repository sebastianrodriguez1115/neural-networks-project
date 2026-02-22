# Sources

This file consolidates sources for the proposal and groups them by purpose.

## Problem Definition and Motivation (AMR, genetics, WGS, ML)

## [1] Antibiotic resistance is driven by genetic mutations and acquired resistance genes
- Evidence: Acquired resistance can result from mutations in chromosomal genes or acquisition of external genetic determinants via horizontal gene transfer.
- Source: Munita and Arias, "Mechanisms of Antibiotic Resistance," Microbiology Spectrum, 2016. https://pmc.ncbi.nlm.nih.gov/articles/PMC4888801/
- Summary: Reviews genetic and biochemical mechanisms of bacterial resistance and frames acquired resistance as arising from chromosomal mutations and gene acquisition.

## [2] Whole genome sequencing enables genotype-based AMR detection and prediction
- Evidence: Sequencing-based methods (including WGS) enable detection and characterization of antimicrobial resistance determinants; tools exist for detecting AMR genes from WGS data.
- Source: Boolchandani et al., "Sequencing-based methods and resources to study antimicrobial resistance," Nature Reviews Genetics, 2019. https://pmc.ncbi.nlm.nih.gov/articles/PMC6525649/
- Summary: Surveys sequencing technologies, computational pipelines, and databases used to detect and characterize AMR determinants.

## [3] Machine learning can predict AMR from WGS data and identify associated mutations
- Evidence: Machine learning models (e.g., RF, SVM, CNN) can effectively predict AMR using WGS encodings; studies identify mutations associated with resistance.
- Source: Ren et al., "Prediction of antimicrobial resistance based on whole-genome sequencing and machine learning," Bioinformatics, 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8722762/
- Summary: Compares multiple ML models and sequence encodings for AMR prediction from WGS data and reports mutation associations.

## [4] Antibiotic resistance evolves via diverse mechanisms and gene transfer
- Evidence: Reviews the evolution of antibiotic resistance and the role of resistance genes and horizontal gene transfer.
- Source: Davies and Davies, "Origins and evolution of antibiotic resistance," Microbiology and Molecular Biology Reviews, 2010. https://journals.asm.org/doi/10.1128/mmbr.00016-10
- Summary: Classic review of how resistance emerges and spreads through genetic capacity and gene transfer.

## [5] Molecular mechanisms of antibiotic resistance (review)
- Evidence: Summarizes intrinsic and acquired resistance mechanisms, including target modification, efflux, and enzymatic inactivation.
- Source: Blair et al., "Molecular mechanisms of antibiotic resistance," Nature Reviews Microbiology, 2015. https://www.nature.com/articles/nrmicro3380
- Summary: Synthesizes key molecular mechanisms that underlie bacterial resistance phenotypes.

## [6] WGS as a tool to control antimicrobial resistance
- Evidence: Positions microbial WGS as a central tool for AMR control, including diagnostics and surveillance.
- Source: Koser et al., "Whole-genome sequencing to control antimicrobial resistance," Trends in Genetics, 2014. https://www.cell.com/trends/genetics/fulltext/S0168-9525(14)00114-0
- Summary: Discusses WGS applications for AMR control, diagnostics, and surveillance workflows.

## [7] Antimicrobial susceptibility prediction from genomes
- Evidence: Reviews genome-based prediction of antimicrobial susceptibility and its role in diagnostics.
- Source: Werner et al., "Antimicrobial susceptibility prediction from genomes," Trends in Microbiology, 2024. https://www.cell.com/trends/microbiology/fulltext/S0966-842X(24)00052-0
- Summary: Reviews genome-based diagnostics and prediction of resistance phenotypes from sequence data.

## [8] Computational prediction of AMR phenotypes
- Evidence: Reviews computational approaches for AMR prediction from genomic data and benchmarking considerations.
- Source: Hu et al., "Assessing computational predictions of antimicrobial resistance," Briefings in Bioinformatics, 2024. https://academic.oup.com/bib/article/25/3/bbae206/7665136
- Summary: Reviews computational AMR prediction methods and emphasizes evaluation and benchmarking considerations.

## [9] ML prediction of MICs from WGS (large-scale example)
- Evidence: Demonstrates machine learning models predicting antimicrobial MICs from genomic features in nontyphoidal Salmonella.
- Source: Nguyen et al., "Using machine learning to predict antimicrobial MICs and associated genomic features for nontyphoidal Salmonella," Journal of Clinical Microbiology, 2019. https://pmc.ncbi.nlm.nih.gov/articles/PMC6355527/
- Summary: Demonstrates large-scale ML prediction of MICs from genome-derived features.

## [10] Generalizability of ML for AMR prediction
- Evidence: Evaluates machine learning generalizability for AMR prediction using WGS datasets across regions.
- Source: Nsubuga et al., "Generalizability of machine learning in predicting antimicrobial resistance in E. coli," BMC Genomics, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC10946178/
- Summary: Examines generalizability of ML models across regions using WGS datasets for E. coli.

## Proposed Approach and Related Methods (sequence modeling, k-mers)

## [11] Reference article: BRNN for whole genome bacteria identification
- Evidence: Uses a bidirectional GRU with k-mer based distributed representation for bacteria identification.
- Source: Lugo and Barreto-Hernandez, "A Recurrent Neural Network approach for whole genome bacteria identification," Applied Artificial Intelligence, 2021. https://doi.org/10.1080/08839514.2021.1922842
- Summary: Provides the core sequence representation and BRNN architecture that inspires the proposed deep model.
- Relation: Direct methodological inspiration for the k-mer representation and bidirectional recurrent architecture in our deep model.

## [12] Deep learning approach to DNA sequence classification
- Evidence: CNN-based classification using spectral or k-mer representations for DNA sequences.
- Source: Rizzo et al., "A Deep Learning Approach to DNA Sequence Classification," Springer (CIMB), 2016. https://link.springer.com/chapter/10.1007/978-3-319-44332-4_10
- Summary: Demonstrates deep learning on DNA sequences using k-mer style representations, supporting sequence-based modeling choices.
- Relation: Supports using k-mer-like representations and CNN baselines for DNA sequence classification.

## [13] Deep learning architectures for DNA sequence classification
- Evidence: Compares CNN and RNN architectures for DNA sequence classification.
- Source: Lo Bosco and Di Gangi, "Deep Learning Architectures for DNA Sequence Classification," Springer, 2016. https://www.springerprofessional.de/en/deep-learning-architectures-for-dna-sequence-classification/12047536
- Summary: Provides comparative evidence for CNN and RNN approaches on DNA sequence data.
- Relation: Justifies evaluating recurrent vs convolutional architectures for sequence classification.

## [14] Deep learning for bacterial taxonomic classification using k-mers
- Evidence: k-mer representation with deep learning for bacterial taxonomy from sequencing data.
- Source: Fiannaca et al., "Deep learning models for bacteria taxonomic classification of metagenomic data," BMC Bioinformatics, 2018. https://pmc.ncbi.nlm.nih.gov/articles/PMC6069770/
- Summary: Uses k-mer representations and deep learning for bacterial sequence classification, aligned with k-mer based inputs.
- Relation: Reinforces k-mer based inputs for bacterial sequence classification tasks.

## [15] Deep learning taxonomic classification for metagenomics
- Evidence: Deep learning framework for microbial taxonomic classification from sequence data.
- Source: Liang et al., "DeepMicrobes: taxonomic classification for metagenomics with deep learning," NAR Genomics and Bioinformatics, 2020. https://pubmed.ncbi.nlm.nih.gov/33575556/
- Summary: Shows deep learning applied to large-scale microbial sequence classification tasks.
- Relation: Demonstrates deep learning for taxonomic identification from sequence reads at scale.

## [16] Pretrained sequence representations for DNA
- Evidence: Bidirectional transformer model trained on k-mers for DNA, providing transferable sequence embeddings.
- Source: Ji et al., "DNABERT," Bioinformatics, 2021. https://academic.oup.com/bioinformatics/article/37/15/2112/6128680
- Summary: Introduces pretrained DNA sequence embeddings that can inform feature representations in genomic modeling.
- Relation: Offers an alternative representation learning approach for DNA sequences based on k-mers.

## [17] Deep learning taxonomy with k-mer embedding + biLSTM + attention
- Evidence: DeepMicrobes uses k-mer embedding, bidirectional LSTM, and self-attention for taxonomic classification of metagenomic reads.
- Source: Liang et al., "DeepMicrobes: taxonomic classification for metagenomics with deep learning," NAR Genomics and Bioinformatics, 2020. https://pubmed.ncbi.nlm.nih.gov/33575556/
- Summary: Deep learning framework for taxonomic classification that parallels the reference article's sequence modeling and attention usage.
- Relation: Uses k-mer embedding and bidirectional recurrent layers with attention, closely aligned with the reference architecture style.

## [18] Deep learning sequence representations via genomic style matrices
- Evidence: Proposes deep learning on genome sequence representations derived from k-mer style matrices.
- Source: Yoshimura et al., "Genomic style: yet another deep-learning approach to characterize genome sequences," Bioinformatics Advances, 2021. https://academic.oup.com/bioinformaticsadvances/article/1/1/vbab039/6447506
- Summary: Shows representation learning for genome sequences using k-mer based matrices and deep learning classifiers.
- Relation: Reinforces the k-mer representation choice used in the reference article.

## [19] CNN on k-mer derived relative abundance profiles for taxonomy
- Evidence: Uses CNN with k-mer based Relative Abundance Index profiles for taxonomic classification of metagenomic fragments.
- Source: Karagoz and Nalbantoglu, "Taxonomic classification of metagenomic sequences from Relative Abundance Index profiles using deep learning," Biomedical Signal Processing and Control, 2021. https://www.sciencedirect.com/science/article/abs/pii/S1746809421001361
- Summary: CNN-based classifier using compositional k-mer statistics for taxonomy tasks on sequencing reads.
- Relation: Provides an alternative deep model and k-mer representation approach relevant to sequence-based bacterial identification.

## [20] Self-attention transformer model for metagenomic reads
- Evidence: Applies self-attention to metagenomic read classification.
- Source: Wichmann et al., "MetaTransformer: deep metagenomic sequencing read classification using self-attention models," NAR Genomics and Bioinformatics, 2023. https://pubmed.ncbi.nlm.nih.gov/37705831/
- Summary: Uses transformer-style attention for taxonomic classification of metagenomic reads.
- Relation: Offers a modern attention-based alternative to recurrent models for sequence classification.

## [21] ML classification of bacteria and archaea using whole-genome features
- Evidence: Uses ML (including neural networks) on whole-genome features to classify bacteria vs archaea.
- Source: Bergamini et al., "Machine learning classification of archaea and bacteria identifies novel predictive genomic features," BMC Genomics, 2024. https://link.springer.com/article/10.1186/s12864-024-10832-y
- Summary: Whole-genome feature-based classification with high accuracy and feature importance analysis.
- Relation: Supports genome-level classification using ML, adjacent to the reference article's whole-genome focus.

## [22] Fuzzy-weighted CNN for bacterial DNA taxonomic classification
- Evidence: CNN with fuzzy weighting for taxonomic classification of bacterial DNA segments (e.g., 500 bp).
- Source: Ghabashy et al., "Exploiting fuzzy weights in CNN model-based taxonomic classification of 500-bp sequence bacterial dataset," Scientific Reports, 2025. https://www.nature.com/articles/s41598-025-24836-5
- Summary: Enhances CNN classification for bacterial DNA segments with fuzzy logic and feature selection.
- Relation: Demonstrates recent CNN-based taxonomy classification on short DNA sequences, relevant to sequence-based identification.

## [23] Genome-based taxonomic classification using deep learning
- Evidence: Deep learning models for taxonomic classification of bacterial sequences.
- Source: Fiannaca et al., "Deep learning models for bacteria taxonomic classification of metagenomic data," BMC Bioinformatics, 2018. https://pmc.ncbi.nlm.nih.gov/articles/PMC6069770/
- Summary: k-mer based deep learning approach for bacterial taxonomy across ranks.
- Relation: Extends the reference article's sequence classification context to metagenomic taxonomy tasks.
