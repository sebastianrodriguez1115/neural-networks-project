## Project Goal and Scope

### Problem Definition
Antibiotic resistance is a genetic phenomenon that emerges through chromosomal mutations and acquisition of resistance determinants, with multiple molecular mechanisms such as target modification, efflux, and enzymatic inactivation shaping the phenotype [1][4][5]. With modern sequencing, whole genome data can be used to detect and characterize resistance determinants, and WGS is now positioned as a central tool for AMR diagnostics and surveillance [2][6][7]. The problem is to leverage whole genome sequences to predict resistance phenotypes or associated mutations reliably and at scale, using machine learning methods that have shown strong performance in AMR prediction from genomic data, while also considering generalizability across datasets and settings [3][8][9][10].

### Project Goal
Build a neural network based system that predicts antibiotic resistance from whole genome bacterial sequences, using a shallow NN baseline and a deep sequence model inspired by the reference articles.

### Proposed Approach
This project must deliver two neural network based approaches: a shallow baseline and a deeper model. The approach below follows methods used in genome sequence classification literature and the reference articles.

#### 1) Data Representation (shared by both models)
The input to any model must be numeric. For DNA, a common approach is to represent sequences as k-mers (short substrings of length k) and count their occurrences. This captures local sequence patterns without manual feature engineering.

- Choose k = 3, 4, 5 and compute k-mer histograms for each genome, as in the reference article [11].
- Concatenate the three histograms into one vector (for k = 3, 4, 5), then standardize (mean 0, variance 1) so features are comparable in scale [11].
- Optional: also stack the histograms into a 3 x 1024 matrix (one row per k) for a sequence-like input to the deep model [11].

Why this matters:
- k-mer representations are widely used in DNA classification and taxonomy tasks [12][14][19].
- They work well for both shallow models (vector input) and deeper models (matrix or embedding input) [11][12][17].

#### 2) Model A: Shallow Neural Network Baseline
This is the required shallow NN approach. It provides a strong baseline and helps quantify the gains from the deep model.

Architecture (conceptual):
- Input: concatenated k-mer vector (e.g., 1 x 1344 for k = 3, 4, 5) [11].
- One or two dense layers with ReLU activation.
- Output: softmax layer for multi-class resistance labels.

Why this baseline is reasonable:
- k-mer vectors are effective features for genome classification [12][14][19].
- A shallow MLP can learn nonlinear decision boundaries while remaining interpretable and fast to train.

#### 3) Model B: Deep Sequence Model (Bidirectional RNN + Attention)
This is the deep NN approach inspired by the reference article and related work. It is designed to capture sequence context and long-range dependencies.

Architecture (conceptual):
- Input: stacked k-mer matrix (3 x 1024) or k-mer embedding sequence.
- Bidirectional recurrent layer (GRU or LSTM) to process sequence context from both directions [11][17].
- Attention mechanism over hidden states to focus on informative subsequences [11][17][20].
- Dense layer(s) + softmax for final classification.

Why this model is appropriate:
- The reference article uses a bidirectional GRU with attention and shows improved performance in sequence classification [11].
- DeepMicrobes uses k-mer embedding + bidirectional LSTM + attention for taxonomic classification, validating this architecture family in genomic tasks [17].
- Attention provides a principled way to highlight informative regions of a sequence, improving interpretability and accuracy [11][20].

#### 4) Optional Alternative Deep Models (for literature alignment)
These are not required, but help justify choices and provide comparison points in the literature review.

- CNN-based DNA classifiers are widely used for k-mer or spectral representations [12][13][22].
- Transformer-based models (e.g., DNABERT) provide pretrained k-mer embeddings and attention-based sequence modeling [16].

#### 5) Training and Evaluation Plan (high level)
Given the 2–3 month project window, evaluation will focus on a practical core set of methods.

- Split dataset into train/validation/test with stratification by resistance labels.
- Use cross-entropy loss and Adam optimizer.
- Evaluate both models with accuracy, precision, recall, and F-score to handle class imbalance [8].
- Compare shallow vs deep performance and report improvements and limitations.

Next steps beyond this project timeline:
- Add heavier cross-validation or external validation across datasets to test generalizability [8][10].
- Explore alternative deep architectures (CNN or transformer) for sequence modeling [12][16][22].
- Expand to additional resistance targets and incorporate richer metadata (e.g., species or lineage context).

### Project Goal and Scope Draft
This project develops a neural network based system to detect bacterial antibiotic resistance from whole genome sequence data, building on recurrent sequence models used for bacterial identification in the literature [11][17]. Within a 2–3 month timeline, two complementary models are implemented: a shallow neural network baseline and a deep bidirectional RNN with attention, and their performance is compared on a labeled public dataset [11][17]. The scope covers data acquisition, preprocessing, model training, and evaluation with accuracy, precision, recall, and F-score following AMR prediction practices [8][9], while excluding clinical validation, wet lab confirmation of resistance, and deployment in clinical workflows.
