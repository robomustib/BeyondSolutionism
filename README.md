# BeyondSolutionism
This repository contains all scripts and data required to reproduce the analyses for the study 
"Beyond Solutionism: A Critical Audit of Techno-Ableism in AI-Generated Educational Narratives"

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/beyond-solutionism-replication.git
cd beyond-solutionism-replication
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Download German spaCy model
```bash
python -m spacy download de_core_news_sm
```

## Data
The file data/vignetten_nrw.csv contains the 500 generated vignettes
and must be present for the analysis. It includes the following columns:

| Column | Description |
|--------|-------------|
| `model` | Language model: `gpt`, `gemini`, `groq` |
| `condition` | Experimental condition: `normative`, `disability` |
| `marker` | Disability marker (e.g., "Förderschwerpunkt Lernen") or `none` |
| `prompt` | The exact prompt used for generation |
| `text` | The generated vignette text |
| `iteration` | Generation iteration number |

## Execution
Full Pipeline (Recommended)
```bash
python scripts/full_pipeline.py
```

This will:

- Load the vignette data
- Run context-sensitive lexical analysis (with negation detection)
- Perform statistical tests (Kruskal-Wallis, Mann-Whitney-U)
- Apply multiple testing corrections (FDR + Holm)
- Run SBERT-based construct validation
- Generate all publication-ready figures
- Create a comprehensive report

## Individual Modules

## Only statistical analysis (requires analyzed_data.csv)

```bash
python scripts/statistics.py
```

## Only SBERT validation

```bash
python scripts/sbert_validator.py
```

## Only visualizations (requires analyzed_data.csv)
```bash
python scripts/visualizations.py
```

## Only statistical analysis (requires analyzed_data.csv)
python scripts/statistics.py

## Only SBERT validation
python scripts/sbert_validator.py

## Only visualizations (requires analyzed_data.csv)
```bash
python scripts/visualizations.py
```

## Expected Outputs

```bash
results/
├── analyzed_data.csv              # Texts with bias scores
├── comprehensive_report.txt       # Full analysis report
├── results_summary.json           # Structured results (for paper)
├── sbert_validation_results.json  # SBERT construct validation
└── figures/
    ├── violin_comparison.png      # Violin plots (Figure 1)
    ├── autonomy_gap.png           # Autonomy gap (Figure 2)
    ├── correlation_heatmap.png    # Correlation matrix (Figure 3)
    └── scatter_agency_inspiration.png

```

## Reproducibility Statement

- Random Seed: 42 (fixed for all stochastic processes)
- Expected Results: See Table 3 in the paper
- Runtime: Approximately 5-10 minutes (depending on SBERT model download)

## API Keys (Only for Data Regeneration)

If you wish to regenerate the vignettes from scratch (instead of using the
provided CSV file), you will need API keys for:

- OpenAI (GPT-5.1)
- Google (Gemini 3.1 Pro)
- Groq (Llama 3)

Copy `.env. example` to `.env` and add your keys:

```bash

cp .env.example .env
```

Then edit `.env` with your actual API keys.

Note: The provided `vignetten_nrw.csv` already contains all 500 generated
vignettes. API keys are only required if you want to regenerate the data.

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** License.
**Note:** You are free to share and adapt the material for non-commercial purposes, provided you give appropriate credit. Commercial use is not permitted without prior consent. For details, see the [LICENSE](LICENSE) file.

Copyright (c) 2026 Mustafa Bilgin

## Citation

If you use this software for your research, please cite it as follows:

**APA Format:**
> Bilgin, M. (2026). *Beyond Solutionism: A Critical Audit of Techno-Ableism in AI-Generated Educational Narratives* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.xxxxxxxx

**BibTeX:**
```bibtex

@software{BeyondSolutionism,
  author       = {Bilgin, Mustafa},
  title        = {Beyond Solutionism: A Critical Audit of Techno-Ableism in AI-Generated Educational Narratives},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.xxxxxxxx},
  url          = {https://doi.org/10.5281/zenodo.xxxxxxxx}
}

```
