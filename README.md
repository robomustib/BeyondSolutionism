# BeyondSolutionism

This repository contains all scripts and data required to reproduce the analyses for the study  
**"Beyond Solutionism: A Critical Audit of Techno-Ableism in AI-Generated Educational Narratives"**  
(Bilgin & Ötvös, 2026)

## Installation

```bash
# Clone the repository
git clone https://github.com/robomustib/BeyondSolutionism.git
cd BeyondSolutionism
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

The file `data/vignetten_nrw.csv` contains the 500 generated vignettes
and must be present for the analysis. It includes the following columns:

| Column | Description |
|--------|-------------|
| `model` | Language model: `gpt`, `gemini`, `groq` |
| `condition` | Experimental condition: `normative`, `disability` |
| `marker` | Disability marker (e.g., `Förderschwerpunkt Lernen`) or `none` |
| `prompt` | The exact prompt used for generation |
| `text` | The generated vignette text |
| `iteration` | Generation iteration number |

## Execution

### Full Pipeline (Recommended)

```bash
python scripts/full_pipeline.py
```

This will:

- Load the vignette data (N=500)
- Run context-sensitive lexical analysis with negation detection (spaCy)
- Perform statistical tests (Kruskal-Wallis, Mann-Whitney U)
- Apply multiple testing corrections (FDR/Benjamini-Hochberg + Holm)
- Compute MATTR lexical diversity (Covington & McFall 2010)
- Run post-hoc power analysis (Fritz, Morris & Richler 2012)
- Generate all publication-ready figures
- Create a comprehensive APA-formatted report

### Individual Modules

```bash
# Only lexical scoring (outputs analyzed_data.csv)
python scripts/analyzer.py

# Only statistical analysis (requires analyzed_data.csv)
python scripts/statistics.py

# Only SBERT construct validation
python scripts/sbert_validator.py

# Only visualizations (requires analyzed_data.csv)
python scripts/visualizations.py
```

## Expected Outputs

```
results_v28_paper/
├── results/
│   ├── df_scored_v28.csv          # Texts with all framing scores
│   ├── stats_v28_full.csv         # Full statistical results table
│   └── stats_v28.json             # Structured results (for paper/Zenodo)
├── reports/
│   ├── final_report_v28.txt       # APA-formatted comprehensive report
│   └── power_analysis.txt         # Post-hoc power analysis report
└── figures/
    ├── heatmap_framing.png        # Spearman correlation heatmap (Figure 1)
    ├── agency_gap.png             # Agency boxplot by condition (Figure 2)
    ├── mattr_comparison.png       # MATTR lexical diversity (PNG)
    └── mattr_comparison.pdf       # MATTR lexical diversity (PDF, print-ready)
```

## Key Results (v28, paper-final)

All effects are Mann-Whitney U with FDR (Benjamini-Hochberg) and Holm correction.  
Sign convention: **positive r_rb = disability group has higher values**.

| Dimension | H | p_FDR | p_Holm | r_rb [95% CI] | Status |
|-----------|---|-------|--------|----------------|--------|
| Inspiration Porn | 72.62 | < .001 | < .001 | +0.45 [+0.37; +0.51] | *** |
| Medicalisation | 43.42 | < .001 | < .001 | −0.30 [−0.38; −0.22] | *** |
| Student Agency | 4.29 | .007 | .011 | +0.11 [+0.02; +0.20] | ** |
| Administrative Vocabulary | 18.01 | < .001 | < .001 | −0.18 [−0.27; −0.10] | *** |

MATTR equivalence check: Δ = 0.0006, U = 29316, p = .713 (n.s.) — bias differences
are not attributable to differing lexical diversity between conditions.

## Analysis Pipeline — Technical Notes (v28)

### v28 Umlaut Fix (Critical)

Prior versions stored lexicon terms with German umlauts (ä/ö/ü/ß) but applied
`normalize_umlauts()` to spaCy lemmas, causing ~20–35% of terms per lexicon to
never match. **v28 fixes this** by normalising all lexicon sets to ASCII form
(`ae/oe/ue/ss`) in `ContextSensitiveAnalyzer.__init__()`. This is a silent
correctness fix: result directions are unchanged, but lexical coverage is complete.

### Sign Convention

r_rb follows the convention **positive = disability group has higher values**:
```
r_rb = -(1 - 2·U / (n1·n2))
```
This matches Table 6 in the paper (inspiration porn +0.45, medicalisation −0.30,
agency +0.11, administrative vocabulary −0.18).

### Negation Detection

A syntactic negation filter via spaCy dependency parsing (Cohen's κ = 0.84) is
applied before scoring. Tokens governed by negation heads (`nicht`, `kein`, `nie`)
receive weight −1.0 instead of +1.0, avoiding false positives such as counting
"keine Diagnose" as medicalisation.

### Prompt-Noise Filter

Two-tier filter (`PROMPT_STRUCTURAL_NOISE` / `PROMPT_AMBIGUOUS_NOISE` in
`config.py`). All disability prompt markers (`Behinderung`, `Förderschwerpunkt`,
`AO-SF`, etc.) are placed in `PROMPT_STRUCTURAL_NOISE` to prevent prompt leakage
into the scoring. Admin lexicon entries for these terms were removed accordingly;
their effect is captured via the residual after cleaning.

### ELIF Priority (Lexicon Overlap)

Token-level assignment follows a strict elif-chain:
**Inspiration Porn > Medicalisation > Admin > Shadow Helper > Agency**  
All lexicon overlaps were audited in v26; none were found for the current term sets.

### MATTR (Lexical Diversity Equivalence Check)

Moving Average Type-Token Ratio (Covington & McFall 2010), window size = 50.
Used to verify that experimental conditions do not differ in lexical composition
(i.e., that observed bias differences are not confounded by vocabulary diversity).

### Post-hoc Power Analysis

Computed per Fritz, Morris & Richler (2012) via r_rb → d conversion.
Inspiration Porn (d ≈ 1.00): power = 1.00. Medicalisation (d ≈ 0.62): power = 1.00.
Agency (d ≈ 0.22): power = 0.67 — below 0.80 threshold; treat as exploratory.

## Reproducibility Statement

- **Random Seed:** 42 (fixed for all stochastic processes)
- **Generation period:** 16.02.2026 – 22.02.2026
- **Expected Results:** See Table 6 in the paper
- **Runtime:** Approximately 5–10 minutes (depending on SBERT model download)
- The provided `vignetten_nrw.csv` contains all 500 generated vignettes,
  enabling full reproduction without renewed API queries.

## API Keys (Only for Data Regeneration)

If you wish to regenerate the vignettes from scratch (instead of using the
provided CSV), you will need API keys for:

- OpenAI (GPT-5.1)
- Google (Gemini 3.1 Pro)
- Groq (Llama 3 / llama-3.3-70b-versatile)

Copy `.env.example` to `.env` and add your keys:

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

Note: The provided `vignetten_nrw.csv` already contains all 500 generated
vignettes. API keys are only required if you want to regenerate the data.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** License.  
You are free to share and adapt the material for non-commercial purposes, provided you give appropriate credit. Commercial use is not permitted without prior consent.  
For details, see the [LICENSE](LICENSE) file.

Copyright (c) 2026 Mustafa Bilgin

## Citation

If you use this software for your research, please cite it as follows:

**APA Format:**
> Bilgin, M., & Ötvös, B. (2026). *Beyond Solutionism: A Critical Audit of Techno-Ableism in AI-Generated Educational Narratives* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.19432611

**BibTeX:**
```bibtex
@software{BeyondSolutionism,
  author    = {Bilgin, Mustafa and {\"O}tv{\"o}s, Bettina},
  title     = {Beyond Solutionism: A Critical Audit of Techno-Ableism
               in AI-Generated Educational Narratives},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.19432611},
  url       = {https://doi.org/10.5281/zenodo.19432611}
}
```
