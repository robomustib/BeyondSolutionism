import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from config import SEED, DATA_PATH, OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR
from analyzer import ContextSensitiveAnalyzer
from statistics import BiasAuditor, generate_power_analysis_report
from visualizations import AdvancedVisualizer
from sbert_validator import SBERTValidator

def main():
    print("=" * 80)
    print("BEYOND SOLUTIONISM — FULL PIPILINE")
    print("=" * 80)
    print(f"Seed: {SEED}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    for d in [RESULTS_DIR, FIGURES_DIR, REPORTS_DIR, OUTPUT_DIR / "coding"]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Lade Daten...")
    if not DATA_PATH.exists():
        print(f"Fehler: {DATA_PATH} nicht gefunden.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    print(f"✓ {len(df)} Vignetten geladen.")
    print(f"   Conditions: {df['condition'].value_counts().to_dict()}")
    print(f"   Models:     {df['model'].value_counts().to_dict()}")

    print("\n[2/6] Kontextsensitives Scoring (Lemmatisierung + Noise-Filter)...")
    analyzer = ContextSensitiveAnalyzer()

    from tqdm import tqdm
    scores = []
    for text in tqdm(df['text'], desc="Scoring"):
        scores.append(analyzer.score_text(text))
    df = pd.concat([df, pd.DataFrame(scores)], axis=1)
    print("   → Scoring abgeschlossen.")

    print("\n[3/6] Statistische Validierung (Kruskal-Wallis, r_rb, FDR, Holm, MATTR)...")
    auditor = BiasAuditor()
    stats_results, df = auditor.calculate_statistics(df)
    print("   → FDR- & Holm-Korrektur angewendet.")

    print("\n   Key Results (FDR-korrigiert):")
    for key in ['inspiration', 'medicalization', 'agency', 'admin']:
        if key in stats_results:
            res = stats_results[key]
            sig = ('***' if res.get('p_fdr', 1) < 0.001 else
                   '**'  if res.get('p_fdr', 1) < 0.01  else
                   '*'   if res.get('p_fdr', 1) < 0.05  else 'n.s.')
            print(f"      {res['label']:<22}: r_rb={res['effect_size_rrb']:+.2f}, "
                  f"p_fdr={res.get('p_fdr', 1):.4f} {sig}")
    if 'mattr' in stats_results:
        m = stats_results['mattr']
        print(f"      MATTR: Δ={m['delta']:.4f}, p={m['p_value']:.4f}")

    print("\n[4/6] PCA zur latenten Bias-Dimension...")
    dims  = ['medicalization', 'inspiration', 'agency', 'admin', 'shadow_helper']
    X     = df[[d for d in dims if d in df.columns]].fillna(0).values
    X_s   = StandardScaler().fit_transform(X)
    pca   = PCA(n_components=2)
    pca_r = pca.fit_transform(X_s)
    df['pca1'], df['pca2'] = pca_r[:, 0], pca_r[:, 1]
    print(f"   → PC1 erklärt {pca.explained_variance_ratio_[0] * 100:.1f}% der Framing-Varianz.")

    print("\n[5/6] SBERT-Konstrukt-Validierung...")
    try:
        validator      = SBERTValidator()
        sbert_results  = validator.validate(df)
        sbert_silhouette = validator.calculate_silhouette_score(df)
        print(f"   → Silhouette Score: {sbert_silhouette:.3f}")
    except Exception as e:
        print(f"   ! SBERT-Validierung übersprungen: {e}")

    print("\n[6/6] Visualisierungen & Reports...")
    vis = AdvancedVisualizer(df, stats_results, FIGURES_DIR)
    vis.plot_heatmap()
    vis.plot_autonomy_gap()
    if hasattr(vis, 'plot_mattr_comparison'):
        vis.plot_mattr_comparison()
    print(f"   → Abbildungen gespeichert: {FIGURES_DIR}")

    generate_power_analysis_report(df, stats_results, REPORTS_DIR)

    df.to_csv(RESULTS_DIR / 'df_scored.csv', index=False, encoding='utf-8')

    stats_export = {k: v for k, v in stats_results.items() if k != 'mattr'}
    pd.DataFrame(stats_export).to_csv(
        RESULTS_DIR / 'stats_full.csv', encoding='utf-8'
    )

    def _convert(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, dict):        return {k: _convert(v) for k, v in obj.items()}
        return obj

    with open(RESULTS_DIR / 'stats.json', 'w', encoding='utf-8') as f:
        json_out = {
            k: {k2: v2 for k2, v2 in v.items() if k2 != 'apa_string'}
            for k, v in stats_export.items()
        }
        json.dump(_convert(json_out), f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✓ ANALYSE ABGESCHLOSSEN (PAPER READY)")
    print(f"   Outputs: {OUTPUT_DIR}")
    print("   * reports/power_analysis.txt")
    print("   * results/stats.csv & stats.json")
    print("   * results/df_scored.csv")
    print("   * figures/heatmap_framing.png")
    print("   * figures/agency_gap.png")
    print("   * figures/mattr_comparison.png/pdf")
    print("=" * 80)

if __name__ == "__main__":
    main()

