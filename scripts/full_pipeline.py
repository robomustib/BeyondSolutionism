import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import SEED, DATA_PATH, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR
from analyzer import ContextSensitiveAnalyzer
from statistics import BiasAuditor
from visualizations import AdvancedVisualizer
from sbert_validator import SBERTValidator


def main():
    print("=" * 80)
    print("BEYOND SOLUTIONISM - FULL PIPELINE v25.0")
    print("=" * 80)
    print(f"Seed: {SEED}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    print("\n[1/5] Loading data...")
    
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure data/vignetten_nrw.csv exists")
        sys.exit(1)
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    print(f"Loaded {len(df)} vignettes")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")
    print(f"Models: {df['model'].value_counts().to_dict()}")
    
    # ============================================================================
    # 2. CONTEXT ANALYSIS (Stages 1+2)
    # ============================================================================
    print("\n[2/5] Running context-sensitive analysis...")
    
    analyzer = ContextSensitiveAnalyzer()
    results_df = analyzer.analyze_corpus(df)
    print(f"Analysis complete: {len(results_df)} texts")
    
    # Save analyzed data
    results_df.to_csv(RESULTS_DIR / "analyzed_data.csv", index=False, encoding='utf-8')
    print(f"   → Saved: {RESULTS_DIR / 'analyzed_data.csv'}")
    
    # ============================================================================
    # 3. STATISTICAL ANALYSIS
    # ============================================================================
    print("\n[3/5] Running statistical analysis...")
    
    auditor = BiasAuditor()
    stats_results = auditor.calculate_statistics(results_df)
    
    print("\n   Key Results (FDR-corrected):")
    for key in ['medicalization', 'inspiration', 'agency']:
        if key in stats_results:
            d = stats_results[key]
            sig = '***' if d.get('p_value_fdr', 1) < 0.001 else '**' if d.get('p_value_fdr', 1) < 0.01 else '*' if d.get('p_value_fdr', 1) < 0.05 else 'n.s.'
            print(f"      {key}: r_rb={d.get('effect_size_rrb', 0):.3f}, p={d.get('p_value_fdr', 1):.4f} {sig}")
    
    # ============================================================================
    # 4. SBERT VALIDATION (Stage 3)
    # ============================================================================
    print("\n[4/5] Running SBERT construct validation...")
    
    validator = SBERTValidator()
    sbert_results = validator.validate(df)
    sbert_discriminant = validator.calculate_discriminant_validity(df)
    sbert_silhouette = validator.calculate_silhouette_score(df)
    
    print(f"\n   Silhouette score: {sbert_silhouette:.3f}")
    
    # ============================================================================
    # 5. VISUALIZATIONS
    # ============================================================================
    print("\n[5/5] Generating visualizations...")
    
    visualizer = AdvancedVisualizer(results_df, stats_results, FIGURES_DIR)
    visualizer.plot_heatmap()
    visualizer.plot_autonomy_gap()
    visualizer.plot_pca()
    visualizer.plot_scatter_agency_inspiration()
    visualizer.plot_violin_comparison()
    print(f"All figures saved to {FIGURES_DIR}")
    
    # ============================================================================
    # 6. GENERATE REPORT
    # ============================================================================
    print("\n[6/6] Generating comprehensive report...")
    
    report_lines = [
        "=" * 80,
        "BEYOND SOLUTIONISM – COMPREHENSIVE RESEARCH REPORT v25.0",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Seed: {SEED}",
        "=" * 80,
        "",
        "1. DATA",
        "-" * 40,
        f"   Total texts: {len(results_df)}",
        f"   Normative: {len(results_df[results_df['condition'] == 'normative'])}",
        f"   Disability: {len(results_df[results_df['condition'] == 'disability'])}",
        "",
        "2. STATISTICAL RESULTS (FDR-corrected)",
        "-" * 40,
    ]
    
    for key, label in [('medicalization', 'Medikalisierung'), 
                        ('inspiration', 'Inspiration Porn'),
                        ('agency', 'Student Agency')]:
        if key in stats_results:
            d = stats_results[key]
            sig = '***' if d.get('p_value_fdr', 1) < 0.001 else '**' if d.get('p_value_fdr', 1) < 0.01 else '*' if d.get('p_value_fdr', 1) < 0.05 else 'n.s.'
            report_lines.append(f"\n   {label}:")
            report_lines.append(f"      Normative: M={d.get('norm_mean', 0):.4f} (SD={d.get('norm_std', 0):.4f})")
            report_lines.append(f"      Disability: M={d.get('dis_mean', 0):.4f} (SD={d.get('dis_std', 0):.4f})")
            report_lines.append(f"      r_rb = {d.get('effect_size_rrb', 0):.2f}, p = {d.get('p_value_fdr', 1):.4f} {sig}")
    
    report_lines.extend([
        "",
        "3. SBERT CONSTRUCT VALIDATION",
        "-" * 40,
        f"\n   Silhouette score: {sbert_silhouette:.3f}",
        ""
    ])
    
    for construct, res in sbert_results.items():
        sig = '***' if res['p_cond'] < 0.001 else '**' if res['p_cond'] < 0.01 else '*' if res['p_cond'] < 0.05 else 'n.s.'
        status = "Valid" if res['significant'] and res['cohens_d'] > 0.2 else "Not valid"
        report_lines.append(f"\n   {construct.upper()}:")
        report_lines.append(f"      r_cond = {res['r_cond']:.3f} {sig}")
        report_lines.append(f"      Cohen's d = {res['cohens_d']:.2f}")
        report_lines.append(f"      Status: {status}")
    
    report_lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80
    ])
    
    report_path = REPORTS_DIR / "comprehensive_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"   → Report saved: {report_path}")
    
    # ============================================================================
    # 7. SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n Total texts analyzed: {len(results_df)}")
    print(f"\n Key findings (FDR-corrected):")
    print(f" Inspiration Porn: r_rb = {stats_results.get('inspiration', {}).get('effect_size_rrb', 0):.3f}")
    print(f" Medicalization: r_rb = {stats_results.get('medicalization', {}).get('effect_size_rrb', 0):.3f}")
    print(f" Student Agency: r_rb = {stats_results.get('agency', {}).get('effect_size_rrb', 0):.3f}")
    print(f"\n SBERT validation:")
    best_construct = max(sbert_results.items(), key=lambda x: x[1]['cohens_d'])
    print(f" Best construct: {best_construct[0]} (d={best_construct[1]['cohens_d']:.2f})")
    print(f"\n Outputs saved to: {RESULTS_DIR}")
    print("analyzed_data.csv")
    print("comprehensive_report.txt")
    print("statistics_results.json")
    print("sbert_validation_results.json")
    print(f"figures/ ({len(list(FIGURES_DIR.glob('*.png')))} files)")
    print("\n Full pipeline complete. All analyses are reproducible.")
    print("=" * 80)


if __name__ == "__main__":
    main()
