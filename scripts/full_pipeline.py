import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import modules
sys.path.insert(0, str(Path(__file__).parent))

from config import SEED, DATA_PATH, RESULTS_DIR, FIGURES_DIR
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
    
    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    print("\n[1/6] Loading data...")
    
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure data/vignetten_nrw.csv exists")
        sys.exit(1)
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    print(f"Loaded {len(df)} vignettes")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")
    print(f"Models: {df['model'].value_counts().to_dict()}")
    
    # ============================================================================
    # 2. CONTEXT ANALYSIS
    # ============================================================================
    print("\n[2/6] Running context-sensitive analysis...")
    
    analyzer = ContextSensitiveAnalyzer()
    if analyzer.nlp is None:
        print("SpaCy model not loaded.")
        print("Install with: python -m spacy download de_core_news_sm")
        sys.exit(1)
    
    results_df = analyzer.analyze_corpus(df)
    print(f"Analysis complete: {len(results_df)} texts")
    
    # Save analyzed data
    results_df.to_csv(RESULTS_DIR / "analyzed_data.csv", index=False, encoding='utf-8')
    print(f"   → Saved: {RESULTS_DIR / 'analyzed_data.csv'}")
    
    # ============================================================================
    # 3. STATISTICAL ANALYSIS
    # ============================================================================
    print("\n[3/6] Running statistical analysis...")
    
    auditor = BiasAuditor()
    stats_results = auditor.calculate_statistics(results_df)
    model_results = auditor.calculate_model_comparisons(results_df)
    shadow_effect = auditor.calculate_schattenlehrer_effect(results_df)
    
    print("\n   Key Results (FDR-corrected):")
    for key in ['medicalization', 'inspiration', 'student_agency']:
        if key in stats_results:
            d = stats_results[key]
            sig = '***' if d.get('p_value_fdr', 1) < 0.001 else '**' if d.get('p_value_fdr', 1) < 0.01 else '*' if d.get('p_value_fdr', 1) < 0.05 else 'n.s.'
            print(f"      {key}: r_rb={d.get('effect_size_rrb', 0):.3f}, p={d.get('p_value_fdr', 1):.4f} {sig}")
    
    # ============================================================================
    # 4. SBERT VALIDATION
    # ============================================================================
    print("\n[4/6] Running SBERT construct validation...")
    
    validator = SBERTValidator()
    sbert_results = validator.validate(df)
    sbert_discriminant = validator.calculate_discriminant_validity(df)
    sbert_silhouette = validator.calculate_silhouette_score(df)
    
    print("\n   SBERT Results:")
    for construct, res in sbert_results.items():
        sig = '***' if res['p_cond'] < 0.001 else '**' if res['p_cond'] < 0.01 else '*' if res['p_cond'] < 0.05 else 'n.s.'
        status = "OK" if res['significant'] and res['cohens_d'] > 0.2 else "NOT"
        print(f"      {status} {construct}: r_cond={res['r_cond']:.3f} {sig}, d={res['cohens_d']:.2f}")
    
    print(f"\n   Silhouette score (construct separability): {sbert_silhouette:.3f}")
    
    # ============================================================================
    # 5. VISUALIZATIONS
    # ============================================================================
    print("\n[5/6] Generating visualizations...")
    
    visualizer = AdvancedVisualizer(results_df, stats_results, FIGURES_DIR)
    visualizer.plot_violin_comparison()
    visualizer.plot_autonomy_gap()
    visualizer.plot_scatter_agency_inspiration()
    visualizer.plot_heatmap()
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
        "2. GROUP COMPARISONS (FDR-corrected)",
        "-" * 40,
    ]
    
    for key, label in [('medicalization', 'Medicalization'), 
                        ('inspiration', 'Inspiration Porn'),
                        ('student_agency', 'Student Agency')]:
        if key in stats_results:
            d = stats_results[key]
            sig = '***' if d.get('p_value_fdr', 1) < 0.001 else '**' if d.get('p_value_fdr', 1) < 0.01 else '*' if d.get('p_value_fdr', 1) < 0.05 else 'n.s.'
            report_lines.append(f"\n   {label}:")
            report_lines.append(f"      Normative: M={d.get('norm_mean', 0):.4f} (SD={d.get('norm_std', 0):.4f})")
            report_lines.append(f"      Disability: M={d.get('dis_mean', 0):.4f} (SD={d.get('dis_std', 0):.4f})")
            report_lines.append(f"      r_rb = {d.get('effect_size_rrb', 0):.2f}, p = {d.get('p_value_fdr', 1):.4f} {sig}")
            report_lines.append(f"      95% CI [{d.get('effect_size_rrb_ci_lower', 0):.2f}, {d.get('effect_size_rrb_ci_upper', 0):.2f}]")
    
    if model_results:
        report_lines.extend([
            "",
            "3. MODEL COMPARISONS (Kruskal-Wallis)",
            "-" * 40,
        ])
        for metric, res in model_results.items():
            sig = '***' if res['p_value'] < 0.001 else '**' if res['p_value'] < 0.01 else '*' if res['p_value'] < 0.05 else 'n.s.'
            report_lines.append(f"\n   {metric}:")
            report_lines.append(f"      H = {res['h_statistic']:.2f}, p = {res['p_value']:.4f} {sig}")
            report_lines.append(f"      ε² = {res['epsilon_squared']:.3f}")
    
    if shadow_effect:
        report_lines.extend([
            "",
            "4. SHADOW TEACHER EFFECT",
            "-" * 40,
            f"\n   With helper: M={shadow_effect['mean_with_helper']:.4f}",
            f"   Without helper: M={shadow_effect['mean_without_helper']:.4f}",
            f"   Change: {shadow_effect['percent_change']:+.1f}%",
            f"   r_rb = {shadow_effect['effect_size_rrb']:.2f}, p = {shadow_effect['p_value_raw']:.4f}"
        ])
    
    report_lines.extend([
        "",
        "5. SBERT CONSTRUCT VALIDATION",
        "-" * 40,
        f"\n   Silhouette score: {sbert_silhouette:.3f}",
        "",
        "   Construct validation results:"
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
        "6. DISCRIMINANT VALIDITY (Inter-construct correlations)",
        "-" * 40,
    ])
    
    for col in sbert_discriminant.columns:
        report_lines.append(f"\n   {col}:")
        for idx, val in sbert_discriminant[col].items():
            if idx != col:
                report_lines.append(f"      vs {idx}: r = {val:.3f}")
    
    report_lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80
    ])
    
    report_path = RESULTS_DIR / "comprehensive_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"   → Report saved: {report_path}")
    
    # ============================================================================
    # 7. SAVE RESULTS SUMMARY (JSON)
    # ============================================================================
    import json
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        return obj
    
    results_summary = {
        'version': '25.0',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'seed': SEED,
        'n_total': len(results_df),
        'statistics': convert_to_serializable(stats_results),
        'model_comparisons': convert_to_serializable(model_results),
        'shadow_teacher_effect': convert_to_serializable(shadow_effect),
        'sbert_validation': convert_to_serializable(sbert_results),
        'sbert_silhouette': float(sbert_silhouette),
        'sbert_discriminant': convert_to_serializable(sbert_discriminant)
    }
    
    with open(RESULTS_DIR / "results_summary.json", 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"   → Summary saved: {RESULTS_DIR / 'results_summary.json'}")
    
    # ============================================================================
    # 8. FINAL SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n Total texts analyzed: {len(results_df)}")
    print(f"\n Key findings (FDR-corrected):")
    print(f"Inspiration Porn: r_rb = {stats_results.get('inspiration', {}).get('effect_size_rrb', 0):.3f}")
    print(f"Medicalization: r_rb = {stats_results.get('medicalization', {}).get('effect_size_rrb', 0):.3f}")
    print(f"Student Agency: r_rb = {stats_results.get('student_agency', {}).get('effect_size_rrb', 0):.3f}")
    print(f"\n SBERT validation:")
    best_construct = max(sbert_results.items(), key=lambda x: x[1]['cohens_d'])
    print(f"Best construct: {best_construct[0]} (d={best_construct[1]['cohens_d']:.2f})")
    print(f"Silhouette score: {sbert_silhouette:.3f}")
    print(f"\n Outputs saved to: {RESULTS_DIR}")
    print("analyzed_data.csv")
    print("comprehensive_report.txt")
    print("results_summary.json")
    print("sbert_validation_results.json")
    print(f"figures/ ({len(list(FIGURES_DIR.glob('*.png')))} files)")
    print("\n Full pipeline complete. All analyses are reproducible.")
    print("=" * 80)

if __name__ == "__main__":
    main()
