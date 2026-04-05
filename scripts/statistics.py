import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def normalize_umlauts(text):
    """Replace German umlauts"""
    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
        'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue'
    }
    for umlaut, replacement in replacements.items():
        text = text.replace(umlaut, replacement)
    return text


def interpret_effect_size_rrb(r):
    """Interpretation nach Kerby (2014) - vollständig"""
    if abs(r) < 0.10:
        return "vernachlässigbar"
    elif abs(r) < 0.20:
        return "klein"
    elif abs(r) < 0.30:
        return "mittel"
    elif abs(r) < 0.50:
        return "groß"
    return "sehr groß"


def format_apa(label, h_stat, p_fdr, r_rb, ci_low, ci_high, d=None):
    """APA-konforme Formatierung für Paper"""
    p_str = f"p < .001" if p_fdr < 0.001 else f"p = {p_fdr:.3f}"
    sig = "***" if p_fdr < 0.001 else "**" if p_fdr < 0.01 else "*" if p_fdr < 0.05 else "n.s."
    d_str = f", d = {d:.2f}" if d is not None else ""
    return f"{label}: H = {h_stat:.2f}, {p_str}, r_rb = {r_rb:.2f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}]{d_str} {sig}"


class BiasAuditor:
    """Statistical analysis for bias detection - vollständig wie v21_5"""
    
    def __init__(self):
        self.metrics = [
            ('medicalization', 'Medikalisierung'),
            ('inspiration', 'Inspiration Porn'),
            ('agency', 'Student Agency'),
            ('admin', 'Admin-Vokabular'),
            ('shadow_helper', 'Schattenlehrer')
        ]
    
    def calculate_statistics(self, df):
        """Calculate all statistics with FDR and Holm correction"""
        stats_results = {}
        all_p_raw = []
        test_keys = []
        
        for key, label in self.metrics:
            if key not in df.columns:
                continue
            
            dis = df[df['condition'] == 'disability'][key].dropna()
            norm = df[df['condition'] == 'normative'][key].dropna()
            
            if len(dis) < 3 or len(norm) < 3:
                continue
            
            # Kruskal-Wallis H test
            h_stat, p_kw = stats.kruskal(dis, norm)
            
            # Mann-Whitney U
            u_stat, p_u = stats.mannwhitneyu(dis, norm, alternative='two-sided')
            
            # Rank-Biserial Correlation (r_rb) - wie in v21_5
            n1, n2 = len(dis), len(norm)
            r_rb = 1 - (2 * u_stat) / (n1 * n2)
            
            # 95% CI for r_rb (Fisher's z-approximation)
            z = np.arctanh(r_rb)
            se = 1 / np.sqrt(n1 + n2 - 3)
            ci_low = np.tanh(z - 1.96 * se)
            ci_high = np.tanh(z + 1.96 * se)
            
            stats_results[key] = {
                'label': label,
                'norm_mean': float(norm.mean()),
                'norm_std': float(norm.std()),
                'norm_n': len(norm),
                'dis_mean': float(dis.mean()),
                'dis_std': float(dis.std()),
                'dis_n': len(dis),
                'h_stat': h_stat,
                'p_kruskal': p_kw,
                'u_stat': u_stat,
                'p_value_raw': p_u,
                'effect_size_rrb': r_rb,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'interpretation': interpret_effect_size_rrb(r_rb)
            }
            all_p_raw.append(p_u)
            test_keys.append(key)
        
        # FDR correction (Benjamini-Hochberg)
        if all_p_raw:
            _, p_fdr, _, _ = multipletests(all_p_raw, method='fdr_bh')
            for i, key in enumerate(test_keys):
                stats_results[key]['p_value_fdr'] = p_fdr[i]
                stats_results[key]['significant_fdr'] = p_fdr[i] < 0.05
        
        # Holm correction (conservative robustness check)
        if all_p_raw:
            _, p_holm, _, _ = multipletests(all_p_raw, method='holm')
            for i, key in enumerate(test_keys):
                stats_results[key]['p_value_holm'] = p_holm[i]
                stats_results[key]['significant_holm'] = p_holm[i] < 0.05
        
        # APA strings
        for key, res in stats_results.items():
            res['apa_string'] = format_apa(
                res['label'], res['h_stat'],
                res.get('p_value_fdr', 1.0),
                res['effect_size_rrb'],
                res['ci_low'], res['ci_high']
            )
        
        return stats_results


def main():
    """Main function"""
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / "results" / "analyzed_data.csv"
    
    if not DATA_PATH.exists():
        DATA_PATH = BASE_DIR / "data" / "vignetten_nrw.csv"
    
    if not DATA_PATH.exists():
        print(f"Error: Data file not found. Run analyzer.py first.")
        return
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    print(f"Loaded {len(df)} texts")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")
    
    auditor = BiasAuditor()
    results = auditor.calculate_statistics(df)
    
    print("\n" + "=" * 60)
    print("STATISTICAL RESULTS (FDR-corrected)")
    print("=" * 60)
    
    for key, res in results.items():
        sig = '***' if res.get('p_value_fdr', 1) < 0.001 else '**' if res.get('p_value_fdr', 1) < 0.01 else '*' if res.get('p_value_fdr', 1) < 0.05 else 'n.s.'
        print(f"\n{res['label']}:")
        print(f"   Normative: M={res['norm_mean']:.4f} (SD={res['norm_std']:.4f})")
        print(f"   Disability: M={res['dis_mean']:.4f} (SD={res['dis_std']:.4f})")
        print(f"   r_rb = {res['effect_size_rrb']:.2f}, p = {res.get('p_value_fdr', 1):.4f} {sig}")
        print(f"   {res['apa_string']}")
    
    # Save results
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    import json
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj
    
    with open(RESULTS_DIR / "statistics_results.json", 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {RESULTS_DIR / 'statistics_results.json'}")


if __name__ == "__main__":
    main()
