import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def rank_biserial_correlation(x, y, n_bootstrap=5000, seed=42):
    """
    Calculate rank-biserial correlation (r_rb) with bootstrap confidence interval
    
    r_rb = (2 * U) / (n1 * n2) - 1
    where U is the Mann-Whitney U statistic
    
    Interpretation:
        r_rb = 0.10 → small effect
        r_rb = 0.30 → medium effect
        r_rb = 0.50 → large effect
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    u_stat, p_val = stats.mannwhitneyu(x, y, alternative='two-sided', method='auto')
    r_rb = (2 * u_stat) / (n1 * n2) - 1
    
    # Bootstrap confidence interval
    rng = np.random.default_rng(seed)
    boot_effects = []
    for _ in range(n_bootstrap):
        x_boot = rng.choice(x, size=n1, replace=True)
        y_boot = rng.choice(y, size=n2, replace=True)
        u_boot, _ = stats.mannwhitneyu(x_boot, y_boot, alternative='two-sided', method='auto')
        boot_effects.append((2 * u_boot) / (n1 * n2) - 1)
    
    ci_lower = np.percentile(boot_effects, 2.5)
    ci_upper = np.percentile(boot_effects, 97.5)
    return r_rb, u_stat, p_val, ci_lower, ci_upper


def fdr_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction (primary correction)"""
    if len(p_values) == 0:
        return np.array([]), np.array([])
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    return p_corrected, rejected


def holm_correction(p_values, alpha=0.05):
    """Holm-Bonferroni correction (conservative robustness check)"""
    if len(p_values) == 0:
        return np.array([]), np.array([])
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='holm')
    return p_corrected, rejected


def cohens_d(x, y):
    """Cohen's d effect size (secondary, reported for SBERT)"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_sd


def kruskal_wallis_test(df, metric, group_col='model'):
    """Kruskal-Wallis H test for model comparisons"""
    if group_col not in df.columns or metric not in df.columns:
        return None
    
    groups = [df[df[group_col] == m][metric].dropna().values for m in df[group_col].unique()]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return None
    
    h_stat, p_val = stats.kruskal(*groups)
    
    # Epsilon-squared effect size
    n_total = sum(len(g) for g in groups)
    epsilon_sq = h_stat / ((n_total**2 - 1) / (n_total + 1)) if n_total > 1 else 0
    
    return {
        'h_statistic': float(h_stat),
        'p_value': float(p_val),
        'epsilon_squared': float(epsilon_sq),
        'significant': p_val < 0.05
    }


class BiasAuditor:
    """Statistical analysis for bias detection"""
    
    def __init__(self):
        self.metrics = [
            ('medicalization_score', 'medicalization', 'Medikalisierung'),
            ('inspiration_score', 'inspiration', 'Inspiration Porn'),
            ('agency_score', 'agency', 'Agency'),
            ('student_agency', 'student_agency', 'Student Agency'),
            ('admin_score', 'admin', 'Admin-Vokabular'),
            ('helper_score', 'helper', 'Helfer')
        ]
    
    def calculate_statistics(self, df):
        """
        Calculate all statistics with FDR and Holm correction
        
        Returns:
            dict: Comprehensive statistics including:
                - Group means and standard deviations
                - Mann-Whitney U statistics
                - Rank-biserial correlation (r_rb) with 95% CI
                - FDR-corrected and Holm-corrected p-values
        """
        norm = df[df['condition'] == 'normative']
        dis = df[df['condition'] == 'disability']
        
        stats_results = {
            'metadata': {
                'n_total': len(df),
                'n_normative': len(norm),
                'n_disability': len(dis),
                'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'seed': 42
            }
        }
        
        all_p_values = []
        test_keys = []
        
        for col, key, label in self.metrics:
            if col not in df.columns:
                continue
            
            dis_vals = dis[col].dropna().values
            norm_vals = norm[col].dropna().values
            
            if len(dis_vals) < 3 or len(norm_vals) < 3:
                continue
            
            r_rb, u_stat, p_raw, ci_lower, ci_upper = rank_biserial_correlation(dis_vals, norm_vals)
            d = cohens_d(dis_vals, norm_vals)
            
            stats_results[key] = {
                'label': label,
                'norm_mean': float(norm_vals.mean()),
                'norm_std': float(norm_vals.std()),
                'norm_n': len(norm_vals),
                'dis_mean': float(dis_vals.mean()),
                'dis_std': float(dis_vals.std()),
                'dis_n': len(dis_vals),
                'p_value_raw': float(p_raw),
                'effect_size_rrb': float(r_rb),
                'effect_size_rrb_ci_lower': float(ci_lower),
                'effect_size_rrb_ci_upper': float(ci_upper),
                'effect_size_cohens_d': float(d),
                'u_statistic': float(u_stat)
            }
            all_p_values.append(float(p_raw))
            test_keys.append(key)
        
        # FDR correction (primary)
        if all_p_values:
            p_fdr, _ = fdr_correction(all_p_values)
            for i, key in enumerate(test_keys):
                if key in stats_results:
                    stats_results[key]['p_value_fdr'] = float(p_fdr[i])
                    stats_results[key]['significant_fdr'] = p_fdr[i] < 0.05
        
        # Holm correction (conservative robustness check)
        if all_p_values:
            p_holm, _ = holm_correction(all_p_values)
            for i, key in enumerate(test_keys):
                if key in stats_results:
                    stats_results[key]['p_value_holm'] = float(p_holm[i])
                    stats_results[key]['significant_holm'] = p_holm[i] < 0.05
        
        return stats_results
    
    def calculate_model_comparisons(self, df):
        """Kruskal-Wallis tests for model differences"""
        if 'model' not in df.columns:
            return {}
        
        results = {}
        metrics = ['medicalization_score', 'inspiration_score', 'agency_score', 'admin_score']
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            kw_result = kruskal_wallis_test(df, metric, group_col='model')
            if kw_result:
                results[metric] = kw_result
        
        return results
    
    def calculate_schattenlehrer_effect(self, df):
        """Compare medicalization scores with vs without helper"""
        if 'has_helper' not in df.columns or 'medicalization_score' not in df.columns:
            return None
        
        with_helper = df[df['has_helper'] == True]['medicalization_score'].dropna()
        without_helper = df[df['has_helper'] == False]['medicalization_score'].dropna()
        
        if len(with_helper) < 3 or len(without_helper) < 3:
            return None
        
        r_rb, u_stat, p_raw, ci_lower, ci_upper = rank_biserial_correlation(with_helper, without_helper)
        mean_with = with_helper.mean()
        mean_without = without_helper.mean()
        percent_change = ((mean_with - mean_without) / mean_without * 100) if mean_without > 0 else 0
        
        return {
            'mean_with_helper': float(mean_with),
            'mean_without_helper': float(mean_without),
            'percent_change': float(percent_change),
            'effect_size_rrb': float(r_rb),
            'p_value_raw': float(p_raw),
            'n_with_helper': len(with_helper),
            'n_without_helper': len(without_helper),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper)
        }


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
    print(f"Models: {df['model'].value_counts().to_dict()}")
    
    auditor = BiasAuditor()
    results = auditor.calculate_statistics(df)
    model_results = auditor.calculate_model_comparisons(df)
    shadow_effect = auditor.calculate_schattenlehrer_effect(df)
    
    print("\n" + "=" * 60)
    print("STATISTICAL RESULTS (FDR-corrected)")
    print("=" * 60)
    
    for key in ['medicalization', 'inspiration', 'student_agency']:
        if key in results:
            d = results[key]
            sig = '***' if d.get('p_value_fdr', 1) < 0.001 else '**' if d.get('p_value_fdr', 1) < 0.01 else '*' if d.get('p_value_fdr', 1) < 0.05 else 'n.s.'
            print(f"\n{d['label']}:")
            print(f"   Normative: M={d['norm_mean']:.4f} (SD={d['norm_std']:.4f})")
            print(f"   Disability: M={d['dis_mean']:.4f} (SD={d['dis_std']:.4f})")
            print(f"   r_rb = {d['effect_size_rrb']:.2f}, p = {d['p_value_fdr']:.4f} {sig}")
            print(f"   95% CI [{d['effect_size_rrb_ci_lower']:.2f}, {d['effect_size_rrb_ci_upper']:.2f}]")
    
    if model_results:
        print("\n" + "=" * 60)
        print("MODEL COMPARISONS (Kruskal-Wallis)")
        print("=" * 60)
        for metric, res in model_results.items():
            sig = '***' if res['p_value'] < 0.001 else '**' if res['p_value'] < 0.01 else '*' if res['p_value'] < 0.05 else 'n.s.'
            print(f"\n{metric}: H = {res['h_statistic']:.2f}, p = {res['p_value']:.4f} {sig}")
            print(f"   ε² = {res['epsilon_squared']:.3f}")
    
    if shadow_effect:
        print("\n" + "=" * 60)
        print("SHADOW TEACHER EFFECT")
        print("=" * 60)
        print(f"\n   With helper: M={shadow_effect['mean_with_helper']:.4f}")
        print(f"   Without helper: M={shadow_effect['mean_without_helper']:.4f}")
        print(f"   Change: {shadow_effect['percent_change']:+.1f}%")
        print(f"   r_rb = {shadow_effect['effect_size_rrb']:.2f}, p = {shadow_effect['p_value_raw']:.4f}")
    
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
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(RESULTS_DIR / "statistics_results.json", 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {RESULTS_DIR / 'statistics_results.json'}")


if __name__ == "__main__":
    main()
