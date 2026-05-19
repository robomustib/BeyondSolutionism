
import json
import math
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

def normalize_umlauts(text: str) -> str:
    return (text
            .replace('\u00e4', 'ae').replace('\u00f6', 'oe')
            .replace('\u00fc', 'ue').replace('\u00df', 'ss')
            .replace('\u00c4', 'Ae').replace('\u00d6', 'Oe')
            .replace('\u00dc', 'Ue'))

def interpret_effect_size_rrb(r: float) -> str:
    if abs(r) < 0.10: return "vernachlässigbar"
    elif abs(r) < 0.20: return "klein"
    elif abs(r) < 0.30: return "mittel"
    elif abs(r) < 0.50: return "groß"
    return "sehr groß"

def format_apa(label: str, h_stat: float, p_fdr: float,
               r_rb: float, r_ci_low: float, r_ci_high: float,
               d: float = None) -> str:
    p_str = "p < .001" if p_fdr < 0.001 else f"p = {p_fdr:.3f}"
    sig   = "***" if p_fdr < 0.001 else "**" if p_fdr < 0.01 else "*" if p_fdr < 0.05 else "n.s."
    d_str = f", d = {d:.2f}" if d is not None else ""
    return (f"{label}: H = {h_stat:.2f}, {p_str}, r_rb = {r_rb:.2f}, "
            f"95% CI [{r_ci_low:.2f}, {r_ci_high:.2f}]{d_str} {sig}")

def calculate_mattr(text: str, window_size: int = 50) -> float:
    if not isinstance(text, str):
        text = str(text) if text else ""
    words = text.lower().split()
    if not words:
        return 0.0
    if len(words) < window_size:
        return len(set(words)) / len(words)
    ttr_sum   = 0.0
    n_windows = len(words) - window_size + 1
    for i in range(n_windows):
        window   = words[i:i + window_size]
        ttr_sum += len(set(window)) / window_size
    return ttr_sum / n_windows

def calculate_mattr_for_corpus(df: pd.DataFrame,
                               text_column: str = 'text',
                               window_size: int = 50) -> pd.DataFrame:
    df = df.copy()
    df['mattr_score'] = df[text_column].apply(
        lambda x: calculate_mattr(x, window_size)
    )
    return df

def calculate_post_hoc_power(r_rb: float, n1: int, n2: int,
                             alpha: float = 0.05) -> dict:
    if abs(r_rb) < 0.99:
        d = 2 * r_rb / math.sqrt(1 - r_rb ** 2)
    else:
        d = r_rb * 2
    n_total = n1 + n2
    se_d    = math.sqrt(n_total / (n1 * n2) + (d ** 2) / (2 * (n_total - 2)))
    ncp     = d / se_d if se_d > 0 else d * math.sqrt(n1 * n2 / n_total)
    z_crit  = sp_stats.norm.ppf(1 - alpha / 2)
    power   = 1 - sp_stats.norm.cdf(z_crit - ncp)
    return {
        'r_rb': r_rb, 'd': d,
        'n1': n1, 'n2': n2, 'n_total': n_total,
        'power': power,
        'interpretation': 'ausreichend' if power >= 0.80 else 'ungenügend',
    }

class BiasAuditor:

    def __init__(self):
        self.metrics = [
            ('medicalization', 'Medikalisierung'),
            ('inspiration',    'Inspiration Porn'),
            ('agency',         'Student Agency'),
            ('admin',          'Admin-Vokabular'),
            ('shadow_helper',  'Schattenlehrer'),
        ]

    def calculate_statistics(self, df: pd.DataFrame) -> tuple:
        stats_results: dict = {}
        all_p_raw: list     = []
        test_keys: list     = []

        for key, label in self.metrics:
            if key not in df.columns:
                continue
            dis  = df.loc[df['condition'] == 'disability', key].dropna()
            norm = df.loc[df['condition'] == 'normative',  key].dropna()
            if len(dis) < 3 or len(norm) < 3:
                continue

            h_stat, _   = kruskal(dis, norm)
            u_stat, p_u = mannwhitneyu(dis, norm, alternative='two-sided')

            n1, n2   = len(dis), len(norm)
            r_rb_raw = 1 - (2 * u_stat) / (n1 * n2)
            r_rb     = -r_rb_raw

            z       = np.arctanh(np.clip(r_rb, -0.9999, 0.9999))
            se      = 1 / np.sqrt(n1 + n2 - 3)
            ci_low  = np.tanh(z - 1.96 * se)
            ci_high = np.tanh(z + 1.96 * se)

            stats_results[key] = {
                'label':           label,
                'h_stat':          h_stat,
                'p_kruskal':       p_u,
                'u_stat':          u_stat,
                'p_raw':           p_u,
                'effect_size_rrb': r_rb,
                'ci_low':          ci_low,
                'ci_high':         ci_high,
                'd':               None,
                'n_dis':           n1,
                'n_norm':          n2,
                'mean_dis':        float(dis.mean()),
                'mean_norm':       float(norm.mean()),
                'interpretation':  interpret_effect_size_rrb(r_rb),
            }
            all_p_raw.append(p_u)
            test_keys.append(key)

        if all_p_raw:
            _, p_fdr,  _, _ = multipletests(all_p_raw, method='fdr_bh')
            _, p_holm, _, _ = multipletests(all_p_raw, method='holm')
            for i, key in enumerate(test_keys):
                stats_results[key]['p_fdr']      = p_fdr[i]
                stats_results[key]['p_holm']      = p_holm[i]
                stats_results[key]['significant'] = p_fdr[i] < 0.05
                stats_results[key]['apa_string']  = format_apa(
                    stats_results[key]['label'],
                    stats_results[key]['h_stat'],
                    p_fdr[i],
                    stats_results[key]['effect_size_rrb'],
                    stats_results[key]['ci_low'],
                    stats_results[key]['ci_high'],
                )

        if 'mattr_score' not in df.columns and 'text' in df.columns:
            df = calculate_mattr_for_corpus(df, window_size=50)
        if 'mattr_score' in df.columns:
            dis_m  = df.loc[df['condition'] == 'disability', 'mattr_score'].dropna()
            norm_m = df.loc[df['condition'] == 'normative',  'mattr_score'].dropna()
            if len(dis_m) >= 3 and len(norm_m) >= 3:
                u_m, p_m = mannwhitneyu(dis_m, norm_m, alternative='two-sided')
                stats_results['mattr'] = {
                    'label':       'MATTR (lexikalische Diversität)',
                    'norm_mean':   float(norm_m.mean()),
                    'dis_mean':    float(dis_m.mean()),
                    'delta':       abs(float(norm_m.mean()) - float(dis_m.mean())),
                    'p_value':     p_m,
                    'u_statistic': u_m,
                    'n_norm':      len(norm_m),
                    'n_dis':       len(dis_m),
                    'window_size': 50,
                }

        return stats_results, df

def generate_power_analysis_report(df: pd.DataFrame,
                                   stats_results: dict,
                                   output_dir: Path) -> Path:
    n_norm = len(df[df['condition'] == 'normative'])
    n_dis  = len(df[df['condition'] == 'disability'])

    effects = []
    for key, name in [
        ('medicalization', 'Medikalisierung (kleinster sign. Effekt)'),
        ('agency',         'Agency'),
        ('inspiration',    'Inspiration Porn'),
        ('shadow_helper',  'Schattenlehrer (n.s., H0-Prüfung)'),
    ]:
        if key in stats_results:
            r = abs(stats_results[key]['effect_size_rrb'])
            effects.append((name, r))

    lines = [
        "=" * 60,
        "POST-HOC-POWER-ANALYSE",
        "=" * 60,
        "alpha = 0.05, zwei-seitig",
        f"n_normativ = {n_norm}, n_disability = {n_dis}",
        "",
        f"{'Effekt':<40} | {'r_rb':<6} | {'d':<6} | {'Power':<8} | Status",
        "-" * 80,
    ]
    for name, r_rb in effects:
        pr     = calculate_post_hoc_power(r_rb=r_rb, n1=n_dis, n2=n_norm)
        status = "✓ ausreichend" if pr['power'] >= 0.80 else "! ungenügend"
        lines.append(
            f"{name:<40} | {pr['r_rb']:<6.3f} | {pr['d']:<6.3f} | "
            f"{pr['power']:<8.3f} | {status}"
        )
    lines += ["", "Interpretation nach Fritz, Morris & Richler (2012):"]
    for name, r_rb in effects:
        pr = calculate_post_hoc_power(r_rb=r_rb, n1=n_dis, n2=n_norm)
        size_label = (
            "vernachlässigbarer Effekt" if abs(r_rb) < 0.10 else
            "kleiner Effekt"            if abs(r_rb) < 0.30 else
            "mittlerer Effekt"          if abs(r_rb) < 0.50 else
            "großer Effekt"
        )
        lines.append(
            f"  * {name}: r_rb = {r_rb:.3f} → d ≈ {pr['d']:.2f} "
            f"({size_label}), Power = {pr['power']:.3f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'power_analysis.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines[:14]))
    return report_path

def main():
    from config import DATA_PATH, RESULTS_DIR, REPORTS_DIR

    data_path = RESULTS_DIR / "analyzed_data.csv"
    if not data_path.exists():
        data_path = DATA_PATH
    if not data_path.exists():
        print(f"Fehler: Datei nicht gefunden. Zunächst analyzer.py ausführen.")
        return

    df = pd.read_csv(data_path, encoding='utf-8')
    print(f"{len(df)} Texte geladen.")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")

    auditor = BiasAuditor()
    results, df = auditor.calculate_statistics(df)

    print("\n" + "=" * 60)
    print("STATISTISCHE ERGEBNISSE (FDR-korrigiert)")
    print("=" * 60)

    for key, res in results.items():
        if key == 'mattr':
            continue
        sig = ('***' if res.get('p_fdr', 1) < 0.001 else
               '**'  if res.get('p_fdr', 1) < 0.01  else
               '*'   if res.get('p_fdr', 1) < 0.05  else 'n.s.')
        print(f"\n{res['label']}:")
        print(f"   Normativ:   M = {res['mean_norm']:.4f}")
        print(f"   Disability: M = {res['mean_dis']:.4f}")
        print(f"   r_rb = {res['effect_size_rrb']:.2f}, "
              f"p_fdr = {res.get('p_fdr', 1):.4f}, "
              f"p_holm = {res.get('p_holm', 1):.4f} {sig}")
        print(f"   {res['apa_string']}")

    if 'mattr' in results:
        m = results['mattr']
        print(f"\nMATTR: Normativ={m['norm_mean']:.4f}, "
              f"Disability={m['dis_mean']:.4f}, "
              f"Δ={m['delta']:.4f}, p={m['p_value']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_export = {k: v for k, v in results.items() if k != 'mattr'}

    def _convert(obj):
        if isinstance(obj, (float, int)):
            return obj
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    with open(RESULTS_DIR / "statistics_results.json", 'w', encoding='utf-8') as f:
        json.dump(_convert(stats_export), f, indent=2, ensure_ascii=False)
    print(f"\nErgebnisse gespeichert: {RESULTS_DIR / 'statistics_results.json'}")

    generate_power_analysis_report(df, results, REPORTS_DIR)

if __name__ == "__main__":
    main()

