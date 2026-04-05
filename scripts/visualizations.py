#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Publication-ready visualizations for bias analysis
Vollständig basierend auf beyond_solutionism_v21_5.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")


class AdvancedVisualizer:
    """Generate all publication-ready visualizations - vollständig"""
    
    def __init__(self, df, stats, output_dir):
        self.df = df
        self.stats = stats
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.colors = {
            'gpt': '#E69F00',
            'gemini': '#56B4E9',
            'groq': '#009E73',
            'normative': '#A9A9A9',
            'disability': '#D55E00'
        }
        
        self.model_labels = {
            'gpt': 'GPT-5.1',
            'gemini': 'Gemini Pro',
            'groq': 'Llama 3'
        }
    
    def plot_heatmap(self, filename='heatmap_framing.png'):
        """Correlation heatmap of bias dimensions"""
        dims = ['medicalization', 'admin', 'inspiration', 'agency', 'shadow_helper']
        avail = [d for d in dims if d in self.df.columns]
        
        if len(avail) < 2:
            print("   Warning: Not enough dimensions for heatmap")
            return
        
        corr = self.df[avail].corr(method='spearman')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", 
                   vmin=-1, vmax=1, linewidths=0.5, square=True)
        plt.title("Inter-Dimension Korrelationen (Spearman)", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.savefig(self.output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"   → {filename}")
    
    def plot_autonomy_gap(self, filename='agency_gap.png'):
        """Autonomy gap by condition"""
        if 'agency' not in self.df.columns:
            print("   Warning: agency not found in data")
            return
        
        plt.figure(figsize=(7, 5))
        sns.boxplot(x='condition', y='agency', data=self.df, palette='Set2', width=0.5)
        plt.title("Agency-Score nach Experimentalbedingung", fontsize=14, weight='bold')
        plt.ylabel("Normalisierter Agency-Score", fontsize=11)
        plt.xlabel("Bedingung", fontsize=11)
        plt.grid(alpha=0.3)
        
        # Add significance annotation
        dis = self.df[self.df['condition'] == 'disability']['agency'].dropna()
        norm = self.df[self.df['condition'] == 'normative']['agency'].dropna()
        if len(dis) > 0 and len(norm) > 0:
            u, p = mannwhitneyu(dis, norm)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            y_max = max(dis.max(), norm.max())
            plt.text(0.5, y_max + 0.05, f'p {sig}', ha='center', fontsize=12, weight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.savefig(self.output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"   → {filename}")
    
    def plot_pca(self, filename='pca_framing.png'):
        """PCA plot for latent bias dimension"""
        dims = ['medicalization', 'inspiration', 'agency', 'admin', 'shadow_helper']
        avail = [d for d in dims if d in self.df.columns]
        
        if len(avail) < 2:
            print("   Warning: Not enough dimensions for PCA")
            return
        
        X = self.df[avail].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for cond, color, label in [('disability', 'darkred', 'Disability'), 
                                     ('normative', 'darkblue', 'Normativ')]:
            mask = self.df['condition'] == cond
            ax.scatter(pca_res[mask, 0], pca_res[mask, 1], 
                      c=color, alpha=0.6, s=50, label=label, edgecolors='w')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Latente Bias-Dimension (PCA)', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.savefig(self.output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"   → {filename}")
    
    def plot_scatter_agency_inspiration(self, filename='scatter_agency_inspiration.png'):
        """Scatter plot: agency vs inspiration porn"""
        if 'agency' not in self.df.columns or 'inspiration' not in self.df.columns:
            print("   Warning: agency or inspiration not found")
            return
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for model, color in self.colors.items():
            if model not in ['gpt', 'gemini', 'groq']:
                continue
            m_data = self.df[self.df['model'] == model]
            
            norm_mask = m_data['condition'] == 'normative'
            ax.scatter(
                m_data.loc[norm_mask, 'inspiration'],
                m_data.loc[norm_mask, 'agency'],
                c=color, marker='o', alpha=0.3, s=60, edgecolors='none'
            )
            
            dis_mask = m_data['condition'] == 'disability'
            ax.scatter(
                m_data.loc[dis_mask, 'inspiration'],
                m_data.loc[dis_mask, 'agency'],
                c=color, marker='s', alpha=0.7, s=80, edgecolors='w',
                label=self.model_labels.get(model, model.upper())
            )
        
        rho, p = spearmanr(self.df['inspiration'], self.df['agency'])
        sns.regplot(data=self.df, x='inspiration', y='agency',
                   scatter=False, color='gray', ax=ax,
                   line_kws={'linestyle': '--', 'alpha': 0.5})
        
        ax.set_xlabel('Inspiration Porn (Score)', weight='bold')
        ax.set_ylabel('Agency (Score)', weight='bold')
        
        p_str = "< 0.001" if p < 0.001 else f"= {p:.3f}"
        ax.set_title(f'Zusammenhang Agency vs. Inspiration Porn\nρ = {rho:.2f} (p {p_str})', fontsize=14)
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Normativ', markerfacecolor='gray', markersize=8),
            Line2D([0], [0], marker='s', color='w', label='Disability', markerfacecolor='gray', markersize=8),
            Line2D([0], [0], color='none', label='---'),
            Line2D([0], [0], marker='s', color='w', label='GPT-5.1', markerfacecolor=self.colors['gpt']),
            Line2D([0], [0], marker='s', color='w', label='Gemini Pro', markerfacecolor=self.colors['gemini']),
            Line2D([0], [0], marker='s', color='w', label='Llama 3', markerfacecolor=self.colors['groq'])
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.savefig(self.output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"   → {filename}")
    
    def plot_violin_comparison(self, filename='violin_comparison.png'):
        """Violin plots for bias dimensions by condition"""
        metrics = ['medicalization', 'inspiration', 'agency']
        labels = ['Medikalisierung', 'Inspiration Porn', 'Agency']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            if metric not in self.df.columns:
                continue
            
            plot_data = []
            for model in self.df['model'].unique():
                for cond in ['normative', 'disability']:
                    mask = (self.df['model'] == model) & (self.df['condition'] == cond)
                    vals = self.df.loc[mask, metric].dropna()
                    for v in vals:
                        plot_data.append({
                            'model': self.model_labels.get(model, model.upper()),
                            'condition': cond,
                            'score': v
                        })
            
            df_plot = pd.DataFrame(plot_data)
            
            sns.violinplot(data=df_plot, x='model', y='score', hue='condition',
                          split=True, inner='quart', palette='muted', ax=axes[i],
                          bw_method=0.3)
            
            # Add effect size
            key = metric
            if key in self.stats:
                d = self.stats[key]
                r_rb = d.get('effect_size_rrb', 0)
                p = d.get('p_value_fdr', 1)
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
                axes[i].set_title(f"{label}\nr_rb = {r_rb:.2f} {sig}", fontweight='bold')
            else:
                axes[i].set_title(label, fontweight='bold')
            
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Score (normalisiert)')
            axes[i].legend(title='Condition', loc='upper right')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.savefig(self.output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"   → {filename}")


def main():
    """Main function"""
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / "results" / "analyzed_data.csv"
    
    if not DATA_PATH.exists():
        DATA_PATH = BASE_DIR / "data" / "vignetten_nrw.csv"
    
    if not DATA_PATH.exists():
        print(f"Error: Data file not found")
        return
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    print(f"Loaded {len(df)} texts")
    
    stats = {}
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    visualizer = AdvancedVisualizer(df, stats, RESULTS_DIR / "figures")
    visualizer.plot_heatmap()
    visualizer.plot_autonomy_gap()
    visualizer.plot_pca()
    visualizer.plot_scatter_agency_inspiration()
    visualizer.plot_violin_comparison()
    
    print(f"\nAll figures saved to {RESULTS_DIR / 'figures'}")


if __name__ == "__main__":
    main()
