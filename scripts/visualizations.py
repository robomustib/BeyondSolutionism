import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready style
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")


class AdvancedVisualizer:
    """Generate all publication-ready visualizations"""
    
    def __init__(self, df, stats, output_dir):
        """
        Args:
            df: DataFrame with bias scores and metadata
            stats: Statistics results (for effect size annotations)
            output_dir: Directory to save figures
        """
        self.df = df
        self.stats = stats
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme (colorblind-friendly)
        self.colors = {
            'gpt': '#E69F00',      # Orange
            'gemini': '#56B4E9',   # Light blue
            'groq': '#009E73',     # Green
            'normative': '#A9A9A9', # Gray
            'disability': '#D55E00' # Red-orange
        }
        
        # Model labels for plotting
        self.model_labels = {
            'gpt': 'GPT-5.1',
            'gemini': 'Gemini Pro',
            'groq': 'Llama 3'
        }
    
    def plot_violin_comparison(self, filename='violin_comparison.png'):
        """
        Violin plots for bias dimensions by condition
        Corresponds to Figure 1 in the paper
        """
        metrics = ['medicalization_score', 'inspiration_score', 'agency_score']
        labels = ['Medikalisierung', 'Inspiration Porn', 'Agency']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            if metric not in self.df.columns:
                print(f"   Warning: {metric} not found in data")
                continue
            
            # Prepare data for plotting
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
            
            # Create violin plot
            sns.violinplot(
                data=df_plot, 
                x='model', 
                y='score', 
                hue='condition',
                split=True, 
                inner='quart', 
                palette='muted', 
                ax=axes[i],
                bw_method=0.3
            )
            
            # Add effect size annotation
            key = metric.replace('_score', '')
            if key in self.stats:
                d = self.stats[key]
                r_rb = d.get('effect_size_rrb', 0)
                p = d.get('p_value_fdr', 1)
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
                axes[i].set_title(f"{label}\nr_rb = {r_rb:.2f} {sig}", fontweight='bold')
            else:
                axes[i].set_title(label, fontweight='bold')
            
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Score (pro 100 Wörter)')
            axes[i].legend(title='Condition', loc='upper right')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"   → {filename}")
    
    def plot_autonomy_gap(self, filename='autonomy_gap.png'):
        """
        Boxplots showing autonomy gap across models
        Corresponds to Figure 2 in the paper
        """
        if 'agency_score' not in self.df.columns:
            print("   Warning: agency_score not found in data")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        models = ['gpt', 'gemini', 'groq']
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            model_data = self.df[self.df['model'] == model]
            
            norm_agency = model_data[model_data['condition'] == 'normative']['agency_score'].dropna()
            dis_agency = model_data[model_data['condition'] == 'disability']['agency_score'].dropna()
            
            if len(norm_agency) == 0 or len(dis_agency) == 0:
                ax.set_title(f'{self.model_labels.get(model, model)} (no data)')
                continue
            
            # Create boxplot
            bp = ax.boxplot(
                [norm_agency, dis_agency], 
                positions=[1, 2], 
                widths=0.6,
                patch_artist=True, 
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='gray'),
                capprops=dict(color='gray'),
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.3)
            )
            
            # Color the boxes
            bp['boxes'][0].set_facecolor(self.colors[model])
            bp['boxes'][1].set_facecolor(self.colors[model])
            bp['boxes'][0].set_alpha(0.4)  # Normative lighter
            bp['boxes'][1].set_alpha(0.8)  # Disability darker
            
            # Mark means
            m1, m2 = norm_agency.mean(), dis_agency.mean()
            ax.plot(1, m1, 'd', color='darkred', markersize=9, markeredgecolor='white')
            ax.plot(2, m2, 'd', color='darkred', markersize=9, markeredgecolor='white')
            
            # Percentage difference
            diff_pct = ((m2 - m1) / m1 * 100) if m1 > 0 else 0
            upper_y = max(norm_agency.max(), dis_agency.max()) + 0.05
            ax.text(
                1.5, upper_y, f'{diff_pct:+.1f}%', 
                ha='center', fontsize=12, weight='bold',
                color='darkred', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6)
            )
            
            # Significance test
            u, p = mannwhitneyu(norm_agency, dis_agency)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            ax.text(1.5, -0.12, f'p {sig}', ha='center', fontsize=10, weight='bold')
            
            # Labels
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['normativ', 'Disability'], fontsize=11)
            ax.set_title(self.model_labels.get(model, model), fontsize=14, weight='bold')
            ax.set_ylabel('Agency-Score' if idx == 0 else '', fontsize=12)
            ax.set_ylim(-0.15, 1.1)
            ax.grid(True, axis='y', alpha=0.2)
        
        plt.suptitle(
            'Die Autonomie-Lücke\nAgency in normativen vs. Disability-Texten', 
            fontsize=16, weight='bold', y=1.02
        )
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"   → {filename}")
    
    def plot_scatter_agency_inspiration(self, filename='scatter_agency_inspiration.png'):
        """
        Scatter plot: agency vs inspiration porn
        Shows the relationship between the two main constructs
        """
        if 'agency_score' not in self.df.columns or 'inspiration_score' not in self.df.columns:
            print("   Warning: agency_score or inspiration_score not found in data")
            return
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot by model with different markers for condition
        for model, color in self.colors.items():
            if model not in ['gpt', 'gemini', 'groq']:
                continue
            m_data = self.df[self.df['model'] == model]
            
            # Normative (circles)
            norm_mask = m_data['condition'] == 'normative'
            ax.scatter(
                m_data.loc[norm_mask, 'inspiration_score'],
                m_data.loc[norm_mask, 'agency_score'],
                c=color, marker='o', alpha=0.3, s=60, edgecolors='none'
            )
            
            # Disability (squares)
            dis_mask = m_data['condition'] == 'disability'
            ax.scatter(
                m_data.loc[dis_mask, 'inspiration_score'],
                m_data.loc[dis_mask, 'agency_score'],
                c=color, marker='s', alpha=0.7, s=80, edgecolors='w', 
                label=self.model_labels.get(model, model.upper())
            )
        
        # Add regression line
        rho, p = spearmanr(self.df['inspiration_score'], self.df['agency_score'])
        sns.regplot(
            data=self.df, 
            x='inspiration_score', 
            y='agency_score',
            scatter=False, 
            color='gray', 
            ax=ax, 
            line_kws={'linestyle': '--', 'alpha': 0.5}
        )
        
        # Labels and title
        ax.set_xlabel('Inspiration Porn (Score pro 100 Wörter)', weight='bold')
        ax.set_ylabel('Agency (Subjekt-Akteurschaft)', weight='bold')
        
        p_str = "< 0.001" if p < 0.001 else f"= {p:.3f}"
        ax.set_title(
            f'Zusammenhang Agency vs. Inspiration\nρ = {rho:.2f} (p {p_str})', 
            fontsize=14
        )
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Normativ', 
                   markerfacecolor='gray', markersize=8),
            Line2D([0], [0], marker='s', color='w', label='Disability', 
                   markerfacecolor='gray', markersize=8),
            Line2D([0], [0], color='none', label='---'),
            Line2D([0], [0], marker='s', color='w', label='GPT-5.1', 
                   markerfacecolor=self.colors['gpt']),
            Line2D([0], [0], marker='s', color='w', label='Gemini Pro', 
                   markerfacecolor=self.colors['gemini']),
            Line2D([0], [0], marker='s', color='w', label='Llama 3', 
                   markerfacecolor=self.colors['groq'])
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"   → {filename}")
    
    def plot_heatmap(self, filename='correlation_heatmap.png'):
        """
        Correlation heatmap of bias dimensions
        Corresponds to Figure 3 in the paper
        """
        metrics = [
            'medicalization_score', 'inspiration_score', 'agency_score', 
            'admin_score', 'word_count'
        ]
        available = [m for m in metrics if m in self.df.columns]
        
        if len(available) < 2:
            print("   Warning: Not enough metrics for heatmap")
            return
        
        # Calculate Spearman correlations
        corr_matrix = self.df[available].corr(method='spearman')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Labels for better readability
        label_map = {
            'medicalization_score': 'Medikalisierung',
            'inspiration_score': 'Inspiration Porn',
            'agency_score': 'Agency',
            'admin_score': 'Admin-Vokabular',
            'word_count': 'Textlänge'
        }
        labels = [label_map.get(c, c) for c in available]
        
        # Mask upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            annot=True, 
            fmt='.2f', 
            cmap='RdBu_r',
            center=0, 
            vmin=-0.6, 
            vmax=0.6, 
            square=True, 
            linewidths=.5,
            cbar_kws={'label': 'Spearman-ρ'}, 
            ax=ax,
            xticklabels=labels, 
            yticklabels=labels,
            annot_kws={'size': 11, 'weight': 'normal'}
        )
        
        # Add significance stars
        for i in range(len(available)):
            for j in range(i):
                rho, p = spearmanr(self.df[available[i]], self.df[available[j]])
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
                else:
                    sig = ''
                if sig:
                    ax.text(j + 0.5, i + 0.25, sig, 
                            ha='center', va='bottom', 
                            color='black', weight='bold', fontsize=12)
        
        ax.set_title('Spearman-Korrelationen der Bias-Dimensionen', fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"   → {filename}")


def main():
    """Main function for testing"""
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / "results" / "analyzed_data.csv"
    
    if not DATA_PATH.exists():
        DATA_PATH = BASE_DIR / "data" / "vignetten_nrw.csv"
    
    if not DATA_PATH.exists():
        print(f"Error: Data file not found")
        return
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    print(f"Loaded {len(df)} texts")
    
    # Create mock stats if not available (for testing)
    stats = {}
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    visualizer = AdvancedVisualizer(df, stats, RESULTS_DIR / "figures")
    visualizer.plot_violin_comparison()
    visualizer.plot_autonomy_gap()
    visualizer.plot_scatter_agency_inspiration()
    visualizer.plot_heatmap()
    
    print(f"\nAll figures saved to {RESULTS_DIR / 'figures'}")


if __name__ == "__main__":
    main()
