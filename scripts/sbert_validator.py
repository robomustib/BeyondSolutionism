import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from scipy.stats import pointbiserialr, spearmanr
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

# Import config
from config import SBERT_PROTOTYPES, THEORETICAL_CONSTRUCTS, CLUSTER_TO_CONSTRUCT


class SBERTValidator:    
    def __init__(self):
        self.model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        self.model = None
        self.prototypes = SBERT_PROTOTYPES
        self.theoretical_constructs = THEORETICAL_CONSTRUCTS
        self.cluster_to_construct = CLUSTER_TO_CONSTRUCT
    
    def load_model(self):
        """Load SBERT model"""
        if self.model is None:
            print(f"   Loading {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"   Model loaded (embedding dimension: {self.model.get_sentence_embedding_dimension()})")
        return self.model
    
    def validate(self, df, text_column='text'):
        """Run construct validation on all texts (vollständig wie clusterbased.py)"""
        self.load_model()
        
        # Encode all texts
        print("   Encoding texts...")
        texts = df[text_column].tolist()
        text_embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
        print(f"   Encoded {len(texts)} texts")
        
        results = {}
        condition_binary = (df['condition'] == 'disability').astype(int)
        
        for construct, prototypes in self.prototypes.items():
            print(f"\n   Validating construct: {construct}")
            
            proto_embeddings = self.model.encode(prototypes, convert_to_tensor=True)
            similarities = util.cos_sim(text_embeddings, proto_embeddings).cpu().numpy()
            mean_similarities = np.mean(similarities, axis=1)
            max_similarities = np.max(similarities, axis=1)
            median_similarities = np.median(similarities, axis=1)
            
            # Convergence validity
            r_cond, p_cond = pointbiserialr(condition_binary, mean_similarities)
            
            # Cohen's d from r
            cohens_d = 2 * r_cond / np.sqrt(1 - r_cond**2) if abs(r_cond) < 0.99 else 0
            
            results[construct] = {
                'r_cond': float(r_cond),
                'p_cond': float(p_cond),
                'cohens_d': float(cohens_d),
                'mean_similarity': float(np.mean(mean_similarities)),
                'std_similarity': float(np.std(mean_similarities)),
                'mean_max': float(np.mean(max_similarities)),
                'mean_median': float(np.mean(median_similarities)),
                'significant': p_cond < 0.05
            }
            
            # Interpretation
            if p_cond < 0.05 and cohens_d > 0.2:
                if cohens_d > 0.8:
                    interp = "Valid (large effect)"
                elif cohens_d > 0.5:
                    interp = "Valid (medium effect)"
                else:
                    interp = "Valid (small effect)"
            else:
                interp = "Not valid"
            
            print(f"      r_cond = {r_cond:.3f}, p = {p_cond:.4f}, d = {cohens_d:.2f} → {interp}")
        
        return results
    
    def calculate_discriminant_validity(self, df, text_column='text'):
        """Calculate discriminant validity (inter-construct correlations)"""
        self.load_model()
        
        texts = df[text_column].tolist()
        text_embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
        
        construct_scores = {}
        for construct, prototypes in self.prototypes.items():
            proto_embeddings = self.model.encode(prototypes, convert_to_tensor=True)
            similarities = util.cos_sim(text_embeddings, proto_embeddings).cpu().numpy()
            construct_scores[construct] = np.mean(similarities, axis=1)
        
        df_scores = pd.DataFrame(construct_scores)
        corr_matrix = df_scores.corr(method='spearman')
        
        return corr_matrix
    
    def calculate_silhouette_score(self, df, text_column='text'):
        """Calculate silhouette score for construct separability"""
        self.load_model()
        
        texts = df[text_column].tolist()
        text_embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
        
        construct_scores = {}
        for construct, prototypes in self.prototypes.items():
            proto_embeddings = self.model.encode(prototypes, convert_to_tensor=True)
            similarities = util.cos_sim(text_embeddings, proto_embeddings).cpu().numpy()
            construct_scores[construct] = np.mean(similarities, axis=1)
        
        df_scores = pd.DataFrame(construct_scores)
        feature_matrix = df_scores.values
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        labels = np.argmax(feature_matrix_scaled, axis=1)
        
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(feature_matrix_scaled, labels, metric='cosine')
        else:
            sil_score = -1.0
        
        return sil_score
    
    def create_validation_summary(self, df, text_column='text'):
        """Create complete validation summary like clusterbased.py"""
        results = self.validate(df, text_column)
        discriminant = self.calculate_discriminant_validity(df, text_column)
        silhouette = self.calculate_silhouette_score(df, text_column)
        
        validation_results = []
        for construct, res in results.items():
            # Calculate discriminant correlations
            other_constructs = [c for c in results.keys() if c != construct]
            disc_corrs = {}
            for other in other_constructs:
                if other in discriminant.columns and construct in discriminant.index:
                    disc_corrs[other] = discriminant.loc[construct, other]
            max_discr = max(abs(v) for v in disc_corrs.values()) if disc_corrs else 0
            
            # Status
            if res['p_cond'] < 0.05 and abs(res['r_cond']) > 0.15 and max_discr < 0.7:
                status = "Valide"
            elif res['p_cond'] < 0.05 and abs(res['r_cond']) > 0.10:
                status = "Begrenzt valide"
            else:
                status = "Nicht valide"
            
            validation_results.append({
                'construct': construct,
                'corr_with_condition': res['r_cond'],
                'p_value_condition': res['p_cond'],
                'effect_size_cohens_d': res['cohens_d'],
                'max_discriminance': max_discr,
                'status': status
            })
        
        return {
            'validation_results': validation_results,
            'silhouette_score': silhouette,
            'discriminant_matrix': discriminant
        }


def main():
    """Main function"""
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / "data" / "vignetten_nrw.csv"
    
    if not DATA_PATH.exists():
        DATA_PATH = BASE_DIR / "results" / "analyzed_data.csv"
    
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    print(f"Loaded {len(df)} texts")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")
    
    validator = SBERTValidator()
    
    print("\n" + "=" * 60)
    print("SBERT CONSTRUCT VALIDATION")
    print("=" * 60)
    results = validator.validate(df)
    
    print("\n" + "=" * 60)
    print("DISCRIMINANT VALIDITY")
    print("=" * 60)
    corr_matrix = validator.calculate_discriminant_validity(df)
    print(corr_matrix.round(3))
    
    print("\n" + "=" * 60)
    print("SILHOUETTE SCORE")
    print("=" * 60)
    sil_score = validator.calculate_silhouette_score(df)
    print(f"Silhouette score: {sil_score:.3f}")
    
    # Summary table
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"\n{'Construct':<20} {'r_cond':<8} {'p_cond':<10} {'Cohen\'s d':<10} {'Status':<15}")
    print("-" * 65)
    for construct, res in results.items():
        status = "Valide" if res['significant'] and res['cohens_d'] > 0.2 else "Nicht valide"
        p_str = f"{res['p_cond']:.4f}"
        if res['p_cond'] < 0.001:
            p_str = "<.001"
        print(f"{construct:<20} {res['r_cond']:+.3f}   {p_str:<10} {res['cohens_d']:.2f}       {status:<15}")
    
    # Save results
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(RESULTS_DIR / "sbert_validation_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    corr_matrix.to_csv(RESULTS_DIR / "discriminant_validity_matrix.csv")
    
    print(f"\nResults saved to {RESULTS_DIR / 'sbert_validation_results.json'}")
    print(f"Correlation matrix saved to {RESULTS_DIR / 'discriminant_validity_matrix.csv'}")

if __name__ == "__main__":
    main()
