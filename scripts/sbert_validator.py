import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from scipy.stats import pointbiserialr, spearmanr
import json
import warnings
warnings.filterwarnings('ignore')


class SBERTValidator:
    """SBERT-based construct validation for theoretical framing constructs"""
    
    def __init__(self):
        self.model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        self.model = None
        
        # Optimized prototype sentences per construct (theoretically distinct)
        self.prototypes = {
            'inspiration_porn': [
                "Trotz seiner Beeinträchtigung bewältigt er die Aufgabe mit bewundernswerter Stärke.",
                "Ihr unerschütterlicher Mut und ihre positive Einstellung inspirieren die gesamte Klasse.",
                "Er überwindet die Hürden mit einer Haltung, die alle zum Staunen bringt.",
                "Was für ein heldenhafter Einsatz gegen die Widrigkeiten des Alltags.",
                "Sein Wille ist wirklich beeindruckend – er gibt niemals auf."
            ],
            'medikalisierung': [
                "Die Intervention zielt auf die Reduktion der diagnostizierten Defizite ab.",
                "Therapeutische Maßnahmen werden nach symptomorientiertem Förderplan umgesetzt.",
                "Der sonderpädagogische Unterstützungsbedarf erfordert gezielte Behandlungsschritte.",
                "Auffälliges Verhalten wird als Symptom einer zugrundeliegenden Störung eingeordnet.",
                "Die Diagnose zeigt eine Entwicklungsverzögerung im kognitiven Bereich."
            ],
            'agency': [
                "Er wählt selbstständig die Strategie und begründet seine Entscheidung gegenüber der Klasse.",
                "Der Schüler gestaltet den Lernprozess aktiv nach eigenen Vorstellungen mit.",
                "Sie trifft die finale Auswahl des Themas und übernimmt die Verantwortung für das Ergebnis.",
                "Autonomie und Partizipation stehen im Zentrum der pädagogischen Begleitung.",
                "Er plant seinen Lernweg eigenständig und reflektiert seine Fortschritte."
            ],
            'schattenlehrer': [
                "Die Schulbegleitung interveniert nur bei Bedarf und zieht sich dann bewusst zurück.",
                "Die Assistenzkraft arbeitet nach dem Prinzip der Hilfe zur Selbsthilfe.",
                "Unterstützung erfolgt unauffällig im Hintergrund, um Abhängigkeit zu vermeiden.",
                "Die Rollenverteilung zwischen Lehrkraft und Inklusionshelfer ist klar und temporär begrenzt.",
                "Die Schulbegleitung agiert diskret und fördert die Eigenständigkeit des Schülers."
            ]
        }
    
    def load_model(self):
        """Load SBERT model (downloaded on first call)"""
        if self.model is None:
            print(f"   Loading {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"   Model loaded (embedding dimension: {self.model.get_sentence_embedding_dimension()})")
        return self.model
    
    def validate(self, df, text_column='text'):
        """
        Run construct validation on all texts
        
        Args:
            df: DataFrame with 'text' and 'condition' columns
            text_column: Name of column containing text to analyze
        
        Returns:
            dict: Validation results for each construct containing:
                - r_cond: Point-biserial correlation with condition
                - p_cond: p-value for correlation
                - cohens_d: Effect size (Cohen's d)
                - mean_similarity: Mean cosine similarity to prototypes
                - std_similarity: Standard deviation of similarities
        """
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
            
            # Encode prototypes
            proto_embeddings = self.model.encode(prototypes, convert_to_tensor=True)
            
            # Calculate cosine similarities
            similarities = util.cos_sim(text_embeddings, proto_embeddings).cpu().numpy()
            mean_similarities = np.mean(similarities, axis=1)
            
            # Convergence validity (correlation with condition)
            r_cond, p_cond = pointbiserialr(condition_binary, mean_similarities)
            
            # Cohen's d from r (conversion formula)
            cohens_d = 2 * r_cond / np.sqrt(1 - r_cond**2) if abs(r_cond) < 0.99 else 0
            
            results[construct] = {
                'r_cond': float(r_cond),
                'p_cond': float(p_cond),
                'cohens_d': float(cohens_d),
                'mean_similarity': float(np.mean(mean_similarities)),
                'std_similarity': float(np.std(mean_similarities)),
                'significant': p_cond < 0.05
            }
            
            # Interpretation
            if p_cond < 0.05 and cohens_d > 0.2:
                if cohens_d > 0.8:
                    interpretation = "Valid (large effect)"
                elif cohens_d > 0.5:
                    interpretation = "Valid (medium effect)"
                else:
                    interpretation = "Valid (small effect)"
            else:
                interpretation = "Not valid"
            
            print(f"      r_cond = {r_cond:.3f}, p = {p_cond:.4f}, d = {cohens_d:.2f} → {interpretation}")
        
        return results
    
    def calculate_discriminant_validity(self, df, text_column='text'):
        """
        Calculate discriminant validity (inter-construct correlations)
        
        Returns:
            DataFrame: Correlation matrix between constructs
        """
        self.load_model()
        
        # Encode all texts
        texts = df[text_column].tolist()
        text_embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
        
        # Calculate similarity for each construct
        construct_scores = {}
        for construct, prototypes in self.prototypes.items():
            proto_embeddings = self.model.encode(prototypes, convert_to_tensor=True)
            similarities = util.cos_sim(text_embeddings, proto_embeddings).cpu().numpy()
            construct_scores[construct] = np.mean(similarities, axis=1)
        
        # Correlation matrix
        df_scores = pd.DataFrame(construct_scores)
        corr_matrix = df_scores.corr(method='spearman')
        
        return corr_matrix
    
    def calculate_silhouette_score(self, df, text_column='text'):
        """
        Calculate silhouette score for construct separability
        
        Returns:
            float: Silhouette score (higher = better separation)
        """
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        
        self.load_model()
        
        # Encode all texts
        texts = df[text_column].tolist()
        text_embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
        
        # Calculate similarity for each construct
        construct_scores = {}
        for construct, prototypes in self.prototypes.items():
            proto_embeddings = self.model.encode(prototypes, convert_to_tensor=True)
            similarities = util.cos_sim(text_embeddings, proto_embeddings).cpu().numpy()
            construct_scores[construct] = np.mean(similarities, axis=1)
        
        # Create feature matrix
        df_scores = pd.DataFrame(construct_scores)
        feature_matrix = df_scores.values
        
        # Standardize
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Silhouette score (how well constructs separate)
        # We need labels - using the construct with highest score as "label" for each text
        labels = np.argmax(feature_matrix_scaled, axis=1)
        
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(feature_matrix_scaled, labels, metric='cosine')
        else:
            sil_score = -1.0
        
        return sil_score


def main():
    """Main function"""
    import sys
    from pathlib import Path
    
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / "data" / "vignetten_nrw.csv"
    
    if not DATA_PATH.exists():
        DATA_PATH = BASE_DIR / "results" / "analyzed_data.csv"
    
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure data/vignetten_nrw.csv exists")
        sys.exit(1)
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    print(f"Loaded {len(df)} texts")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")
    
    validator = SBERTValidator()
    
    # Run construct validation
    print("\n" + "=" * 60)
    print("SBERT CONSTRUCT VALIDATION")
    print("=" * 60)
    results = validator.validate(df)
    
    # Calculate discriminant validity
    print("\n" + "=" * 60)
    print("DISCRIMINANT VALIDITY (Inter-construct correlations)")
    print("=" * 60)
    corr_matrix = validator.calculate_discriminant_validity(df)
    print(corr_matrix.round(3))
    
    # Calculate silhouette score
    print("\n" + "=" * 60)
    print("SILHOUETTE SCORE (Construct separability)")
    print("=" * 60)
    sil_score = validator.calculate_silhouette_score(df)
    print(f"Silhouette score: {sil_score:.3f}")
    if sil_score > 0.25:
        print("   → Good separation between constructs")
    elif sil_score > 0.1:
        print("   → Moderate separation between constructs")
    else:
        print("   → Poor separation between constructs (needs prototype optimization)")
    
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
    
    # Save correlation matrix
    corr_matrix.to_csv(RESULTS_DIR / "discriminant_validity_matrix.csv")
    
    print(f"\nResults saved to {RESULTS_DIR / 'sbert_validation_results.json'}")
    print(f"Correlation matrix saved to {RESULTS_DIR / 'discriminant_validity_matrix.csv'}")


if __name__ == "__main__":
    main()
