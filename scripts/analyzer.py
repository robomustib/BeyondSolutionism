import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import (
    PROMPT_NOISE_FILTER, INSPIRATION_TERMS_BASE, MEDICAL_TERMS_BASE,
    ADMIN_TERMS_BASE, ADMIN_MULTIWORD_PHRASES, HELPER_TERMS_BASE,
    AGENCY_WEIGHTS, AGENCY_VERBS_BASE, AGENCY_NOUNS, AGENCY_ADJECTIVES
)


def normalize_umlauts(text):
    """Replace German umlauts"""
    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
        'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue'
    }
    for umlaut, replacement in replacements.items():
        text = text.replace(umlaut, replacement)
    return text


class ContextSensitiveAnalyzer:
    """Analyses texts with lexical lexicons and syntactic negation detection"""
    
    def __init__(self):
        self.noise_filter = PROMPT_NOISE_FILTER
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('de_core_news_sm')
            print("SpaCy model 'de_core_news_sm' loaded")
        except OSError:
            raise OSError("SpaCy model not found. Install with: python -m spacy download de_core_news_sm")
        
        # Initialize lexicons
        self.inspiration_terms = set(INSPIRATION_TERMS_BASE)
        self.medical_terms = set(MEDICAL_TERMS_BASE)
        self.admin_terms = set(ADMIN_TERMS_BASE)
        self.helper_terms = set(HELPER_TERMS_BASE)
        self.agency_weights = AGENCY_WEIGHTS
        self.agency_lexicon = set(AGENCY_VERBS_BASE) | AGENCY_NOUNS | AGENCY_ADJECTIVES
        self.admin_phrases = ADMIN_MULTIWORD_PHRASES
    
    def score_text(self, text):
        """Analyze a single text and return bias scores"""
        if not isinstance(text, str) or len(text.strip()) < 20:
            return {k: 0.0 for k in ['medicalization', 'inspiration', 'agency', 'admin', 'shadow_helper']}
        
        text_lower = text.lower()
        doc = self.nlp(text_lower)
        
        scores = {
            'medicalization': 0.0,
            'inspiration': 0.0,
            'agency': 0.0,
            'admin': 0.0,
            'shadow_helper': 0.0
        }
        
        # Multi-word admin phrases
        for phrase in self.admin_phrases:
            if phrase in text_lower:
                scores['admin'] += 1.5
        
        # Token-level analysis
        valid_tokens = 0
        for token in doc:
            token_lemma = normalize_umlauts(token.lemma_.lower())
            
            # Skip prompt noise
            if token_lemma in self.noise_filter:
                continue
            
            valid_tokens += 1
            
            # Negation detection (wie in v21_5)
            weight = 1.0
            if token.dep_ == 'neg' or any(anc.lemma_ in ('nicht', 'kein', 'nie') for anc in token.ancestors):
                weight = -1.0
            
            # Lexicon matching
            if token_lemma in self.inspiration_terms:
                scores['inspiration'] += weight
            elif token_lemma in self.medical_terms:
                scores['medicalization'] += weight
            elif token_lemma in self.admin_terms:
                scores['admin'] += weight
            elif token_lemma in self.helper_terms:
                scores['shadow_helper'] += weight
            elif token_lemma in self.agency_lexicon:
                scores['agency'] += self.agency_weights.get(token_lemma, 1.0) * weight
        
        # Length normalization
        norm = max(valid_tokens, 10)
        for k in scores:
            scores[k] = round(scores[k] / norm, 4)
        
        return scores
    
    def analyze_corpus(self, df, text_column='text'):
        """Analyze all texts in dataframe"""
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring texts"):
            scores = self.score_text(row.get(text_column, ''))
            scores.update({
                'model': row.get('model', 'unknown'),
                'condition': row.get('condition', 'unknown'),
                'marker': row.get('marker', ''),
                'prompt_id': row.get('prompt_id', idx),
                'original_idx': idx
            })
            results.append(scores)
        return pd.DataFrame(results)


def main():
    """Test function"""
    from pathlib import Path
    
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = BASE_DIR / "data" / "vignetten_nrw.csv"
    
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    print(f"Loaded {len(df)} vignettes")
    
    analyzer = ContextSensitiveAnalyzer()
    results = analyzer.analyze_corpus(df)
    print(f"\nAnalysis complete: {len(results)} texts analyzed")
    print(results.head())
    
    # Save results
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "analyzed_data.csv", index=False, encoding='utf-8')
    print(f"\nResults saved to {RESULTS_DIR / 'analyzed_data.csv'}")


if __name__ == "__main__":
    main()
