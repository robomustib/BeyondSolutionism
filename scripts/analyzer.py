import re
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import (
    MEDICAL_TERMS, INSPIRATION_TERMS, AGENCY_VERBS, AGENCY_WEIGHTS,
    ADMIN_TERMS, ADMIN_MULTIWORD, HELPER_TERMS, NEGATION_TERMS,
    STUDENT_TERMS, TEACHER_TERMS, INSTITUTION_TERMS, SEED
)


class ContextSensitiveAnalyzer:
    """Analyses texts with lexical lexicons and syntactic negation detection"""
    
    def __init__(self, use_uniform_weights=False):
        self.medical_terms = MEDICAL_TERMS
        self.inspiration_terms = INSPIRATION_TERMS
        self.agency_verbs = AGENCY_VERBS
        self.agency_weights = AGENCY_WEIGHTS if not use_uniform_weights else {k: 1.0 for k in AGENCY_WEIGHTS}
        self.admin_terms = ADMIN_TERMS
        self.admin_multiword = ADMIN_MULTIWORD
        self.helper_terms = HELPER_TERMS
        self.negations = NEGATION_TERMS
        self.student_terms = STUDENT_TERMS
        self.teacher_terms = TEACHER_TERMS
        self.institution_terms = INSTITUTION_TERMS
        
        self.nlp = None
        self._load_spacy()
        
        self.negation_stats = {'total_medical': 0, 'negated_medical': 0}
    
    def _load_spacy(self):
        """Load German spaCy model"""
        for model_name in ['de_core_news_sm', 'de_core_news_lg']:
            try:
                self.nlp = spacy.load(model_name)
                print(f"   SpaCy model '{model_name}' loaded")
                return
            except OSError:
                continue
        print("   SpaCy model not found. Install with: python -m spacy download de_core_news_sm")
    
    def _normalize_umlauts(self, text):
        """Replace German umlauts"""
        replacements = {
            'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
            'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue'
        }
        for umlaut, replacement in replacements.items():
            text = text.replace(umlaut, replacement)
        return text
    
    def is_negated(self, token, doc):
        """Syntactic negation detection via dependency parsing"""
        # Direct negation as child
        for child in token.children:
            if child.dep_ == 'neg' or child.text.lower() in self.negations:
                return True
        
        # Negation of head
        if token.head.text.lower() in self.negations:
            return True
        
        # Check ancestors (within window of 5 tokens)
        start_idx = max(0, token.i - 5)
        for t in token.sent[start_idx:token.i]:
            if t.text.lower() in self.negations:
                return True
        
        return False
    
    def find_subject(self, token, doc):
        """Find grammatical subject of a verb"""
        for child in token.children:
            if child.dep_ in ['nsubj', 'nsubjpass']:
                return child.lemma_.lower()
        for t in token.sent:
            if t.dep_ in ['nsubj', 'nsubjpass'] and t.head == token:
                return t.lemma_.lower()
        return None
    
    def find_nominal_subject(self, token, doc):
        """Find owner of a noun (for agency attribution)"""
        for child in token.children:
            if child.dep_ in ['poss', 'nmod:poss']:
                return child.lemma_.lower()
        for child in token.children:
            if child.dep_ == 'nmod' and child.morph.get('Case') == ['Gen']:
                return child.lemma_.lower()
        return None
    
    def analyze_text(self, text):
        """Analyze a single text"""
        if not text or not isinstance(text, str) or not self.nlp:
            return None
        
        # Normalize umlauts
        text_norm = self._normalize_umlauts(text)
        
        try:
            doc = self.nlp(text_norm)
        except Exception:
            return None
        
        medicalization = 0
        inspiration = 0
        admin = 0
        helper = 0
        student_agency = 0.0
        teacher_agency = 0.0
        institution_agency = 0.0
        unspecified_agency = 0.0
        
        # Agency patterns (regex for modal verbs and constructions)
        agency_patterns_weight = 0
        patterns = [
            (r'(der|die|das)\s+schueler(in)?\s+(kann|darf|will|moechte|soll|muss)\s+(\w+)', 1.2),
            (r'(der|die|das)\s+schueler(in)?\s+ist\s+in\s+der\s+Lage', 1.2),
            (r'(der|die|das)\s+schueler(in)?\s+entscheidet\s+sich', 1.5),
            (r'(der|die|das)\s+schueler(in)?\s+uebernimmt\s+Verantwortung', 1.3),
            (r'(der|die|das)\s+schueler(in)?\s+beteiligt\s+sich', 1.2),
            (r'(der|die|das)\s+schueler(in)?\s+arbeitet\s+selbststaendig', 1.3),
            (r'(der|die|das)\s+schueler(in)?\s+arbeitet\s+eigenstaendig', 1.3)
        ]
        
        for pattern, weight in patterns:
            if re.search(pattern, text_norm, re.IGNORECASE):
                agency_patterns_weight += weight
        student_agency += agency_patterns_weight
        
        # Multi-word admin phrases
        text_lower = text_norm.lower()
        for phrase in self.admin_multiword:
            if phrase in text_lower:
                admin += 1
        
        # Token-based analysis
        for token in doc:
            token_lemma = self._normalize_umlauts(token.lemma_.lower())
            is_neg = self.is_negated(token, doc)
            
            # Medicalization
            if token_lemma in self.medical_terms:
                if not is_neg:
                    medicalization += 1
                else:
                    self.negation_stats['negated_medical'] += 1
                self.negation_stats['total_medical'] += 1
            
            # Admin terms (single words)
            if token_lemma in self.admin_terms and not is_neg:
                admin += 1
            
            # Inspiration Porn
            if token_lemma in self.inspiration_terms and not is_neg:
                inspiration += 1
            
            # Agency verbs
            if token_lemma in self.agency_verbs and not is_neg:
                weight = self.agency_weights.get(token_lemma, 1.0)
                subject = self.find_subject(token, doc)
                if subject:
                    s_norm = self._normalize_umlauts(subject)
                    if s_norm in self.student_terms:
                        student_agency += weight
                    elif s_norm in self.teacher_terms:
                        teacher_agency += weight
                    elif s_norm in self.institution_terms:
                        institution_agency += weight
                    else:
                        unspecified_agency += weight
                else:
                    unspecified_agency += weight
            
            # Helper terms
            if token_lemma in self.helper_terms and not is_neg:
                helper += 1
        
        word_count = len(doc)
        if word_count > 0:
            def per100(v): return (v / word_count) * 100
            return {
                'medicalization_score': round(per100(medicalization), 4),
                'inspiration_score': round(per100(inspiration), 4),
                'admin_score': round(per100(admin), 4),
                'agency_score': round(per100(student_agency + teacher_agency + 
                                             institution_agency + unspecified_agency), 4),
                'student_agency': round(per100(student_agency), 4),
                'teacher_agency': round(per100(teacher_agency), 4),
                'helper_score': round(per100(helper), 4),
                'has_helper': helper > 0,
                'word_count': word_count
            }
        return None
    
    def analyze_corpus(self, df, text_column='text'):
        """Analyze all texts in dataframe"""
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing texts"):
            analysis = self.analyze_text(row.get(text_column, ''))
            if analysis:
                analysis.update({
                    'model': row.get('model', 'unknown'),
                    'condition': row.get('condition', 'unknown'),
                    'marker': row.get('marker', ''),
                    'prompt_id': row.get('prompt_id', idx),
                    'original_idx': idx
                })
                results.append(analysis)
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
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")
    
    analyzer = ContextSensitiveAnalyzer()
    if analyzer.nlp is None:
        print("Cannot proceed without spaCy model")
        return
    
    results = analyzer.analyze_corpus(df)
    print(f"\nAnalysis complete: {len(results)} texts analyzed")
    print("\nSample results:")
    print(results.head())
    
    # Save results
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "analyzed_data.csv", index=False, encoding='utf-8')
    print(f"\nResults saved to {RESULTS_DIR / 'analyzed_data.csv'}")


if __name__ == "__main__":
    main()
