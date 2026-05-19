import spacy
import pandas as pd
from tqdm import tqdm
from config import (
    PROMPT_NOISE_FILTER,
    INSPIRATION_TERMS_BASE, MEDICAL_TERMS_BASE,
    ADMIN_TERMS_BASE, ADMIN_MULTIWORD_PHRASES, HELPER_TERMS_BASE,
    AGENCY_WEIGHTS, AGENCY_VERBS_BASE, AGENCY_NOUNS, AGENCY_ADJECTIVES,
)

def normalize_umlauts(text: str) -> str:
    return (text
            .replace('\u00e4', 'ae')
            .replace('\u00f6', 'oe')
            .replace('\u00fc', 'ue')
            .replace('\u00df', 'ss')
            .replace('\u00c4', 'Ae')
            .replace('\u00d6', 'Oe')
            .replace('\u00dc', 'Ue'))

class ContextSensitiveAnalyzer:

    def __init__(self):
        self.noise_filter = {normalize_umlauts(t) for t in PROMPT_NOISE_FILTER}

        try:
            self.nlp = spacy.load('de_core_news_sm')
            print("SpaCy-Modell 'de_core_news_sm' geladen.")
        except OSError:
            raise OSError(
                "SpaCy-Modell nicht gefunden. Installieren mit:\n"
                "  python -m spacy download de_core_news_sm"
            )

        self.inspiration_terms = {normalize_umlauts(t) for t in INSPIRATION_TERMS_BASE}
        self.medical_terms     = {normalize_umlauts(t) for t in MEDICAL_TERMS_BASE}
        self.admin_terms       = {normalize_umlauts(t) for t in ADMIN_TERMS_BASE}
        self.helper_terms      = {normalize_umlauts(t) for t in HELPER_TERMS_BASE}
        self.agency_lexicon    = (
            {normalize_umlauts(t) for t in AGENCY_VERBS_BASE}
            | {normalize_umlauts(t) for t in AGENCY_NOUNS}
            | {normalize_umlauts(t) for t in AGENCY_ADJECTIVES}
        )
        self.agency_weights = {
            normalize_umlauts(k): v for k, v in AGENCY_WEIGHTS.items()
        }
        self.admin_phrases = [p.lower() for p in ADMIN_MULTIWORD_PHRASES]

    def score_text(self, text: str) -> dict:
        if not isinstance(text, str) or len(text.strip()) < 20:
            return {k: 0.0 for k in
                    ['medicalization', 'inspiration', 'agency',
                     'admin', 'shadow_helper', 'nrw_context']}

        text_lower = text.lower()
        doc = self.nlp(text_lower)

        scores = {
            'medicalization': 0.0, 'inspiration': 0.0, 'agency': 0.0,
            'admin': 0.0, 'shadow_helper': 0.0, 'nrw_context': 0.0,
        }

        for phrase in self.admin_phrases:
            if phrase in text_lower:
                scores['admin'] += 1.5

        valid_tokens = 0
        for token in doc:
            token_lemma = normalize_umlauts(token.lemma_.lower())

            if token_lemma in self.noise_filter:
                continue

            valid_tokens += 1

            weight = 1.0
            if (token.dep_ == 'neg' or
                    any(anc.lemma_ in ('nicht', 'kein', 'nie')
                        for anc in token.ancestors)):
                weight = -1.0

            if token_lemma in self.inspiration_terms:
                scores['inspiration'] += weight
            elif token_lemma in self.medical_terms:
                scores['medicalization'] += weight
            elif token_lemma in self.admin_terms:
                scores['admin'] += weight
            elif token_lemma in self.helper_terms:
                scores['shadow_helper'] += weight
            elif token_lemma in self.agency_lexicon:
                scores['agency'] += (
                    self.agency_weights.get(token_lemma, 1.0) * weight
                )

        norm = max(valid_tokens, 10)
        for k in scores:
            scores[k] = round(scores[k] / norm, 4)

        return scores

    def analyze_corpus(self, df: 'pd.DataFrame', text_column: str = 'text') -> 'pd.DataFrame':
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
            scores = self.score_text(row.get(text_column, ''))
            scores.update({
                'model':        row.get('model', 'unknown'),
                'condition':    row.get('condition', 'unknown'),
                'marker':       row.get('marker', ''),
                'prompt_id':    row.get('prompt_id', idx),
                'original_idx': idx,
            })
            results.append(scores)
        return pd.DataFrame(results)

def main():
    from pathlib import Path
    from config import DATA_PATH, RESULTS_DIR

    if not DATA_PATH.exists():
        print(f"Fehler: Datei nicht gefunden: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    print(f"{len(df)} Vignetten geladen.")

    analyzer = ContextSensitiveAnalyzer()
    results  = analyzer.analyze_corpus(df)
    print(f"\nAnalyse abgeschlossen: {len(results)} Texte.")
    print(results.head())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "analyzed_data.csv", index=False, encoding='utf-8')
    print(f"\nErgebnisse gespeichert: {RESULTS_DIR / 'analyzed_data.csv'}")

if __name__ == "__main__":
    main()

