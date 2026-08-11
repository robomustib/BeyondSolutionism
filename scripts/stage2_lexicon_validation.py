"""
STUFE 2 (FINAL - MIT PROMPT-FILTERUNG): EMPIRISCHE LEXIKON-EXTRAKTION
✅ Filtert alle Prompt-Begriffe aus der Analyse
✅ Verwendet klassische LogReg mit bereinigtem Text
✅ Validiert gegen theoretische Konstrukte
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.max_open_warning': 0, 'figure.figsize': (10, 6)})

# ============================================================================
# 1. KONFIGURATION
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR.parent / "data" / "vignetten_nrw.csv"
OUTPUT_DIR = SCRIPT_DIR.parent / "results" / "stage2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# VOLLSTÄNDIGE LISTE DER PROMPT-BEGRIFFE (aus Ihrem Generator-Skript)
# ============================================================================
PROMPT_TERMS = {
    # Aus den Szenarien (create_prompts)
    'mathematikunterricht', 'sportunterricht', 'projektpräsentation', 'gesamtschule',
    'grundschule', 'realschule', 'förderschule', 'inklusive klasse', 'inklusive',
    'gemeinsames lernen', 'förderplan', 'förderplan gespräch', 'eltern', 'lehrerin',
    'schulbegleitung', 'inklusionshelfer', 'assistive technologie', 'technologie',
    'nachteilsausgleich', 'klassenarbeit', 'arbeitsgruppe', 'kooperiert',
    
    # Aus den Förderschwerpunkten
    'förderschwerpunkt', 'lernen', 'emotionale und soziale entwicklung', 'geistige entwicklung',
    'autismus spektrum', 'rollstuhl', 'gehörlos', 'gebärdensprache',
    
    # Aus den System-Prompts
    'erfahrene lehrkraft', 'nrw', 'pädagogischer sicht', 'pädagogische interaktion',
    'atmosphäre im klassenzimmer', 'beschreibe die szene', 'beschreibe die situation',
    
    # Aus der Analyse (um Prompt-Leakage zu vermeiden)
    'atmosphäre', 'interaktion', 'klassenzimmer', 'szene', 'sicht', 'pädagogischer',
    'leon', 'du', 'sie', 'er', 'ihn', 'ihr', 'dem schüler', 'der schüler', 'den schüler',
    'mit dem', 'die atmosphäre', 'im klassenzimmer', 'pädagogische interaktion',
    'aus pädagogischer', 'pädagogischer sicht', 'die szene', 'schülerinnen', 'kind',
    
    # Formatierungsreste
    'beschreibe', 'beschreiben sie', 'schildere', 'beobachte'
}

# Theoretische Konstrukte
THEORETICAL_LEXICON = {
    'inspiration_porn': {'trotz', 'bewundern', 'schafft', 'held', 'vorbild', 'mutig', 'stark', 'überwinden', 'hürde', 'inspirierend', 'kämpft', 'triumph', 'bewunderung'},
    'medikalisierung': {'therapie', 'behandlung', 'störung', 'defizit', 'symptom', 'diagnose', 'pathologie', 'heilung', 'intervention', 'leidet', 'erkrankung'},
    'agency': {'entscheidet', 'wählt', 'gestaltet', 'selbstbestimmt', 'autonomie', 'partizipation', 'mitbestimmung', 'eigeninitiative', 'plant', 'organisiert', 'führt'},
    'schattenlehrer': {'assistenz', 'begleitung', 'helfer', 'unterstützung', 'schulbegleitung', 'inklusionshelfer', '1:1 betreuung'}
}

print("=" * 80)
print("STUFE 2 (FINAL): EMPIRISCHE LEXIKON-EXTRAKTION MIT PROMPT-FILTERUNG")
print("=" * 80)

# ============================================================================
# 2. DATEN LADEN
# ============================================================================
print(f"\n[1/5] Daten laden von: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, encoding='utf-8')
print(f"   → {len(df)} Texte geladen")
print(f"   → Conditions: {df['condition'].value_counts().to_dict()}")
print(f"   → Modelle: {df['model'].value_counts().to_dict()}")

# ============================================================================
# 3. TEXTBEREINIGUNG (Prompt-Entfernung)
# ============================================================================
print("\n[2/5] Textbereinigung (Entferne Prompt-Begriffe)...")

def clean_text(text):
    """Entfernt Prompt-Begriffe und reinigt Text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Entferne Prompt-Begriffe
    for term in PROMPT_TERMS:
        text = re.sub(r'\b' + re.escape(term) + r'\b', '', text, flags=re.IGNORECASE)
    # Entferne überflüssige Leerzeichen
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

df['text_clean'] = df['text'].apply(clean_text)
df = df[df['text_clean'].str.len() > 30].reset_index(drop=True)
print(f"   → {len(df)} Texte nach Bereinigung")

# Beispiel anzeigen
print(f"\n   Beispiel vorher:")
print(f"   {df['text'].iloc[0][:150]}...")
print(f"\n   Beispiel nachher:")
print(f"   {df['text_clean'].iloc[0][:150]}...")

# ============================================================================
# 4. TF-IDF VEKTORISIERUNG
# ============================================================================
print("\n[3/5] TF-IDF Vektorisierung...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=2000,
    min_df=3,
    max_df=0.85,
    sublinear_tf=True
)
X = vectorizer.fit_transform(df['text_clean'])
feature_names = vectorizer.get_feature_names_out()
y = (df['condition'] == 'disability').astype(int)

print(f"   → Shape: {X.shape}")
print(f"   → Features: {len(feature_names)}")

# ============================================================================
# 5. LOGISTISCHE REGRESSION MIT CROSS-VALIDATION
# ============================================================================
print("\n[4/5] Logistische Regression mit Cross-Validation...")

clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000, C=0.1)

# Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
print(f"   → ROC-AUC (CV): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Finales Modell
clf.fit(X, y)

# Holdout-Validierung
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
roc_holdout = roc_auc_score(y_test, y_pred)
print(f"   → ROC-AUC (Holdout): {roc_holdout:.3f}")

# ============================================================================
# 6. EMPIRISCHES LEXIKON
# ============================================================================
print("\n[5/5] Extrahiere empirisches Lexikon...")

coefs = clf.coef_[0]
top_dis_idx = np.argsort(coefs)[-30:][::-1]
top_norm_idx = np.argsort(coefs)[:30]

# Filtere Prompt-Begriffe noch einmal
def is_valid_term(term):
    term_lower = term.lower()
    return not any(pt in term_lower for pt in PROMPT_TERMS) and len(term) > 2

dis_words = [(feature_names[i], coefs[i]) for i in top_dis_idx if is_valid_term(feature_names[i])]
norm_words = [(feature_names[i], coefs[i]) for i in top_norm_idx if is_valid_term(feature_names[i])]

print(f"\n📊 TOP 20 DISABILITY-INDIKATOREN (nach Prompt-Filterung):")
for i, (word, coef) in enumerate(dis_words[:20], 1):
    print(f"   {i:2d}. '{word}' (coef={coef:.3f})")

print(f"\n📊 TOP 20 NORMATIVE-INDIKATOREN (nach Prompt-Filterung):")
for i, (word, coef) in enumerate(norm_words[:20], 1):
    print(f"   {i:2d}. '{word}' (coef={coef:.3f})")

# ============================================================================
# 7. VALIDIERUNG GEGEN THEORETISCHES LEXIKON
# ============================================================================
print("\n📊 Validierung gegen theoretische Konstrukte:")

empirical_set = set([w for w, _ in dis_words[:30]])
validation_results = []

for construct, theoretical_set in THEORETICAL_LEXICON.items():
    overlap = empirical_set.intersection(theoretical_set)
    precision = len(overlap) / len(empirical_set) if empirical_set else 0
    recall = len(overlap) / len(theoretical_set) if theoretical_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    validation_results.append({
        'construct': construct,
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1_score': round(f1, 3),
        'matching_words': ', '.join(overlap) if overlap else 'keine'
    })
    print(f"   {construct:<18} | F1={f1:.2f} | Overlap: {len(overlap)} | {', '.join(overlap) if overlap else '-'}")

# ============================================================================
# 8. VISUALISIERUNGEN & EXPORT
# ============================================================================
print("\n💾 Exportiere Ergebnisse...")

# Top-Features Plot
fig, ax = plt.subplots(figsize=(10, 6))
top_words = [w for w, _ in dis_words[:15]]
top_coefs = [c for _, c in dis_words[:15]]
colors = ['darkred' if c > 0 else 'darkblue' for c in top_coefs]
ax.barh(range(len(top_words)), top_coefs, color=colors)
ax.set_yticks(range(len(top_words)))
ax.set_yticklabels(top_words)
ax.set_xlabel('Logistic Regression Coefficient')
ax.set_title('Top 15 Empirische Disability-Indikatoren (nach Prompt-Filterung)')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_disability_features.png", dpi=150)
plt.close()

# F1-Plot
fig, ax = plt.subplots(figsize=(6, 4))
df_val = pd.DataFrame(validation_results)
bars = ax.bar(df_val['construct'], df_val['f1_score'], color='skyblue', edgecolor='black')
ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.6, label='Schwellenwert (F1=0.3)')
ax.set_ylabel('F1-Score')
ax.set_title('Konvergenzvalidität: Empirisches vs. Theoretisches Lexikon')
ax.set_ylim(0, 1)
ax.legend()
for bar, f1 in zip(bars, df_val['f1_score']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{f1:.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "construct_overlap_f1.png", dpi=150)
plt.close()

# ROC-Kurve
fig, ax = plt.subplots(figsize=(6, 5))
fpr, tpr, _ = roc_curve(y_test, y_pred)
ax.plot(fpr, tpr, label=f'LogReg (AUC={roc_holdout:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC-Kurve (nach Prompt-Filterung)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
plt.close()

# CSVs
pd.DataFrame({
    'rank': range(1, len(dis_words)+1),
    'word': [w for w, _ in dis_words],
    'coefficient': [c for _, c in dis_words]
}).to_csv(OUTPUT_DIR / "empirical_lexicon_disability.csv", index=False, encoding='utf-8')

pd.DataFrame({
    'rank': range(1, len(norm_words)+1),
    'word': [w for w, _ in norm_words],
    'coefficient': [c for _, c in norm_words]
}).to_csv(OUTPUT_DIR / "empirical_lexicon_normative.csv", index=False, encoding='utf-8')

df_val.to_csv(OUTPUT_DIR / "lexicon_overlap_report.csv", index=False, encoding='utf-8')

# ============================================================================
# 9. ZUSAMMENFASSUNG
# ============================================================================
print("\n" + "=" * 80)
print("📊 ZUSAMMENFASSUNG STUFE 2 (FINAL)")
print("=" * 80)
print(f"\n   ROC-AUC (Cross-Validation): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
print(f"   ROC-AUC (Holdout): {roc_holdout:.3f}")
print(f"   Empirisches Lexikon (Disability): {len(dis_words)} Wörter")
print(f"   Empirisches Lexikon (Normative): {len(norm_words)} Wörter")
print(f"\n   Prompt-Begriffe gefiltert: {len(PROMPT_TERMS)}")
print(f"\n   Validierung (F1-Scores):")
for _, row in df_val.iterrows():
    print(f"      {row['construct']}: {row['f1_score']:.2f}")
print(f"\n💾 Outputs in: {OUTPUT_DIR}")
print("   • empirical_lexicon_disability.csv")
print("   • empirical_lexicon_normative.csv")
print("   • lexicon_overlap_report.csv")
print("   • top_disability_features.png")
print("   • construct_overlap_f1.png")
print("   • roc_curve.png")
print("\n✅ STUFE 2 ABGESCHLOSSEN")
print("=" * 80)