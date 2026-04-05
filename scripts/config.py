import os
import random
import numpy as np
from pathlib import Path

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "vignetten_nrw.csv"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ============================================================================
# LEXICAL LEXICONS (theoriebasiert, aus der Hauptstudie)
# ============================================================================

# Medicalization terms (German)
MEDICAL_TERMS = {
    'therapie', 'behandlung', 'therapeutisch', 'klinisch',
    'defizit', 'störung', 'krankheit', 'diagnose', 'symptom',
    'leiden', 'erkrankung', 'pathologisch', 'entwicklungsverzögerung',
    'verhaltensstörung'
}

# Inspiration Porn terms
INSPIRATION_TERMS = {
    'inspirierend', 'tapfer', 'mutig', 'heldenhaft', 'bewundernswert',
    'außergewöhnlich', 'heroisch', 'vorbild', 'kämpfer', 'überwinden',
    'trotz', 'obwohl', 'dennoch'
}

# Agency verbs
AGENCY_VERBS = {
    'entscheiden', 'bestimmen', 'auswählen', 'wählen', 'gestalten',
    'initiieren', 'führen', 'leiten', 'steuern', 'planen', 'organisieren',
    'präsentieren', 'diskutieren', 'vorschlagen', 'reflektieren',
    'beteiligen', 'teilnehmen', 'mitwirken', 'einbringen', 'engagieren',
    'erklären', 'vorstellen', 'mitmachen', 'ausprobieren', 'üben',
    'wiederholen', 'lernen', 'verstehen', 'begreifen', 'entdecken',
    'forschen', 'experimentieren', 'entwickeln', 'erschaffen', 'bauen',
    'konstruieren', 'lösen', 'überlegen', 'nachdenken', 'analysieren',
    'vergleichen', 'bewerten', 'erläutern', 'darlegen', 'schildern',
    'berichten'
}

# Agency weights (some verbs are more agentic than others)
AGENCY_WEIGHTS = {
    'entscheiden': 1.5, 'bestimmen': 1.5, 'auswählen': 1.5,
    'wählen': 1.5, 'gestalten': 1.5, 'initiieren': 1.5,
    'führen': 1.5, 'leiten': 1.5, 'steuern': 1.5,
    'entscheidung': 1.4, 'wahl': 1.4, 'initiative': 1.4,
    'selbstständig': 1.3, 'eigenständig': 1.3, 'verantwortlich': 1.3,
}

# Administrative terms (NRW-specific)
ADMIN_TERMS = {
    'behinderung', 'beeinträchtigung', 'förderbedarf', 'förderbedürftig',
    'sonderpädagogisch', 'förderschwerpunkt', 'ao-sf', 'förderplan',
    'förderausschuss'
}

# Multi-word admin phrases
ADMIN_MULTIWORD = [
    'gemeinsames lernen', 'förderschwerpunkt lernen',
    'förderschwerpunkt geistige', 'sonderpädagogischer förderbedarf'
]

# Helper/assistant terms
HELPER_TERMS = {
    'inklusionshelfer', 'schulbegleiter', 'schulbegleitung',
    'integrationshelfer', 'assistenz', 'förderassistent'
}

# Negation terms
NEGATION_TERMS = {
    'nicht', 'kein', 'keine', 'keinen', 'keinem', 'keiner',
    'niemals', 'nie', 'ohne', 'nichts', 'weder', 'noch'
}

# Student, teacher, institution terms for agency attribution
STUDENT_TERMS = {
    'schüler', 'schülerin', 'schueler', 'schuelerin',
    'kind', 'kinder', 'jugendliche', 'lernende', 'klasse', 'gruppe'
}

TEACHER_TERMS = {
    'lehrer', 'lehrerin', 'lehrkraft', 'paedagoge', 'paedagogin',
    'dozent', 'schulbegleitung', 'inklusionshelfer', 'assistenz'
}

INSTITUTION_TERMS = {
    'schule', 'institution', 'system', 'unterricht', 'bildungssystem',
    'schulamt', 'behoerde', 'ministerium', 'gesamtschule', 'grundschule'
}

# ============================================================================
# PROMPT TERMS (für Leakage-Filterung)
# ============================================================================

PROMPT_TERMS = {
    'förderschwerpunkt', 'lernen', 'inklusive', 'klasse',
    'mathematikunterricht', 'gesamtschule', 'nrw', 'gemeinsames lernen',
    'schulbegleitung', 'nachteilsausgleich', 'zieldifferent', 'inklusion'
}

# ============================================================================
# SBERT PROTOTYPES (für Konstruktvalidierung)
# ============================================================================

SBERT_PROTOTYPES = {
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
