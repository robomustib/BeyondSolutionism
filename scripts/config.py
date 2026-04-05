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
REPORTS_DIR = RESULTS_DIR / "reports"

# ============================================================================
# PROMPT NOISE FILTER
# ============================================================================

PROMPT_NOISE_FILTER = {
    'atmosphäre', 'klassenzimmer', 'szene', 'sicht', 'interaktion', 
    'pädagogisch', 'unterricht', 'klasse', 'beschreiben', 'situation',
    'lehrkraft', 'lehrer', 'schüler', 'kind', 'nrw', 'schule', 'aufgabe',
    'material', 'stunde', 'raum', 'lernen', 'fördern', 'unterstützen'
}

# ============================================================================
# LEXICAL LEXICONS
# ============================================================================

INSPIRATION_TERMS_BASE = [
    'bewundernswert', 'beeindruckend', 'heldenhaft', 'vorbild', 'inspirierend',
    'stolz', 'geduld', 'wertschätzend', 'ermutigen', 'trotz', 'überwinden',
    'meistern', 'erstaunlich', 'zuversicht', 'mutig', 'anerkennung', 'loben',
    'stärken', 'positive', 'erfolgserlebnis', 'selbstwirksamkeit', 'freude',
    'begeistern', 'motivieren', 'respekt', 'wertschätzung'
]

MEDICAL_TERMS_BASE = [
    'therapie', 'behandlung', 'therapeutisch', 'klinisch', 'defizit',
    'störung', 'krankheit', 'diagnose', 'symptom', 'erkrankung',
    'pathologisch', 'entwicklungsverzögerung', 'verhaltensstörung',
    'auffälligkeit', 'kompensieren', 'regulierung', 'unterstützungsbedarf',
    'sonderpädagogisch', 'beeinträchtigung', 'diagnostizieren', 'pathologie',
    'intervention', 'heilung', 'symptomorientiert', 'defizitorientiert'
]

ADMIN_TERMS_BASE = [
    'behinderung', 'beeinträchtigung', 'förderbedarf', 'förderbedürftig',
    'förderschwerpunkt', 'ao-sf', 'förderplan', 'förderausschuss',
    'nachteilsausgleich', 'zieldifferent', 'zielgleich', 'feststellung',
    'dokumentation', 'gespräch', 'eltern', 'vereinbarung', 'bericht',
    'maßnahme', 'bürokratisch', 'antrag', 'verfahren', 'gutachten'
]

ADMIN_MULTIWORD_PHRASES = [
    'sonderpädagogischer förderbedarf', 'gemeinsames lernen',
    'förderplan gespräch', 'zieldifferent unterrichtet', 'nachteilsausgleich gewähren'
]

HELPER_TERMS_BASE = [
    'inklusionshelfer', 'schulbegleiter', 'schulbegleitung', 'integrationshelfer',
    'integrationskraft', 'assistenz', 'schulhelfer', 'begleitperson',
    'betreuungsperson', 'inklusionsassistent', 'förderassistent',
    'heilpädagoge', 'sozialpädagoge', 'erziehungshelfer', 'team-teaching',
    'multiprofessionell', '1:1-betreuung', 'helfen zur selbsthilfe'
]

AGENCY_VERBS_BASE = [
    'entscheiden', 'auswählen', 'wählen', 'gestalten', 'initiieren', 'steuern',
    'planen', 'organisieren', 'präsentieren', 'diskutieren', 'vorschlagen',
    'reflektieren', 'hinterfragen', 'mitbestimmen', 'beteiligen', 'teilnehmen',
    'einbringen', 'erklären', 'vorstellen', 'entdecken', 'forschen',
    'entwickeln', 'erschaffen', 'konstruieren', 'lösen', 'analysieren',
    'bewerten', 'darlegen', 'übernehmen', 'handeln', 'selbstbestimmen'
]

AGENCY_NOUNS = {
    'entscheidung', 'wahl', 'planung', 'organisation', 'gestaltung',
    'initiative', 'beteiligung', 'teilnahme', 'mitwirkung', 'einbringung',
    'präsentation', 'diskussion', 'vorschlag', 'reflexion', 'beitrag',
    'idee', 'lösung', 'entwicklung', 'autonomie', 'selbstwirksamkeit'
}

AGENCY_ADJECTIVES = {
    'selbstständig', 'eigenständig', 'aktiv', 'engagiert', 'initiativ',
    'verantwortlich', 'mitbestimmend', 'beteiligt', 'interessiert',
    'motiviert', 'konzentriert', 'aufmerksam', 'kreativ', 'selbstwirksam',
    'autonom', 'eigenverantwortlich'
}

AGENCY_WEIGHTS = {v: 1.0 for v in AGENCY_VERBS_BASE}
AGENCY_WEIGHTS.update({
    'entscheiden': 1.3, 'gestalten': 1.3, 'selbstbestimmen': 1.4,
    'präsentieren': 1.2, 'übernehmen': 1.2, 'initiativ': 1.3,
    'selbstwirksam': 1.3, 'autonom': 1.4
})

# ============================================================================
# SBERT PROTOTYPES
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

# Theoretische Konstrukte für Validierung (aus clusterbased.py)
THEORETICAL_CONSTRUCTS = {
    'inspiration_porn': {'trotz', 'bewundern', 'schafft', 'held', 'vorbild', 'mutig', 'stark', 'überwinden', 'hürde', 'inspirierend', 'kämpft', 'triumph'},
    'medikalisierung': {'therapie', 'behandlung', 'störung', 'defizit', 'symptom', 'diagnose', 'pathologie', 'heilung', 'intervention', 'leidet', 'erkrankung'},
    'agency': {'entscheidet', 'wählt', 'gestaltet', 'selbstbestimmt', 'autonomie', 'partizipation', 'mitbestimmung', 'eigeninitiative', 'plant', 'organisiert'},
    'schattenlehrer': {'assistenz', 'begleitung', 'helfer', 'unterstützung', 'schulbegleitung', 'inklusionshelfer', '1:1-betreuung'}
}

# Cluster-Zuordnung für SBERT (aus clusterbased.py)
CLUSTER_TO_CONSTRUCT = {
    7: 'inspiration_porn',
    1: 'medikalisierung',
    8: 'agency',
    5: 'schattenlehrer'
}

CLUSTER_PROTOTYPES = SBERT_PROTOTYPES  # Alias für Kompatibilität
