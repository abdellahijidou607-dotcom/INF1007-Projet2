"""Exceptions personnalisées du projet."""


class DonneeInvalideError(Exception):
    """Levée quand une donnée (saisie, colonne, métadonnée) est invalide."""


class FichierIntrouvableError(Exception):
    """Levée quand un fichier nécessaire au projet n'existe pas."""


class ModeleNonEntraineError(Exception):
    """Levée quand on demande une prédiction à un modèle ni entraîné ni chargé."""
