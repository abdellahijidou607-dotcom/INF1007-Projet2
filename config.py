"""Chemins des fichiers du projet, construits avec le module os."""
import os

from exceptions import FichierIntrouvableError

DOSSIER_PROJET = os.path.dirname(os.path.abspath(__file__))
DOSSIER_DONNEES = os.path.join(DOSSIER_PROJET, "donnees")
DOSSIER_MODELES = os.path.join(DOSSIER_PROJET, "modeles")

CHEMIN_CSV = os.path.join(DOSSIER_DONNEES, "restaurant_survival.csv")
CHEMIN_MODELE = os.path.join(DOSSIER_MODELES, "modele.joblib")
CHEMIN_META = os.path.join(DOSSIER_MODELES, "metadonnees.json")
CHEMIN_RESULTATS = os.path.join(DOSSIER_MODELES, "resultats_evaluation.json")
CHEMIN_HISTORIQUE = os.path.join(DOSSIER_PROJET, "historique_predictions.csv")
CHEMIN_GRAPHIQUE = os.path.join(DOSSIER_PROJET, "graphique_survie.png")


def verifier_fichier(chemin: str) -> None:
    """Vérifie qu'un fichier existe avant de l'ouvrir.

    Reçoit : le chemin du fichier.
    Renvoie : rien.
    Échoue : lève FichierIntrouvableError si le fichier n'existe pas.
    """
    if not os.path.isfile(chemin):
        raise FichierIntrouvableError(f"Fichier introuvable : {chemin}")
