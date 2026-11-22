"""Évaluation du modèle et comparaison avec un modèle de référence."""
import json

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import donnees
from config import CHEMIN_RESULTATS
from metriques import exactitude_numpy, rappel_par_classe
from modele import ModeleSurvie

LIMITES = """Limites :
- Le modèle se trompe encore sur environ 1 restaurant sur 4.
- La régression logistique sépare les classes par une frontière linéaire :
  elle ne capte pas les interactions complexes entre variables.
- Les données semblent synthétiques : les résultats ne garantissent pas
  la même performance sur de vrais restaurants.
- Certaines variables (note moyenne, nombre d'avis) ne sont connues qu'après
  l'ouverture : le modèle ne sert pas à juger un projet avant son lancement."""


def creer_reference(X_train, y_train) -> DummyClassifier:
    """Modèle de référence : prédit toujours la classe la plus fréquente."""
    return DummyClassifier(strategy="most_frequent").fit(X_train, y_train)


def evaluer(y_vrai, y_pred) -> dict:
    """Évalue une série de prédictions.

    Reçoit : les classes réelles et prédites.
    Renvoie : un dict avec l'exactitude, la matrice de confusion et le rappel par classe.
    Échoue : DonneeInvalideError si les tailles diffèrent.
    """
    matrice = confusion_matrix(y_vrai, y_pred, labels=[0, 1])
    return {
        "exactitude": exactitude_numpy(y_vrai, y_pred),
        "matrice_confusion": matrice.tolist(),
        "rappel_par_classe": rappel_par_classe(matrice).round(4).tolist(),
    }


def formater(nom: str, resultats: dict) -> str:
    """Met en forme les résultats d'une évaluation pour l'affichage."""
    return (f"--- {nom} ---\n"
            f"Exactitude : {resultats['exactitude']:.4f}\n"
            f"Matrice de confusion (lignes = réel, colonnes = prédit) :\n"
            f"{np.array(resultats['matrice_confusion'])}\n"
            f"Rappel [ferme, survit] : {resultats['rappel_par_classe']}\n")


def lancer_evaluation(modele: ModeleSurvie | None = None) -> dict:
    """Évalue le modèle et la référence, sauvegarde les résultats en JSON.

    Reçoit : un modèle (sinon le modèle sauvegardé est chargé).
    Renvoie : le dict des résultats.
    Échoue : FichierIntrouvableError si le CSV ou le modèle manque.
    """
    X_train, X_test, y_train, y_test = donnees.obtenir_donnees_separees()
    if modele is None:
        modele = ModeleSurvie.charger()
    y_pred = modele.predire_lot(X_test)
    y_ref = creer_reference(X_train, y_train).predict(X_test)
    resultats = {"modele": evaluer(y_test, y_pred), "reference": evaluer(y_test, y_ref)}
    resultats["gain_exactitude"] = round(
        resultats["modele"]["exactitude"] - resultats["reference"]["exactitude"], 4)
    resultats["exactitude_sklearn"] = float(accuracy_score(y_test, y_pred))
    resultats["rapport"] = classification_report(y_test, y_pred, target_names=["ferme", "survit"])
    with open(CHEMIN_RESULTATS, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in resultats.items() if k != "rapport"}, f, indent=2)
    return resultats


if __name__ == "__main__":
    res = lancer_evaluation()
    print(formater("Régression logistique", res["modele"]))
    print(res["rapport"])
    print(formater("Référence (classe majoritaire)", res["reference"]))
    print(f"Exactitude NumPy : {res['modele']['exactitude']:.4f} "
          f"| scikit-learn : {res['exactitude_sklearn']:.4f}")
    print(f"Gain par rapport à la référence : +{res['gain_exactitude'] * 100:.1f} points\n")
    print(LIMITES)
