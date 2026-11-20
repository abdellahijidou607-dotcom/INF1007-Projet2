"""Évaluation du modèle et comparaison avec un modèle de référence."""
import json

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import donnees
from modele import ModeleSurvie

CHEMIN_RESULTATS = "resultats_evaluation.json"

LIMITES = """Limites :
- Le modèle se trompe encore sur environ 1 restaurant sur 4.
- La régression logistique sépare les classes par une frontière linéaire :
  elle ne capte pas les interactions complexes entre variables.
- Les données semblent synthétiques : les résultats ne garantissent pas
  la même performance sur de vrais restaurants.
- Certaines variables (note moyenne, nombre d'avis) ne sont connues qu'après
  l'ouverture : le modèle ne sert pas à juger un projet avant son lancement."""


def exactitude_numpy(y_vrai, y_pred):
    """Exactitude calculée de façon vectorisée avec NumPy (sans boucle)."""
    return float(np.mean(np.asarray(y_vrai) == np.asarray(y_pred)))


def taux_par_classe(matrice):
    """Rappel de chaque classe à partir de la matrice de confusion (vectorisé) :
    diagonale / somme de chaque ligne."""
    matrice = np.asarray(matrice)
    return np.diag(matrice) / matrice.sum(axis=1)


def creer_reference(X_train, y_train):
    """Modèle de référence : prédit toujours la classe la plus fréquente."""
    return DummyClassifier(strategy="most_frequent").fit(X_train, y_train)


def evaluer(y_vrai, y_pred):
    """Retourne un dict avec l'exactitude, la matrice de confusion et le rappel par classe."""
    matrice = confusion_matrix(y_vrai, y_pred)
    return {
        "exactitude": float(accuracy_score(y_vrai, y_pred)),
        "matrice_confusion": matrice.tolist(),
        "rappel_par_classe": taux_par_classe(matrice).round(4).tolist(),
    }


def afficher(nom, resultats):
    print(f"--- {nom} ---")
    print(f"Exactitude : {resultats['exactitude']:.4f}")
    print("Matrice de confusion (lignes = réel, colonnes = prédit) :")
    print(np.array(resultats["matrice_confusion"]))
    print("Rappel [ferme, survit] :", resultats["rappel_par_classe"], "\n")


def lancer_evaluation(modele=None):
    """Évalue le modèle et la référence, sauvegarde les résultats en JSON
    (utilisables par visualisation.py) et les retourne."""
    X_train, X_test, y_train, y_test = donnees.obtenir_donnees_separees()
    if modele is None:
        modele = ModeleSurvie.charger()

    y_pred = modele.predire_lot(X_test)
    y_ref = creer_reference(X_train, y_train).predict(X_test)

    resultats = {
        "modele": evaluer(y_test, y_pred),
        "reference": evaluer(y_test, y_ref),
    }
    resultats["gain_exactitude"] = round(
        resultats["modele"]["exactitude"] - resultats["reference"]["exactitude"], 4)

    with open(CHEMIN_RESULTATS, "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2)

    afficher("Régression logistique", resultats["modele"])
    print(classification_report(y_test, y_pred,
                                target_names=["ferme", "survit"]))
    afficher("Référence (classe majoritaire)", resultats["reference"])
    print(f"Exactitude (NumPy) : {exactitude_numpy(y_test, y_pred):.4f}")
    print(f"Gain par rapport à la référence : "
          f"+{resultats['gain_exactitude'] * 100:.1f} points de pourcentage\n")
    print(LIMITES)
    return resultats


if __name__ == "__main__":
    lancer_evaluation()
