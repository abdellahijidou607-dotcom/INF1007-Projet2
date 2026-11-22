"""Entraînement, prédiction et sauvegarde du modèle.

Utilisation :
    from modele import ModeleSurvie
    m = ModeleSurvie.charger()
    classe, proba = m.predire({"cuisine": "italian", ..., "review_count": 120})
"""
import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import donnees
from config import CHEMIN_CSV, CHEMIN_META, CHEMIN_MODELE, verifier_fichier
from exceptions import DonneeInvalideError, ModeleNonEntraineError
from metriques import exactitude_numpy
from outils import chronometre


class ModeleSurvie:
    """Prédit si un restaurant survivra 3 ans (régression logistique)."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._colonnes: list[str] = []

    @property
    def colonnes(self) -> list[str]:
        """Copie de la liste des colonnes attendues, dans l'ordre."""
        return list(self._colonnes)

    @property
    def est_pret(self) -> bool:
        """True si le modèle a été entraîné ou chargé."""
        return self._pipeline is not None

    @chronometre
    def entrainer(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ModeleSurvie":
        """Construit le pipeline (encodage + normalisation + modèle) et l'entraîne.

        - Colonnes texte : transformées en colonnes de 0/1 (une par catégorie).
        - Colonnes numériques : ramenées à une moyenne 0 et un écart-type 1.
        Reçoit : X_train et y_train.
        Renvoie : le modèle lui-même (pour enchaîner les appels).
        Échoue : DonneeInvalideError si X_train est vide.
        """
        if X_train.empty:
            raise DonneeInvalideError("Impossible d'entraîner sur des données vides.")
        self._colonnes = list(X_train.columns)
        pretraitement = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), donnees.CATEGORIELLES),
            ("num", StandardScaler(), donnees.colonnes_numeriques(X_train)),
        ])
        self._pipeline = Pipeline([
            ("pretraitement", pretraitement),
            ("modele", LogisticRegression(max_iter=1000)),
        ])
        self._pipeline.fit(X_train, y_train)
        return self

    def _verifier_pret(self) -> None:
        """Lève ModeleNonEntraineError si le modèle n'est ni entraîné ni chargé."""
        if self._pipeline is None:
            raise ModeleNonEntraineError("Le modèle n'est ni entraîné ni chargé.")

    def predire(self, restaurant: dict) -> tuple[int, float]:
        """Prédit la survie d'un restaurant.

        Reçoit : un dict {nom_colonne: valeur} contenant toutes les colonnes.
        Renvoie : (classe, probabilité de survie), classe = 1 (survit) ou 0.
        Échoue : ModeleNonEntraineError si le modèle n'est pas prêt,
        DonneeInvalideError s'il manque des colonnes.
        """
        self._verifier_pret()
        manquantes = [c for c in self._colonnes if c not in restaurant]
        if manquantes:
            raise DonneeInvalideError(f"Colonnes manquantes : {manquantes}")
        ligne = pd.DataFrame([restaurant])[self._colonnes]
        classe = int(self._pipeline.predict(ligne)[0])
        proba = float(self._pipeline.predict_proba(ligne)[0][1])
        return classe, proba

    def predire_lot(self, X: pd.DataFrame):
        """Prédit pour un DataFrame entier. Renvoie un tableau de 0 et de 1."""
        self._verifier_pret()
        return self._pipeline.predict(X[self._colonnes])

    def sauvegarder(self, X_train: pd.DataFrame, metrique: dict,
                    chemin_modele: str = CHEMIN_MODELE,
                    chemin_meta: str = CHEMIN_META) -> None:
        """Sauvegarde le pipeline (joblib) et ses métadonnées (JSON).

        Reçoit : X_train (pour les types, catégories et intervalles), la métrique
        obtenue et les chemins de sortie.
        Renvoie : rien.
        Échoue : ModeleNonEntraineError si le modèle n'est pas entraîné,
        OSError si l'écriture échoue.
        """
        self._verifier_pret()
        os.makedirs(os.path.dirname(os.path.abspath(chemin_modele)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(chemin_meta)), exist_ok=True)
        joblib.dump(self._pipeline, chemin_modele)
        meta = {
            "algorithme": "LogisticRegression",
            "cible": donnees.CIBLE,
            "colonnes": self._colonnes,
            "types": {c: str(X_train[c].dtype) for c in self._colonnes},
            "categories": {c: sorted(X_train[c].unique().tolist())
                           for c in donnees.CATEGORIELLES},
            "binaires": donnees.BINAIRES,
            "plages": {c: [float(X_train[c].min()), float(X_train[c].max())]
                       for c in donnees.colonnes_numeriques(X_train)},
            "metrique": metrique,
        }
        with open(chemin_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def charger(cls, chemin_modele: str = CHEMIN_MODELE,
                chemin_meta: str = CHEMIN_META) -> "ModeleSurvie":
        """Recharge un modèle sauvegardé.

        Reçoit : les chemins du modèle et des métadonnées.
        Renvoie : un ModeleSurvie prêt à prédire.
        Échoue : FichierIntrouvableError si un des deux fichiers manque.
        """
        verifier_fichier(chemin_modele)
        verifier_fichier(chemin_meta)
        m = cls()
        m._pipeline = joblib.load(chemin_modele)
        with open(chemin_meta, encoding="utf-8") as f:
            m._colonnes = json.load(f)["colonnes"]
        return m

    def __repr__(self) -> str:
        etat = "prêt" if self.est_pret else "non entraîné"
        return f"ModeleSurvie({etat}, {len(self._colonnes)} colonnes)"


def calculer_metrique(modele: ModeleSurvie, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series, y_test: pd.Series) -> dict:
    """Calcule l'exactitude du modèle et celle de la référence sur le test.

    Reçoit : le modèle entraîné et les quatre ensembles.
    Renvoie : {"nom", "valeur", "reference"}.
    Échoue : ModeleNonEntraineError si le modèle n'est pas entraîné.
    """
    reference = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    return {
        "nom": "exactitude",
        "valeur": round(exactitude_numpy(y_test, modele.predire_lot(X_test)), 4),
        "reference": round(exactitude_numpy(y_test, reference.predict(X_test)), 4),
    }


def entrainer_et_sauvegarder(chemin_csv: str = CHEMIN_CSV) -> tuple[ModeleSurvie, dict]:
    """Pipeline complet : données -> entraînement -> évaluation -> sauvegarde.

    Reçoit : le chemin du CSV.
    Renvoie : (modèle entraîné, métrique obtenue).
    Échoue : FichierIntrouvableError si le CSV n'existe pas.
    """
    X_train, X_test, y_train, y_test = donnees.obtenir_donnees_separees(chemin_csv)
    modele = ModeleSurvie().entrainer(X_train, y_train)
    metrique = calculer_metrique(modele, X_train, X_test, y_train, y_test)
    modele.sauvegarder(X_train, metrique)
    return modele, metrique


if __name__ == "__main__":
    modele, metrique = entrainer_et_sauvegarder()
    print(f"Entraînement terminé en {ModeleSurvie.entrainer.derniere_duree:.2f} s")
    print(f"Exactitude : {metrique['valeur']:.1%} (référence : {metrique['reference']:.1%})")
    print("Modèle sauvegardé dans", CHEMIN_MODELE, "et", CHEMIN_META)
