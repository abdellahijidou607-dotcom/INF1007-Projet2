"""Entraînement, prédiction et sauvegarde du modèle.

Utilisation par l'interface :
    from modele import ModeleSurvie
    m = ModeleSurvie.charger()
    classe, proba = m.predire({"cuisine": "italian", ..., "review_count": 120})
"""
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import donnees

CHEMIN_MODELE = "modele.joblib"
CHEMIN_META = "metadonnees.json"


class ModeleSurvie:
    """Prédit si un restaurant survivra 3 ans."""

    def __init__(self):
        self.pipeline = None
        self.colonnes = []

    def entrainer(self, X_train, y_train):
        """Construit le pipeline (encodage + normalisation + modèle) et l'entraîne.
        - Colonnes texte : transformées en colonnes de 0/1 (une par catégorie).
        - Colonnes numériques : ramenées à une moyenne 0 et un écart-type 1."""
        self.colonnes = list(X_train.columns)
        pretraitement = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), donnees.CATEGORIELLES),
            ("num", StandardScaler(), donnees.colonnes_numeriques(X_train)),
        ])
        self.pipeline = Pipeline([
            ("pretraitement", pretraitement),
            ("modele", LogisticRegression(max_iter=1000)),
        ])
        self.pipeline.fit(X_train, y_train)
        return self

    def predire(self, restaurant):
        """restaurant : dict {nom_colonne: valeur}.
        Retourne (classe, probabilité de survie) avec classe = 1 (survit) ou 0."""
        if self.pipeline is None:
            raise RuntimeError("Le modèle n'est pas entraîné ni chargé.")
        manquantes = [c for c in self.colonnes if c not in restaurant]
        if manquantes:
            raise ValueError(f"Colonnes manquantes : {manquantes}")
        ligne = pd.DataFrame([restaurant])[self.colonnes]
        classe = int(self.pipeline.predict(ligne)[0])
        proba = float(self.pipeline.predict_proba(ligne)[0][1])
        return classe, proba

    def predire_lot(self, X):
        """Prédit pour un DataFrame entier (utilisé par l'évaluation)."""
        return self.pipeline.predict(X[self.colonnes])

    def sauvegarder(self, X_train, chemin_modele=CHEMIN_MODELE,
                    chemin_meta=CHEMIN_META):
        """Sauvegarde le pipeline (joblib) et ses métadonnées (JSON)."""
        joblib.dump(self.pipeline, chemin_modele)
        meta = {
            "algorithme": "LogisticRegression",
            "cible": donnees.CIBLE,
            "colonnes": self.colonnes,
            "types": {c: str(X_train[c].dtype) for c in self.colonnes},
            "categories": {c: sorted(X_train[c].unique().tolist())
                           for c in donnees.CATEGORIELLES},
            "binaires": donnees.BINAIRES,
            "plages": {c: [float(X_train[c].min()), float(X_train[c].max())]
                       for c in donnees.colonnes_numeriques(X_train)},
        }
        with open(chemin_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def charger(cls, chemin_modele=CHEMIN_MODELE, chemin_meta=CHEMIN_META):
        """Recharge un modèle sauvegardé."""
        m = cls()
        m.pipeline = joblib.load(chemin_modele)
        with open(chemin_meta, encoding="utf-8") as f:
            m.colonnes = json.load(f)["colonnes"]
        return m


def entrainer_et_sauvegarder(chemin_csv=donnees.CHEMIN_CSV):
    """Pipeline complet : données -> entraînement -> sauvegarde. Retourne le modèle."""
    X_train, _, y_train, _ = donnees.obtenir_donnees_separees(chemin_csv)
    modele = ModeleSurvie().entrainer(X_train, y_train)
    modele.sauvegarder(X_train)
    return modele


if __name__ == "__main__":
    modele = entrainer_et_sauvegarder()
    print("Modèle entraîné et sauvegardé dans", CHEMIN_MODELE, "et", CHEMIN_META)

    exemple = {
        "cuisine": "italian", "location_type": "downtown", "is_franchise": 1,
        "owner_experience_years": 8.0, "foot_traffic": 7.5,
        "competition_density": 3.0, "rent_to_revenue": 0.08,
        "starting_capital_months": 12.0, "labor_cost_ratio": 0.30,
        "price_tier": 2, "seats": 80, "menu_size": 30, "has_marketing": 1,
        "avg_review_rating": 4.3, "review_count": 200,
    }
    classe, proba = modele.predire(exemple)
    print(f"Exemple : {'survit' if classe else 'ne survit pas'} "
          f"(probabilité de survie : {proba:.1%})")
