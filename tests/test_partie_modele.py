"""Tests de donnees.py, modele.py et evaluation.py (lancer avec : pytest)."""
import numpy as np
import pandas as pd
import pytest

import donnees
import evaluation
from modele import ModeleSurvie


@pytest.fixture
def faux_df():
    """Petit jeu de données artificiel avec les mêmes colonnes que le vrai CSV.
    La cible dépend du capital de départ pour que le modèle ait quelque chose à apprendre."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "restaurant_id": [f"RST{i:06d}" for i in range(n)],
        "cuisine": rng.choice(["italian", "indian", "cafe"], n),
        "location_type": rng.choice(["downtown", "suburb_strip"], n),
        "is_franchise": rng.integers(0, 2, n),
        "owner_experience_years": rng.uniform(0, 20, n).round(1),
        "foot_traffic": rng.uniform(0, 10, n).round(2),
        "competition_density": rng.uniform(0, 10, n).round(2),
        "rent_to_revenue": rng.uniform(0.02, 0.3, n).round(3),
        "starting_capital_months": rng.uniform(0.5, 36, n).round(1),
        "labor_cost_ratio": rng.uniform(0.15, 0.45, n).round(3),
        "price_tier": rng.integers(1, 5, n),
        "seats": rng.integers(12, 300, n),
        "menu_size": rng.integers(6, 110, n),
        "has_marketing": rng.integers(0, 2, n),
        "avg_review_rating": rng.uniform(2, 5, n).round(2),
        "review_count": rng.integers(0, 900, n),
    })
    df["survived_3y"] = (df["starting_capital_months"] > 12).astype(int)
    df["years_survived"] = np.where(df["survived_3y"] == 1, 3, 1)
    df["failure_reason"] = np.where(df["survived_3y"] == 1, "survived", "undercapitalized")
    return df



def test_preparer_retire_colonnes_qui_trichent(faux_df):
    X, y = donnees.preparer(faux_df)
    for colonne in donnees.COLONNES_EXCLUES + [donnees.CIBLE]:
        assert colonne not in X.columns
    assert len(X) == len(y)


def test_nettoyer_retire_valeurs_impossibles(faux_df):
    """Une note de 9 (impossible) et une valeur manquante doivent être retirées."""
    faux_df.loc[0, "avg_review_rating"] = 9.0
    faux_df.loc[1, "seats"] = None
    propre = donnees.nettoyer(faux_df)
    assert len(propre) == len(faux_df) - 2
    assert propre["avg_review_rating"].max() <= 5


def test_separer_proportions(faux_df):
    X, y = donnees.preparer(faux_df)
    X_train, X_test, y_train, y_test = donnees.separer(X, y)
    assert len(X_test) == pytest.approx(0.2 * len(X), abs=1)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.05)



@pytest.fixture
def modele_entraine(faux_df):
    X, y = donnees.preparer(faux_df)
    return ModeleSurvie().entrainer(X, y), X


def test_predire_retourne_classe_et_proba(modele_entraine):
    modele, X = modele_entraine
    classe, proba = modele.predire(X.iloc[0].to_dict())
    assert classe in (0, 1)
    assert 0.0 <= proba <= 1.0


def test_predire_colonne_manquante(modele_entraine):
    modele, X = modele_entraine
    restaurant = X.iloc[0].to_dict()
    del restaurant["seats"]
    with pytest.raises(ValueError):
        modele.predire(restaurant)


def test_sauvegarde_puis_chargement(modele_entraine, tmp_path):
    modele, X = modele_entraine
    chemin_modele = tmp_path / "m.joblib"
    chemin_meta = tmp_path / "m.json"
    modele.sauvegarder(X, chemin_modele, chemin_meta)
    recharge = ModeleSurvie.charger(chemin_modele, chemin_meta)
    exemple = X.iloc[3].to_dict()
    assert recharge.predire(exemple) == modele.predire(exemple)



def test_exactitude_numpy():
    assert evaluation.exactitude_numpy([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75


def test_taux_par_classe():
    resultat = evaluation.taux_par_classe([[8, 2], [1, 9]])
    assert np.allclose(resultat, [0.8, 0.9])


def test_modele_bat_la_reference(modele_entraine, faux_df):
    modele, X = modele_entraine
    _, y = donnees.preparer(faux_df)
    reference = evaluation.creer_reference(X, y)
    acc_modele = evaluation.exactitude_numpy(y, modele.predire_lot(X))
    acc_reference = evaluation.exactitude_numpy(y, reference.predict(X))
    assert acc_modele > acc_reference
