"""Données artificielles partagées par les tests."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def faux_df() -> pd.DataFrame:
    """Petit jeu de données avec les mêmes colonnes que le vrai CSV.
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


@pytest.fixture
def modele_entraine(faux_df):
    """Modèle entraîné sur le faux jeu de données, avec X et y."""
    import donnees
    from modele import ModeleSurvie
    X, y = donnees.preparer(faux_df)
    return ModeleSurvie().entrainer(X, y), X, y
