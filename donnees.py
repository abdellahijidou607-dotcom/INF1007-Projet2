"""Chargement, exploration, nettoyage et préparation des données.

Jeu de données : donnees/restaurant_survival.csv (45 000 restaurants, 19 colonnes).
Cible : survived_3y (1 = le restaurant a survécu 3 ans, 0 = il a fermé avant).

Colonnes exclues :
- restaurant_id : identifiant, aucune information utile.
- years_survived et failure_reason : connues seulement après coup, elles donnent
  directement la réponse (fuite de données), il faut donc les retirer.

PLAGES_VALIDES : plages de valeurs physiquement possibles (au-delà : erreur de saisie).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CHEMIN_CSV, verifier_fichier
from exceptions import DonneeInvalideError

CIBLE = "survived_3y"
COLONNES_EXCLUES = ["restaurant_id", "years_survived", "failure_reason"]
CATEGORIELLES = ["cuisine", "location_type"]
BINAIRES = ["is_franchise", "has_marketing"]
GRAINE = 42

PLAGES_VALIDES = {
    "owner_experience_years": (0, 60),
    "foot_traffic": (0, 10),
    "competition_density": (0, 10),
    "rent_to_revenue": (0, 1),
    "starting_capital_months": (0, 120),
    "labor_cost_ratio": (0, 1),
    "price_tier": (1, 4),
    "seats": (1, 2000),
    "menu_size": (1, 500),
    "avg_review_rating": (1, 5),
    "review_count": (0, 100_000),
}


def charger_donnees(chemin: str = CHEMIN_CSV) -> pd.DataFrame:
    """Lit le fichier CSV.

    Reçoit : le chemin du CSV.
    Renvoie : un DataFrame.
    Échoue : FichierIntrouvableError si le fichier n'existe pas,
    DonneeInvalideError si la colonne cible est absente.
    """
    verifier_fichier(chemin)
    df = pd.read_csv(chemin)
    if CIBLE not in df.columns:
        raise DonneeInvalideError(f"Colonne cible absente : {CIBLE}")
    return df


def resumer(df: pd.DataFrame) -> dict:
    """Calcule un résumé du jeu de données (sans rien afficher).

    Reçoit : le DataFrame.
    Renvoie : un dict avec dimensions, valeurs manquantes et répartition de la cible.
    Échoue : ne lève rien.
    """
    return {
        "dimensions": df.shape,
        "valeurs_manquantes": int(df.isna().sum().sum()),
        "repartition_cible": df[CIBLE].value_counts(normalize=True).round(3).to_dict(),
    }


def compter_aberrantes_iqr(df: pd.DataFrame) -> pd.Series:
    """Compte les valeurs hors de [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (calcul vectorisé).

    Reçoit : le DataFrame.
    Renvoie : une Series (colonne -> nombre de valeurs extrêmes).
    Échoue : ne lève rien.
    """
    colonnes = [c for c in PLAGES_VALIDES if c in df.columns]
    valeurs = df[colonnes]
    q1 = valeurs.quantile(0.25)
    q3 = valeurs.quantile(0.75)
    iqr = q3 - q1
    masque = (valeurs < q1 - 1.5 * iqr) | (valeurs > q3 + 1.5 * iqr)
    return masque.sum()


def masque_valeurs_impossibles(df: pd.DataFrame) -> np.ndarray:
    """Repère les lignes ayant au moins une valeur physiquement impossible.

    Reçoit : le DataFrame.
    Renvoie : un tableau booléen (True = ligne à retirer).
    Échoue : ne lève rien.
    """
    masque = np.zeros(len(df), dtype=bool)
    for colonne, (mini, maxi) in PLAGES_VALIDES.items():
        if colonne in df.columns:
            masque |= ((df[colonne] < mini) | (df[colonne] > maxi)).to_numpy()
    return masque


def nettoyer(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes incomplètes, en double ou impossibles.

    Les valeurs simplement extrêmes (méthode IQR) sont conservées : elles sont
    plausibles (ex. un grand restaurant de 300 places).
    Reçoit : le DataFrame brut.
    Renvoie : un nouveau DataFrame nettoyé.
    Échoue : ne lève rien.
    """
    df = df.dropna().drop_duplicates()
    return df[~masque_valeurs_impossibles(df)].reset_index(drop=True)


def preparer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Nettoie, retire les colonnes exclues et sépare variables et cible.

    Reçoit : le DataFrame brut.
    Renvoie : (X, y).
    Échoue : DonneeInvalideError s'il ne reste aucune ligne après nettoyage.
    """
    df = nettoyer(df).drop(columns=COLONNES_EXCLUES)
    if df.empty:
        raise DonneeInvalideError("Aucune ligne valide après nettoyage.")
    return df.drop(columns=[CIBLE]), df[CIBLE]


def separer(X: pd.DataFrame, y: pd.Series, taille_test: float = 0.2) -> list:
    """Sépare en entraînement (80 %) et test (20 %) avec une graine fixée.

    stratify=y garde la même proportion de 0 et de 1 dans les deux ensembles.
    Reçoit : X, y et la proportion du test.
    Renvoie : [X_train, X_test, y_train, y_test].
    Échoue : DonneeInvalideError si taille_test n'est pas strictement entre 0 et 1.
    """
    if not 0 < taille_test < 1:
        raise DonneeInvalideError("taille_test doit être strictement entre 0 et 1.")
    return train_test_split(X, y, test_size=taille_test, random_state=GRAINE, stratify=y)


def colonnes_numeriques(X: pd.DataFrame) -> list[str]:
    """Renvoie la liste des colonnes qui ne sont pas catégorielles."""
    return [c for c in X.columns if c not in CATEGORIELLES]


def obtenir_donnees_separees(chemin: str = CHEMIN_CSV) -> list:
    """Charge, prépare et sépare. Renvoie [X_train, X_test, y_train, y_test]."""
    X, y = preparer(charger_donnees(chemin))
    return separer(X, y)


if __name__ == "__main__":
    donnees = charger_donnees()
    resume = resumer(donnees)
    print("Dimensions :", resume["dimensions"])
    print("Valeurs manquantes :", resume["valeurs_manquantes"])
    print("Répartition de la cible :", resume["repartition_cible"])
    print("\nValeurs extrêmes (méthode IQR) par colonne :")
    print(compter_aberrantes_iqr(donnees))
    print("\nLignes avec valeurs impossibles :", int(masque_valeurs_impossibles(donnees).sum()))
    print("Lignes après nettoyage :", len(nettoyer(donnees)))
