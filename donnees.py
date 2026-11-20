"""Chargement, exploration, nettoyage et préparation des données.

Jeu de données : restaurant_survival.csv (45 000 restaurants, 19 colonnes).
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

CHEMIN_CSV = "restaurant_survival.csv"
CIBLE = "survived_3y"
COLONNES_EXCLUES = ["restaurant_id", "years_survived", "failure_reason"]

CATEGORIELLES = ["cuisine", "location_type"]
BINAIRES = ["is_franchise", "has_marketing"]
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


def charger_donnees(chemin=CHEMIN_CSV):
    """Lit le fichier CSV et retourne un DataFrame."""
    return pd.read_csv(chemin)


def explorer(df):
    """Affiche un résumé du jeu de données."""
    print("Dimensions :", df.shape)
    print("Valeurs manquantes :", int(df.isna().sum().sum()))
    print("\nRépartition de la cible :")
    print(df[CIBLE].value_counts(normalize=True).round(3))
    print("\nStatistiques descriptives :")
    print(df.describe().T[["mean", "min", "max"]])


def compter_aberrantes_iqr(df):
    """Compte, pour chaque colonne numérique non binaire, les valeurs hors de
    [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. Calcul vectorisé (pas de boucle sur les lignes)."""
    colonnes = [c for c in PLAGES_VALIDES if c in df.columns]
    valeurs = df[colonnes]
    q1 = valeurs.quantile(0.25)
    q3 = valeurs.quantile(0.75)
    iqr = q3 - q1
    masque = (valeurs < q1 - 1.5 * iqr) | (valeurs > q3 + 1.5 * iqr)
    return masque.sum()


def masque_valeurs_impossibles(df):
    """Retourne un masque booléen : True pour les lignes ayant au moins une
    valeur hors des plages physiquement possibles."""
    masque = np.zeros(len(df), dtype=bool)
    for colonne, (mini, maxi) in PLAGES_VALIDES.items():
        if colonne in df.columns:
            masque |= ((df[colonne] < mini) | (df[colonne] > maxi)).to_numpy()
    return masque


def nettoyer(df):
    """Supprime les lignes incomplètes, en double ou impossibles.
    Les valeurs simplement extrêmes (méthode IQR) sont conservées :
    elles sont plausibles (ex. un grand restaurant de 300 places)."""
    df = df.dropna().drop_duplicates()
    return df[~masque_valeurs_impossibles(df)].reset_index(drop=True)


def preparer(df):
    """Retire les colonnes exclues et sépare les variables (X) de la cible (y)."""
    df = nettoyer(df).drop(columns=COLONNES_EXCLUES)
    X = df.drop(columns=[CIBLE])
    y = df[CIBLE]
    return X, y


def separer(X, y, taille_test=0.2):
    """Sépare en ensembles d'entraînement (80 %) et de test (20 %).
    stratify=y garde la même proportion de 0 et de 1 dans les deux ensembles."""
    return train_test_split(X, y, test_size=taille_test,
                            random_state=42, stratify=y)


def colonnes_numeriques(X):
    """Liste des colonnes qui ne sont pas catégorielles."""
    return [c for c in X.columns if c not in CATEGORIELLES]


def obtenir_donnees_separees(chemin=CHEMIN_CSV):
    """Raccourci : charge, prépare et sépare. Retourne X_train, X_test, y_train, y_test."""
    X, y = preparer(charger_donnees(chemin))
    return separer(X, y)


if __name__ == "__main__":
    donnees = charger_donnees()
    explorer(donnees)
    print("\nValeurs extrêmes (méthode IQR) par colonne :")
    print(compter_aberrantes_iqr(donnees))
    print("\nLignes avec valeurs impossibles :",
          int(masque_valeurs_impossibles(donnees).sum()))
    print("Lignes après nettoyage :", len(nettoyer(donnees)))
