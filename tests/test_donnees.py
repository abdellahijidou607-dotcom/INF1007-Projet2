"""Tests de donnees.py et config.py."""
import pytest

import donnees
from config import verifier_fichier
from exceptions import DonneeInvalideError, FichierIntrouvableError


def test_preparer_retire_colonnes_qui_trichent(faux_df):
    X, y = donnees.preparer(faux_df)
    for colonne in donnees.COLONNES_EXCLUES + [donnees.CIBLE]:
        assert colonne not in X.columns
    assert len(X) == len(y)


def test_nettoyer_retire_valeurs_impossibles_et_manquantes(faux_df):
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


def test_separer_taille_invalide(faux_df):
    X, y = donnees.preparer(faux_df)
    with pytest.raises(DonneeInvalideError):
        donnees.separer(X, y, taille_test=1.5)


def test_preparer_sans_ligne_valide(faux_df):
    faux_df["avg_review_rating"] = 99.0
    with pytest.raises(DonneeInvalideError):
        donnees.preparer(faux_df)


def test_charger_fichier_absent(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        donnees.charger_donnees(str(tmp_path / "absent.csv"))


def test_charger_sans_colonne_cible(tmp_path):
    chemin = tmp_path / "mauvais.csv"
    chemin.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(DonneeInvalideError):
        donnees.charger_donnees(str(chemin))


def test_verifier_fichier_existant(tmp_path):
    chemin = tmp_path / "ok.txt"
    chemin.write_text("x", encoding="utf-8")
    verifier_fichier(str(chemin))
