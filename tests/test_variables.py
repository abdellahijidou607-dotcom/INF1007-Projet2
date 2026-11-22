"""Tests des classes Variable (héritage, polymorphisme, méthodes spéciales)."""
import json

import pytest

from exceptions import DonneeInvalideError, FichierIntrouvableError
from variables import (Variable, VariableCategorielle, VariableNumerique,
                       charger_metadonnees, creer_variables)


@pytest.fixture
def places() -> VariableNumerique:
    return VariableNumerique("seats", "Nombre de places", 12, 300, entier=True)


@pytest.fixture
def ratio() -> VariableNumerique:
    return VariableNumerique("rent_to_revenue", "Ratio loyer", 0.02, 0.32)


@pytest.fixture
def cuisine() -> VariableCategorielle:
    return VariableCategorielle("cuisine", "Type de cuisine", ["cafe", "italian"])


def test_numerique_valeur_usuelle(ratio):
    assert ratio.valider(" 0.1 ") == pytest.approx(0.1)


def test_numerique_virgule_decimale(ratio):
    assert ratio.valider("0,15") == pytest.approx(0.15)


def test_numerique_bornes_incluses(places):
    assert places.valider("12") == 12
    assert places.valider("300") == 300


@pytest.mark.parametrize("texte", ["", "   ", "abc", "nan", "inf", "11", "301", "48.5"])
def test_numerique_saisies_invalides(places, texte):
    with pytest.raises(DonneeInvalideError):
        places.valider(texte)


def test_numerique_entier_renvoie_int(places):
    valeur = places.valider("48.0")
    assert valeur == 48 and isinstance(valeur, int)


def test_intervalle_inverse_refuse():
    with pytest.raises(DonneeInvalideError):
        VariableNumerique("x", "X", 10, 1)


def test_categorielle_valide(cuisine):
    assert cuisine.valider("italian") == "italian"


@pytest.mark.parametrize("texte", ["", "pizza", "Italian"])
def test_categorielle_invalide(cuisine, texte):
    with pytest.raises(DonneeInvalideError):
        cuisine.valider(texte)


def test_categorielle_binaire_renvoie_int():
    franchise = VariableCategorielle("is_franchise", "Franchise", ["0", "1"], convertir_en_entier=True)
    assert franchise.valider("1") == 1


def test_choix_proteges(cuisine):
    cuisine.choix.append("pirate")
    assert "pirate" not in cuisine.choix


def test_variable_abstraite_non_instanciable():
    with pytest.raises(TypeError):
        Variable("x", "X")


def test_polymorphisme(places, cuisine):
    saisies = {"seats": "80", "cuisine": "cafe"}
    resultat = {v.nom: v.valider(saisies[v.nom]) for v in [places, cuisine]}
    assert resultat == {"seats": 80, "cuisine": "cafe"}


def test_methodes_speciales(places, cuisine):
    assert repr(places) == "VariableNumerique(nom='seats')"
    assert "12 à 300" in str(places)
    assert places == VariableNumerique("seats", "Autre libellé", 0, 1)
    assert places != cuisine


def test_creer_variables_depuis_metadonnees():
    meta = {
        "colonnes": ["cuisine", "is_franchise", "seats", "foot_traffic"],
        "types": {"seats": "int64", "foot_traffic": "float64"},
        "categories": {"cuisine": ["cafe", "italian"]},
        "binaires": ["is_franchise"],
        "plages": {"is_franchise": [0, 1], "seats": [12, 300], "foot_traffic": [0, 10]},
    }
    variables = creer_variables(meta)
    assert [type(v).__name__ for v in variables] == [
        "VariableCategorielle", "VariableCategorielle", "VariableNumerique", "VariableNumerique"]
    assert variables[2].valider("50") == 50


def test_creer_variables_colonne_inconnue():
    meta = {"colonnes": ["mystere"], "types": {}, "categories": {}, "plages": {}}
    with pytest.raises(DonneeInvalideError):
        creer_variables(meta)


def test_charger_metadonnees_incompletes(tmp_path):
    chemin = tmp_path / "meta.json"
    chemin.write_text(json.dumps({"colonnes": []}), encoding="utf-8")
    with pytest.raises(DonneeInvalideError):
        charger_metadonnees(str(chemin))


def test_charger_metadonnees_absentes(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        charger_metadonnees(str(tmp_path / "absent.json"))
