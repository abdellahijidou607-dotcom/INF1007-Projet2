"""Tests de modele.py, metriques.py, evaluation.py, historique.py et outils.py."""
import numpy as np
import pytest
from sklearn.metrics import accuracy_score

import evaluation
from exceptions import DonneeInvalideError, FichierIntrouvableError, ModeleNonEntraineError
from historique import Historique
from metriques import exactitude_numpy, rappel_par_classe
from modele import ModeleSurvie, calculer_metrique
from outils import chronometre
from variables import charger_metadonnees, creer_variables


def test_predire_retourne_classe_et_proba(modele_entraine):
    modele, X, _ = modele_entraine
    classe, proba = modele.predire(X.iloc[0].to_dict())
    assert classe in (0, 1)
    assert 0.0 <= proba <= 1.0


def test_predire_colonne_manquante(modele_entraine):
    modele, X, _ = modele_entraine
    restaurant = X.iloc[0].to_dict()
    del restaurant["seats"]
    with pytest.raises(DonneeInvalideError):
        modele.predire(restaurant)


def test_predire_sans_entrainement():
    with pytest.raises(ModeleNonEntraineError):
        ModeleSurvie().predire({})


def test_sauvegarde_puis_chargement(modele_entraine, tmp_path):
    modele, X, _ = modele_entraine
    chemin_modele, chemin_meta = str(tmp_path / "m.joblib"), str(tmp_path / "m.json")
    modele.sauvegarder(X, {"nom": "exactitude", "valeur": 0.9, "reference": 0.5},
                       chemin_modele, chemin_meta)
    recharge = ModeleSurvie.charger(chemin_modele, chemin_meta)
    exemple = X.iloc[3].to_dict()
    assert recharge.predire(exemple) == modele.predire(exemple)
    meta = charger_metadonnees(chemin_meta)
    assert meta["metrique"]["valeur"] == pytest.approx(0.9)
    assert len(creer_variables(meta)) == len(X.columns)


def test_charger_modele_absent(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        ModeleSurvie.charger(str(tmp_path / "a.joblib"), str(tmp_path / "a.json"))


def test_repr_modele(modele_entraine):
    assert repr(ModeleSurvie()) == "ModeleSurvie(non entraîné, 0 colonnes)"
    assert "prêt" in repr(modele_entraine[0])


def test_modele_bat_la_reference(modele_entraine):
    modele, X, y = modele_entraine
    metrique = calculer_metrique(modele, X, X, y, y)
    assert metrique["valeur"] > metrique["reference"]


def test_exactitude_numpy_cas_simple():
    assert exactitude_numpy([1, 0, 1, 1], [1, 0, 0, 1]) == pytest.approx(0.75)


def test_exactitude_numpy_identique_a_sklearn():
    rng = np.random.default_rng(1)
    vrai, pred = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    assert exactitude_numpy(vrai, pred) == pytest.approx(accuracy_score(vrai, pred))


def test_exactitude_numpy_tailles_differentes():
    with pytest.raises(DonneeInvalideError):
        exactitude_numpy([1, 0], [1])


def test_rappel_par_classe():
    assert np.allclose(rappel_par_classe([[8, 2], [1, 9]]), [0.8, 0.9])


def test_rappel_matrice_non_carree():
    with pytest.raises(DonneeInvalideError):
        rappel_par_classe([[1, 2, 3]])


def test_evaluer_matrice_confusion():
    resultats = evaluation.evaluer([0, 0, 1, 1], [0, 1, 1, 1])
    assert resultats["matrice_confusion"] == [[1, 1], [0, 2]]
    assert resultats["exactitude"] == pytest.approx(0.75)


def test_historique_ajout_puis_lecture(tmp_path):
    historique = Historique(str(tmp_path / "h.csv"))
    assert historique.lire() == []
    historique.ajouter({"seats": 80}, 1, 0.8)
    historique.ajouter({"seats": 20}, 0, 0.1)
    assert len(historique) == 2
    assert {e["resultat"] for e in historique.lire()} == {"Survie", "Fermeture"}


def test_historique_refuse_prediction_invalide(tmp_path):
    with pytest.raises(DonneeInvalideError):
        Historique(str(tmp_path / "h.csv")).ajouter({}, 1, 1.5)


def test_historique_ligne_corrompue(tmp_path):
    chemin = tmp_path / "h.csv"
    chemin.write_text("date,prediction,probabilite_survie,donnees\n2026,x,0.5,{}\n", encoding="utf-8")
    with pytest.raises(DonneeInvalideError):
        Historique(str(chemin)).lire()


def test_chronometre_mesure_la_duree():
    @chronometre
    def addition(a: int, b: int) -> int:
        return a + b

    assert addition(2, 3) == 5
    assert addition.derniere_duree >= 0
    assert addition.__name__ == "addition"
