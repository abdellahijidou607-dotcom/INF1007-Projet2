# Prédiction de survie des restaurants

Projet 2 — INF1007 (Introduction à la programmation), Polytechnique Montréal, automne 2026.

**Équipe :** Hicham Omar Mahamoud, Abdellahi Jidou

## Description

Application qui estime si un restaurant a des chances de **survivre au moins 3 ans** à partir de ses caractéristiques : type de cuisine, emplacement, expérience du propriétaire, capital de départ, loyer, fréquentation, avis clients, etc.

L'utilisateur remplit un formulaire, clique sur **Prédire** et obtient un résultat en langage clair (« Survie probable à 3 ans ») avec une probabilité et un graphique. Chaque prédiction est enregistrée dans un historique consultable depuis l'interface.

## Jeu de données

| | |
|---|---|
| Fichier | `donnees/restaurant_survival.csv` |
| Source | [Restaurant Survival — Why They Really Close](https://www.kaggle.com/datasets/sergionefedov/restaurant-survival-why-they-really-close), Sergey Nefedov, Kaggle |
| Licence | CC0 (domaine public) : redistribution autorisée |
| Taille | 45 000 restaurants, 19 colonnes |
| Cible | `survived_3y` (1 = a survécu 3 ans, 0 = a fermé avant) |
| Variables d'entrée | 15 : 2 catégorielles (`cuisine`, `location_type`), 13 numériques |

Aucune donnée personnelle. Les colonnes `restaurant_id`, `years_survived` et `failure_reason` sont retirées : les deux dernières ne sont connues qu'après coup et donnent directement la réponse (fuite de données).

## Installation

Python 3.12 ou plus récent.

```bash
git clone https://github.com/abdellahijidou607-dotcom/INF1007-Projet2.git
cd INF1007-Projet2
pip install -r requirements.txt
```

## Lancement

```bash
python main.py
```

Le modèle entraîné est déjà fourni dans `modeles/`. Pour le ré-entraîner :

```bash
python modele.py
```

Autres commandes :

```bash
python donnees.py          # exploration du jeu de données
python evaluation.py       # évaluation détaillée du modèle
python visualisation.py    # génère le graphique d'exploration (graphique_survie.png)
pytest                     # lance les tests
```

## Structure du code

```
INF1007-Projet2/
├── main.py              point d'entrée : charge le modèle et lance l'interface
├── interface.py         interface graphique Tkinter (classe Application)
├── variables.py         classes Variable, VariableNumerique, VariableCategorielle
├── modele.py            classe ModeleSurvie : entraînement, prédiction, sauvegarde
├── donnees.py           chargement, nettoyage, séparation entraînement / test
├── evaluation.py        évaluation et comparaison avec la référence
├── metriques.py         métriques calculées avec NumPy
├── historique.py        classe Historique : enregistrement des prédictions (CSV)
├── visualisation.py     graphiques Matplotlib
├── exceptions.py        exceptions personnalisées
├── config.py            chemins des fichiers (module os)
├── outils.py            décorateur chronometre
├── donnees/             jeu de données (CSV)
├── modeles/             modèle sauvegardé (joblib) + métadonnées (JSON)
├── tests/               tests pytest
└── .github/workflows/   intégration continue (pytest à chaque push)
```

**Conception orientée objet**
- **Héritage et polymorphisme :** `Variable` est une classe abstraite. `VariableNumerique` et `VariableCategorielle` redéfinissent toutes les deux `valider()` : la première vérifie un nombre et son intervalle, la seconde vérifie que le choix fait partie de la liste.
- **Composition :** `Application` contient le modèle (`ModeleSurvie`), la liste des `Variable` et l'`Historique`.
- **Séparation calcul / affichage :** l'interface ne contient aucune logique. Elle appelle les classes du projet, ce qui rend la logique testable.

## Modèle et résultats

- **Problème :** classification binaire.
- **Modèle :** régression logistique (scikit-learn), précédée d'un encodage one-hot des catégories et d'une normalisation des nombres.
- **Séparation :** 80 % entraînement / 20 % test, graine aléatoire fixée (42), proportions des classes conservées.
- **Métrique :** exactitude. Elle est pertinente ici car les deux classes sont équilibrées (51 % / 49 %).

| Modèle | Exactitude (test, 9 000 restaurants) |
|---|---|
| Régression logistique | **76,7 %** |
| Référence (prédit toujours la classe majoritaire) | 50,6 % |

Matrice de confusion (lignes = réel, colonnes = prédit) :

| | Prédit : ferme | Prédit : survit |
|---|---|---|
| **Réel : ferme** | 3 399 | 1 048 |
| **Réel : survit** | 1 046 | 3 507 |

**Limites :** le modèle se trompe sur environ 1 restaurant sur 4. La régression logistique ne capte pas les interactions complexes entre variables. Les données sont 100 % synthétiques (générées par simulation, d'après l'auteur) : les résultats ne garantissent pas la même performance sur de vrais restaurants. Certaines variables (note moyenne, nombre d'avis) ne sont connues qu'après l'ouverture.

## Tests et couverture

51 tests pytest (cas usuels, cas limites, cas d'erreur avec `pytest.raises`, flottants avec `pytest.approx`), lancés automatiquement par GitHub Actions à chaque push.

```bash
coverage run -m pytest
coverage report
```

**Couverture : 89 %** (l'interface graphique n'est pas testée automatiquement).

## Outil du chapitre 7

Un **décorateur** `@chronometre` (`outils.py`) mesure la durée de l'entraînement du modèle sans modifier le code de la méthode `entrainer()`. Une **lambda** sert aussi de clé de tri pour afficher l'historique de la prédiction la plus récente à la plus ancienne.

## Documentation consultée

- pandas, `read_csv` : https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
- scikit-learn, `LogisticRegression` : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- scikit-learn, `ColumnTransformer` : https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
- scikit-learn, `DummyClassifier` : https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
- Tkinter : https://docs.python.org/3/library/tkinter.html
- Matplotlib dans Tkinter : https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_sgskip.html

## Fichiers complémentaires

- `DECISIONS.md` : fiche des choix et de leurs justifications
- `JOURNAL_IA.md` : journal d'utilisation de l'IA