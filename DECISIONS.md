# Fiche de décisions

Projet 2 INF1007 — Prédiction de survie des restaurants

| Choix | Options envisagées | Choix retenu et justification | Validé par l'auxiliaire (date) |
|---|---|---|---|
| Domaine | Finance (cours de la bourse NVDA, S&P 500, taux de change EUR) ; restaurants | **Restaurants.** Les données financières sont des séries temporelles, mal adaptées à un modèle de classification simple et aux critères de l'énoncé. La survie d'un restaurant est une question concrète qu'un non-informaticien comprend. | |
| Jeu de données | Données boursières ; « Restaurant Survival — Why They Really Close » (Kaggle) | **Restaurant Survival (Kaggle, Sergey Nefedov).** 45 000 lignes et 19 colonnes (dans l'intervalle 500 à 50 000), variables numériques et catégorielles, licence CC0, aucune donnée personnelle. Limite connue : données synthétiques. | |
| Cible | `survived_3y`, `years_survived` (0 à 3), `failure_reason` | **`survived_3y`** (classification binaire). `years_survived` et `failure_reason` sont retirées des caractéristiques : elles ne sont connues qu'après coup et donnent directement la réponse (fuite de données, vérifiée par un tableau croisé). | |
| Métrique | Exactitude, F1, précision, rappel | **Exactitude.** Les deux classes sont équilibrées (51 % / 49 %), donc l'exactitude n'est pas trompeuse. Complétée par la matrice de confusion et le rappel par classe. | |
| Référence | Classe majoritaire, prédiction aléatoire | **Classe majoritaire** (`DummyClassifier`) : 50,6 %. Point de comparaison simple pour juger le modèle. | |
| Modèle | Régression logistique, arbre de décision, forêt aléatoire, k plus proches voisins | **Régression logistique.** Simple, facile à expliquer et donne une probabilité affichable dans l'interface. Résultat : 76,7 % d'exactitude contre 50,6 % pour la référence. | |
| Préparation | Encodage manuel ; pipeline scikit-learn | **Pipeline** (`OneHotEncoder` pour les catégories, `StandardScaler` pour les nombres) : les saisies de l'interface subissent exactement les mêmes transformations que les données d'entraînement. | |
| Séparation | 70/30, 80/20 | **80/20** avec graine fixée (42) et `stratify`, pour des résultats reproductibles et la même proportion de survivants dans les deux ensembles. | |
| Technologie d'interface | Tkinter, Streamlit, PyQt | **Tkinter** : inclus avec Python, fonctionne en local sans installation, et permet d'intégrer un graphique Matplotlib. | |
| Conception | Validation des saisies dans l'interface ; classes dédiées | **Classe abstraite `Variable`** avec `VariableNumerique` et `VariableCategorielle`, qui valident chacune la saisie à leur façon. L'interface ne contient aucune logique, ce qui rend le projet testable. | |
| Organisation du dépôt | Tous les fichiers à la racine ; dossiers `donnees/` et `modeles/` | **Dossiers `donnees/` et `modeles/`**, comme dans la structure suggérée par l'énoncé, après une première version où les chemins étaient incohérents entre nos deux parties. | |
