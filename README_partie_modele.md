## Partie données et modèle

### Fichiers
| Fichier | Rôle |
|---|---|
| `donnees.py` | Chargement, exploration, nettoyage, séparation entraînement/test |
| `modele.py` | Classe `ModeleSurvie` : entraînement, prédiction, sauvegarde/chargement |
| `evaluation.py` | Exactitude, matrice de confusion, comparaison avec une référence |
| `tests/test_partie_modele.py` | Tests unitaires (pytest) |

### Utilisation
```bash
pip install -r requirements.txt
python donnees.py       # exploration du jeu de données
python modele.py        # entraîne et crée modele.joblib + metadonnees.json
python evaluation.py    # évalue et crée resultats_evaluation.json
pytest                  # lance les tests
```

### Choix
- **Cible :** `survived_3y` (1 = survit 3 ans).
- **Colonnes retirées :** `restaurant_id` (identifiant), `years_survived` et `failure_reason` (connues seulement après coup : elles donnent directement la réponse).
- **Valeurs extrêmes :** détectées par la méthode IQR mais conservées, car plausibles. Seules les valeurs impossibles (ex. note > 5) sont supprimées. Il n'y en a aucune dans le jeu de données.
- **Modèle :** régression logistique dans un pipeline scikit-learn (encodage one-hot des catégories + normalisation des nombres).

### Résultats (ensemble de test, 9 000 restaurants)
| Modèle | Exactitude |
|---|---|
| Régression logistique | 76,7 % |
| Référence (classe majoritaire) | 50,6 % |

### Pour l'interface
```python
from modele import ModeleSurvie
m = ModeleSurvie.charger()
classe, proba = m.predire({...})   # dict avec les 15 colonnes de metadonnees.json
```
`metadonnees.json` contient la liste des colonnes, leurs types, les choix possibles pour `cuisine` et `location_type`, ainsi que les plages min/max des valeurs numériques (utile pour les curseurs).
