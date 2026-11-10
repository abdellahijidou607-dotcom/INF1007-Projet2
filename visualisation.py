import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

chemin = Path(__file__).resolve().parent / "donnees" / "restaurant_survival.csv"
donnees = pd.read_csv(chemin)

print(donnees.head())
print("Nombre de restaurants :", len(donnees))
print("Colonnes :", donnees.columns.tolist())

 # Compter les restaurants selon leur survie
effectifs = donnees["survived_3y"].value_counts()

# Créer le graphique
plt.bar(
    ["Fermeture", "Survie"],
    [effectifs.get(0, 0), effectifs.get(1, 0)],
    color=["red", "green"]
)

# Ajouter les titres
plt.title("Survie des restaurants après 3 ans")
plt.xlabel("Résultat")
plt.ylabel("Nombre de restaurants")

# Afficher le graphique
plt.savefig("graphique_survie.png")
plt.show()
