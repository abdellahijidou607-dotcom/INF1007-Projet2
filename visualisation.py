
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def graphique_survie():
    # Chemin du fichier CSV
    chemin = Path(__file__).parent / "donnees" / "restaurant_survival.csv"

    # Lecture des données
    donnees = pd.read_csv(chemin)

    # Compter les restaurants fermés et survivants
    resultats = donnees["survived_3y"].value_counts()

    fermetures = resultats.get(0, 0)
    survies = resultats.get(1, 0)

    # Création du graphique
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        ["Fermeture", "Survie"],
        [fermetures, survies],
        color=["red", "green"]
    )

    ax.set_title("Survie des restaurants après 3 ans")
    ax.set_xlabel("Résultat")
    ax.set_ylabel("Nombre de restaurants")

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    graphique_survie()
    plt.show()
