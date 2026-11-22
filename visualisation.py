"""Graphiques Matplotlib : exploration des données et résultat d'une prédiction."""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from config import CHEMIN_CSV, CHEMIN_GRAPHIQUE, verifier_fichier
from exceptions import DonneeInvalideError


def graphique_survie(chemin: str = CHEMIN_CSV) -> Figure:
    """Crée le graphique d'exploration : restaurants fermés et survivants.

    Reçoit : le chemin du CSV.
    Renvoie : la figure Matplotlib.
    Échoue : FichierIntrouvableError si le CSV manque,
    DonneeInvalideError si la colonne survived_3y est absente.
    """
    verifier_fichier(chemin)
    donnees = pd.read_csv(chemin)

    if "survived_3y" not in donnees.columns:
        raise DonneeInvalideError("Colonne survived_3y introuvable.")

    effectifs = donnees["survived_3y"].value_counts()

    fermes = int(effectifs.get(0, 0))
    survivants = int(effectifs.get(1, 0))
    total = fermes + survivants

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    couleurs = ["#E76F51", "#2A9D8F"]

    barres = ax.bar(
        ["Fermeture", "Survie"],
        [fermes, survivants],
        color=couleurs,
        width=0.48
    )

    ax.set_title(
        "Survie des restaurants après 3 ans",
        fontsize=13,
        fontweight="bold",
        color="#17324D",
        pad=18
    )

    ax.set_ylabel("Nombre de restaurants")
    ax.set_xlabel("Résultat")
    ax.set_ylim(0, max(fermes, survivants, 1) * 1.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.yaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)

    for barre, nombre in zip(
        barres, [fermes, survivants]
    ):
        pourcentage = (
            100 * nombre / total if total else 0
        )

        ax.text(
            barre.get_x() + barre.get_width() / 2,
            barre.get_height() + max(total * 0.01, 1),
            f"{nombre:,}\n({pourcentage:.1f} %)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#17324D"
        )

    fig.tight_layout()
    return fig


def graphique_prediction(probabilite: float) -> Figure:
    """Crée le graphique des probabilités estimées pour une prédiction.

    Reçoit : la probabilité de survie, entre 0 et 1.
    Renvoie : la figure Matplotlib.
    Échoue : DonneeInvalideError si la probabilité est hors de [0, 1].
    """
    if not 0 <= probabilite <= 1:
        raise DonneeInvalideError("Probabilité invalide.")

    fig, ax = plt.subplots(figsize=(6, 3.5))

    valeurs = [
        100 * (1 - probabilite),
        100 * probabilite
    ]

    barres = ax.bar(
        ["Fermeture", "Survie"],
        valeurs,
        color=["#E76F51", "#2A9D8F"],
        width=0.48
    )

    ax.set_ylim(0, 115)
    ax.set_ylabel("Probabilité (%)")
    ax.set_title(
        "Probabilités estimées par le modèle",
        fontweight="bold",
        color="#17324D"
    )

    for barre, valeur in zip(barres, valeurs):
        ax.text(
            barre.get_x() + barre.get_width() / 2,
            valeur + 3,
            f"{valeur:.1f} %",
            ha="center",
            fontweight="bold"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    figure = graphique_survie()
    figure.savefig(
        CHEMIN_GRAPHIQUE,
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()
