
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


DOSSIER = Path(__file__).resolve().parent
CSV = DOSSIER / "donnees" / "restaurant_survival.csv"


def graphique_survie():
    """Crée un graphique présentant les restaurants du CSV."""
    donnees = pd.read_csv(CSV)

    if "survived_3y" not in donnees.columns:
        raise ValueError("Colonne survived_3y introuvable.")

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


def graphique_prediction(probabilite: float):
    """Crée un graphique de probabilité de survie."""
    if not 0 <= probabilite <= 1:
        raise ValueError("Probabilité invalide.")

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
        DOSSIER / "graphique_survie.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()
