
from interface import Application


def main():
    """Démarre l'application de prédiction."""

    # Le modèle sera relié quand il sera disponible.
    predicteur = None

    application = Application(
        predicteur=predicteur
    )

    application.mainloop()


if __name__ == "__main__":
    main()
