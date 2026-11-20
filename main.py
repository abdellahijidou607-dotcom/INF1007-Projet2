
from interface import Application
from modele import ModeleSurvie


def main():
    """Démarre l'application de prédiction."""

    predicteur = ModeleSurvie.charger().predire

    application = Application(
        predicteur=predicteur
    )

    application.mainloop()


if __name__ == "__main__":
    main()
