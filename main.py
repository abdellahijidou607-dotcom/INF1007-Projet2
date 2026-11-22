"""Point d'entrée : charge le modèle et ses métadonnées, puis lance l'interface."""
import sys

from config import CHEMIN_HISTORIQUE
from exceptions import DonneeInvalideError, FichierIntrouvableError
from historique import Historique
from interface import Application
from modele import ModeleSurvie
from variables import charger_metadonnees, creer_variables


def main() -> int:
    """Démarre l'application.

    Renvoie : 0 si tout s'est bien passé, 1 si le modèle ou ses métadonnées manquent.
    """
    try:
        variables = creer_variables(charger_metadonnees())
        modele = ModeleSurvie.charger()
    except (FichierIntrouvableError, DonneeInvalideError) as erreur:
        print(f"Impossible de démarrer : {erreur}")
        print("Entraînez d'abord le modèle avec : python modele.py")
        return 1

    application = Application(modele, variables, Historique(CHEMIN_HISTORIQUE))
    application.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
