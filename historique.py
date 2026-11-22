"""Historique des prédictions, enregistré dans un fichier CSV."""
import csv
import json
import os
from datetime import datetime

from exceptions import DonneeInvalideError

CHAMPS = ["date", "prediction", "probabilite_survie", "donnees"]


class Historique:
    """Ajoute et relit les prédictions dans un fichier CSV."""

    def __init__(self, chemin: str) -> None:
        self._chemin = chemin

    @property
    def chemin(self) -> str:
        return self._chemin

    def ajouter(self, valeurs: dict, classe: int, probabilite: float) -> None:
        """Ajoute une prédiction à la fin du fichier.

        Reçoit : les valeurs saisies, la classe prédite et la probabilité de survie.
        Renvoie : rien.
        Échoue : DonneeInvalideError si la classe ou la probabilité est invalide,
        OSError si l'écriture échoue.
        """
        if classe not in (0, 1) or not 0 <= probabilite <= 1:
            raise DonneeInvalideError("Prédiction invalide, rien n'est enregistré.")
        nouveau = not os.path.isfile(self._chemin) or os.path.getsize(self._chemin) == 0
        with open(self._chemin, "a", newline="", encoding="utf-8") as fichier:
            writer = csv.DictWriter(fichier, fieldnames=CHAMPS)
            if nouveau:
                writer.writeheader()
            writer.writerow({
                "date": datetime.now().isoformat(timespec="seconds"),
                "prediction": classe,
                "probabilite_survie": probabilite,
                "donnees": json.dumps(valeurs, ensure_ascii=False),
            })

    def lire(self) -> list[dict]:
        """Relit toutes les prédictions, de la plus récente à la plus ancienne.

        Reçoit : rien.
        Renvoie : une liste de dicts {date, resultat, probabilite} (vide si aucun fichier).
        Échoue : DonneeInvalideError si une ligne du fichier est corrompue.
        """
        if not os.path.isfile(self._chemin):
            return []
        lignes = []
        with open(self._chemin, newline="", encoding="utf-8") as fichier:
            for ligne in csv.DictReader(fichier):
                try:
                    classe = int(ligne["prediction"])
                    proba = float(ligne["probabilite_survie"])
                except (KeyError, TypeError, ValueError):
                    raise DonneeInvalideError("Ligne corrompue dans l'historique.") from None
                lignes.append({
                    "date": ligne["date"],
                    "resultat": "Survie" if classe == 1 else "Fermeture",
                    "probabilite": proba,
                })
        return sorted(lignes, key=lambda entree: entree["date"], reverse=True)

    def __len__(self) -> int:
        return len(self.lire())

    def __repr__(self) -> str:
        return f"Historique(chemin={self._chemin!r})"
