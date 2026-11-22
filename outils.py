"""Outils génériques (chapitre 7) : décorateur de chronométrage."""
import functools
import time
from typing import Any, Callable


def chronometre(fonction: Callable[..., Any]) -> Callable[..., Any]:
    """Décorateur qui mesure la durée d'exécution d'une fonction.

    Reçoit : la fonction à chronométrer.
    Renvoie : une fonction enveloppe qui se comporte comme l'originale et
    stocke la durée du dernier appel (en secondes) dans son attribut
    derniere_duree.
    Échoue : ne lève rien elle-même ; les exceptions de la fonction passent.
    """

    @functools.wraps(fonction)
    def enveloppe(*args: Any, **kwargs: Any) -> Any:
        debut = time.perf_counter()
        resultat = fonction(*args, **kwargs)
        enveloppe.derniere_duree = time.perf_counter() - debut
        return resultat

    enveloppe.derniere_duree = 0.0
    return enveloppe
