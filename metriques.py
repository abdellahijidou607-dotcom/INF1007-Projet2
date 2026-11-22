"""Métriques calculées de façon vectorisée avec NumPy (chapitre 12)."""
import numpy as np

from exceptions import DonneeInvalideError


def exactitude_numpy(y_vrai: np.ndarray | list, y_pred: np.ndarray | list) -> float:
    """Calcule l'exactitude sans boucle Python.

    Reçoit : les classes réelles et les classes prédites (même longueur).
    Renvoie : la proportion de bonnes prédictions, entre 0 et 1.
    Échoue : lève DonneeInvalideError si les tailles diffèrent ou sont nulles.
    """
    vrai = np.asarray(y_vrai)
    pred = np.asarray(y_pred)
    if vrai.shape != pred.shape or vrai.size == 0:
        raise DonneeInvalideError("Les deux séries doivent avoir la même taille non nulle.")
    return float(np.mean(vrai == pred))


def rappel_par_classe(matrice: np.ndarray | list) -> np.ndarray:
    """Calcule le rappel de chaque classe à partir d'une matrice de confusion.

    Reçoit : une matrice de confusion carrée (lignes = réel, colonnes = prédit).
    Renvoie : un tableau avec, pour chaque classe, diagonale / somme de la ligne.
    Échoue : lève DonneeInvalideError si la matrice n'est pas carrée.
    """
    m = np.asarray(matrice, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise DonneeInvalideError("La matrice de confusion doit être carrée.")
    return np.diag(m) / m.sum(axis=1)
