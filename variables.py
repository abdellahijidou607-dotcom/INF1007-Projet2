"""Variables d'entrée du modèle : héritage et polymorphisme (chapitres 9 à 11).

Variable est une classe abstraite. VariableNumerique et VariableCategorielle
redéfinissent toutes les deux valider() et description(), chacune à sa façon.
"""
import json
import math
from abc import ABC, abstractmethod

from config import CHEMIN_META, verifier_fichier
from exceptions import DonneeInvalideError

LIBELLES = {
    "cuisine": "Type de cuisine",
    "location_type": "Emplacement",
    "is_franchise": "Restaurant franchisé (0 = non, 1 = oui)",
    "owner_experience_years": "Expérience du propriétaire (années)",
    "foot_traffic": "Fréquentation (0 à 10)",
    "competition_density": "Densité de concurrence (0 à 10)",
    "rent_to_revenue": "Ratio loyer / revenus",
    "starting_capital_months": "Capital de départ (mois)",
    "labor_cost_ratio": "Ratio coût du personnel",
    "price_tier": "Catégorie de prix (1 à 4)",
    "seats": "Nombre de places",
    "menu_size": "Nombre de plats au menu",
    "has_marketing": "Marketing (0 = non, 1 = oui)",
    "avg_review_rating": "Note moyenne",
    "review_count": "Nombre d'avis",
}

CLES_META_OBLIGATOIRES = ("colonnes", "types", "categories", "plages")


class Variable(ABC):
    """Variable d'entrée saisie par l'utilisateur (classe abstraite)."""

    def __init__(self, nom: str, libelle: str) -> None:
        self._nom = nom
        self._libelle = libelle

    @property
    def nom(self) -> str:
        """Nom de la colonne dans le jeu de données."""
        return self._nom

    @property
    def libelle(self) -> str:
        """Texte affiché à l'utilisateur."""
        return self._libelle

    def _verifier_non_vide(self, texte: str) -> str:
        """Retire les espaces et lève DonneeInvalideError si la saisie est vide."""
        texte = texte.strip()
        if not texte:
            raise DonneeInvalideError(f"Veuillez remplir : {self._libelle}")
        return texte

    @abstractmethod
    def valider(self, texte: str) -> int | float | str:
        """Convertit la saisie texte en valeur utilisable par le modèle.

        Reçoit : le texte saisi.
        Renvoie : la valeur convertie.
        Échoue : lève DonneeInvalideError si la saisie est invalide.
        """

    @abstractmethod
    def description(self) -> str:
        """Renvoie une aide courte décrivant les valeurs acceptées."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nom={self._nom!r})"

    def __str__(self) -> str:
        return f"{self._libelle} : {self.description()}"

    def __eq__(self, autre: object) -> bool:
        return type(self) is type(autre) and self._nom == autre.nom

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._nom))


class VariableNumerique(Variable):
    """Variable numérique bornée par un intervalle [minimum, maximum]."""

    def __init__(self, nom: str, libelle: str, minimum: float, maximum: float,
                 entier: bool = False) -> None:
        super().__init__(nom, libelle)
        if minimum > maximum:
            raise DonneeInvalideError(f"Intervalle invalide pour {nom}")
        self._minimum = minimum
        self._maximum = maximum
        self._entier = entier

    @property
    def minimum(self) -> float:
        return self._minimum

    @property
    def maximum(self) -> float:
        return self._maximum

    def valider(self, texte: str) -> int | float:
        """Convertit en nombre et vérifie l'intervalle.

        Reçoit : le texte saisi (la virgule décimale est acceptée).
        Renvoie : un int si la variable est entière, sinon un float.
        Échoue : DonneeInvalideError si vide, non numérique, non entier
        alors qu'un entier est attendu, ou hors intervalle.
        """
        texte = self._verifier_non_vide(texte).replace(",", ".")
        try:
            valeur = float(texte)
        except ValueError:
            raise DonneeInvalideError(f"Nombre invalide : {self._libelle}") from None
        if not math.isfinite(valeur):
            raise DonneeInvalideError(f"Nombre invalide : {self._libelle}")
        if self._entier and not valeur.is_integer():
            raise DonneeInvalideError(f"Nombre entier attendu : {self._libelle}")
        if not self._minimum <= valeur <= self._maximum:
            raise DonneeInvalideError(
                f"{self._libelle} doit être entre {self._minimum:g} et {self._maximum:g}"
            )
        return int(valeur) if self._entier else valeur

    def description(self) -> str:
        return f"valeurs acceptées : {self._minimum:g} à {self._maximum:g}"


class VariableCategorielle(Variable):
    """Variable dont la valeur doit faire partie d'une liste de choix."""

    def __init__(self, nom: str, libelle: str, choix: list[str],
                 convertir_en_entier: bool = False) -> None:
        super().__init__(nom, libelle)
        if not choix:
            raise DonneeInvalideError(f"Aucun choix possible pour {nom}")
        self._choix = list(choix)
        self._convertir_en_entier = convertir_en_entier

    @property
    def choix(self) -> list[str]:
        """Copie de la liste des choix (la liste interne reste protégée)."""
        return list(self._choix)

    def valider(self, texte: str) -> int | str:
        """Vérifie que la saisie fait partie des choix.

        Reçoit : le texte choisi.
        Renvoie : le texte, ou un int pour les variables binaires 0/1.
        Échoue : DonneeInvalideError si vide ou hors des choix.
        """
        texte = self._verifier_non_vide(texte)
        if texte not in self._choix:
            raise DonneeInvalideError(f"Choix invalide pour {self._libelle} : {texte}")
        return int(texte) if self._convertir_en_entier else texte

    def description(self) -> str:
        return f"{len(self._choix)} choix possibles"


def charger_metadonnees(chemin: str = CHEMIN_META) -> dict:
    """Lit le fichier JSON de métadonnées du modèle.

    Reçoit : le chemin du fichier JSON.
    Renvoie : le dictionnaire des métadonnées.
    Échoue : FichierIntrouvableError si le fichier n'existe pas,
    DonneeInvalideError si le JSON est illisible ou incomplet.
    """
    verifier_fichier(chemin)
    try:
        with open(chemin, encoding="utf-8") as fichier:
            meta = json.load(fichier)
    except json.JSONDecodeError as erreur:
        raise DonneeInvalideError(f"Métadonnées illisibles : {erreur}") from None
    manquantes = [cle for cle in CLES_META_OBLIGATOIRES if cle not in meta]
    if manquantes:
        raise DonneeInvalideError(f"Clés manquantes dans les métadonnées : {manquantes}")
    return meta


def creer_variables(meta: dict) -> list[Variable]:
    """Construit la liste des variables d'entrée à partir des métadonnées.

    Reçoit : le dictionnaire des métadonnées du modèle.
    Renvoie : une Variable par colonne, dans l'ordre attendu par le modèle.
    Échoue : DonneeInvalideError si une colonne n'a ni catégories ni intervalle.
    """
    binaires = set(meta.get("binaires", []))
    variables: list[Variable] = []
    for nom in meta["colonnes"]:
        libelle = LIBELLES.get(nom, nom)
        if nom in meta["categories"]:
            variables.append(VariableCategorielle(nom, libelle, meta["categories"][nom]))
        elif nom in binaires:
            variables.append(VariableCategorielle(nom, libelle, ["0", "1"], convertir_en_entier=True))
        elif nom in meta["plages"]:
            minimum, maximum = meta["plages"][nom]
            entier = str(meta["types"].get(nom, "")).startswith("int")
            variables.append(VariableNumerique(nom, libelle, minimum, maximum, entier))
        else:
            raise DonneeInvalideError(f"Aucune description pour la colonne {nom}")
    return variables
