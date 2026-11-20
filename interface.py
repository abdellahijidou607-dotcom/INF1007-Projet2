
import csv
import json
import math
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from visualisation import graphique_survie, graphique_prediction


DOSSIER = Path(__file__).resolve().parent
CSV = DOSSIER / "restaurant_survival.csv"
HISTORIQUE = DOSSIER / "historique_predictions.csv"
METADONNEES = DOSSIER / "metadonnees.json"

INTERDITES = {
    "restaurant_id",
    "years_survived",
    "failure_reason",
    "survived_3y"
}

NOMS = {
    "cuisine": "Type de cuisine",
    "location_type": "Emplacement",
    "is_franchise": "Restaurant franchisé",
    "owner_experience_years": "Expérience du propriétaire",
    "foot_traffic": "Fréquentation",
    "competition_density": "Densité de concurrence",
    "rent_to_revenue": "Ratio loyer / revenus",
    "starting_capital_months": "Capital de départ (mois)",
    "labor_cost_ratio": "Ratio coût du personnel",
    "price_tier": "Catégorie de prix",
    "seats": "Nombre de places",
    "menu_size": "Nombre de plats au menu",
    "has_marketing": "Marketing",
    "avg_review_rating": "Note moyenne",
    "review_count": "Nombre d'avis"
}


class Application(tk.Tk):
    """Interface graphique de prédiction des restaurants."""

    def __init__(self, predicteur=None):
        super().__init__()

        self.predicteur = predicteur

        self.title("Restaurant Predictor - INF1007")
        self.geometry("1200x800")
        self.minsize(950, 620)

        self.fond = "#F3F6FA"
        self.bleu = "#17324D"
        self.vert = "#16816F"
        self.blanc = "#FFFFFF"

        self.configure(bg=self.fond)

        self.variables = {}
        self.figure = None
        self.canvas_graphique = None

        # Le formulaire qui pourra défiler
        self.defilement = None
        self.zone_souris = None

        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TEntry", padding=5)
        self.style.configure("TCombobox", padding=5)

        self.definitions = self.charger_definitions()

        self.creer_interface()
        self.afficher_graphique(graphique_survie())

        # Activer la molette uniquement dans la zone du formulaire
        self.bind_all("<MouseWheel>", self.defiler_souris)

    def charger_definitions(self) -> dict:
        """Charge les variables d'entrée et leurs types."""

        donnees = pd.read_csv(CSV)

        colonnes = [
            nom for nom in donnees.columns
            if nom not in INTERDITES
        ]

        if METADONNEES.exists():
            with open(
                METADONNEES, "r", encoding="utf-8"
            ) as fichier:
                meta = json.load(fichier)

            noms_modele = meta.get(
                "colonnes",
                meta.get(
                    "features",
                    meta.get("feature_names")
                )
            )

            if isinstance(noms_modele, list) and all(
                isinstance(nom, str)
                for nom in noms_modele
            ):
                colonnes = [
                    nom for nom in noms_modele
                    if nom not in INTERDITES
                ]

        definitions = {}

        for nom in colonnes:
            if nom not in donnees.columns:
                raise ValueError(
                    f"Colonne inconnue : {nom}"
                )

            serie = donnees[nom].dropna()

            if serie.empty:
                raise ValueError(
                    f"Aucune valeur pour {nom}"
                )

            if (
                not pd.api.types.is_numeric_dtype(serie)
                or nom in ("is_franchise", "has_marketing")
            ):
                choix = sorted(
                    str(v) for v in serie.unique()
                )

                definitions[nom] = {
                    "type": "categorie",
                    "choix": choix
                }

            else:
                entier = pd.api.types.is_integer_dtype(
                    serie
                )

                definitions[nom] = {
                    "type": "entier" if entier else "reel",
                    "min": float(serie.min()),
                    "max": float(serie.max())
                }

        return definitions

    def creer_interface(self) -> None:
        """Crée les différents éléments de la fenêtre."""

        # Bandeau supérieur
        entete = tk.Frame(
            self,
            bg=self.bleu,
            height=90
        )
        entete.pack(fill="x")
        entete.pack_propagate(False)

        tk.Label(
            entete,
            text="RESTAURANT SURVIVAL PREDICTOR",
            font=("Arial", 20, "bold"),
            bg=self.bleu,
            fg=self.blanc
        ).pack(
            anchor="w",
            padx=25,
            pady=(15, 2)
        )

        tk.Label(
            entete,
            text="INF1007 | Prédiction de survie à 3 ans",
            font=("Arial", 11),
            bg=self.bleu,
            fg="#D8E5F0"
        ).pack(anchor="w", padx=25)

        # Contenu principal
        contenu = tk.Frame(
            self,
            bg=self.fond
        )
        contenu.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=18
        )

        contenu.columnconfigure(0, weight=1)
        contenu.columnconfigure(1, weight=2)
        contenu.rowconfigure(0, weight=1)

        # Partie gauche
        gauche = tk.Frame(
            contenu,
            bg=self.blanc
        )
        gauche.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10)
        )

        tk.Label(
            gauche,
            text="Caractéristiques du restaurant",
            font=("Arial", 14, "bold"),
            bg=self.blanc,
            fg=self.bleu
        ).pack(
            anchor="w",
            padx=15,
            pady=15
        )

        # Zone défilante
        zone = tk.Frame(
            gauche,
            bg=self.blanc
        )
        zone.pack(
            fill="both",
            expand=True,
            padx=10
        )

        self.defilement = tk.Canvas(
            zone,
            bg=self.blanc,
            highlightthickness=0
        )

        barre = ttk.Scrollbar(
            zone,
            orient="vertical",
            command=self.defilement.yview
        )

        formulaire = tk.Frame(
            self.defilement,
            bg=self.blanc
        )

        fenetre_formulaire = (
            self.defilement.create_window(
                (0, 0),
                window=formulaire,
                anchor="nw"
            )
        )

        formulaire.bind(
            "<Configure>",
            lambda e: self.defilement.configure(
                scrollregion=self.defilement.bbox("all")
            )
        )

        self.defilement.bind(
            "<Configure>",
            lambda e: self.defilement.itemconfigure(
                fenetre_formulaire,
                width=e.width
            )
        )

        self.defilement.configure(
            yscrollcommand=barre.set
        )

        barre.pack(
            side="right",
            fill="y"
        )

        self.defilement.pack(
            side="left",
            fill="both",
            expand=True
        )

        # Mémoriser la zone du formulaire
        self.zone_souris = zone

        for nom, definition in self.definitions.items():
            self.ajouter_champ(
                formulaire,
                nom,
                definition
            )

        # Boutons sous le formulaire
        actions = tk.Frame(
            gauche,
            bg=self.blanc
        )
        actions.pack(
            fill="x",
            padx=15,
            pady=12
        )

        tk.Button(
            actions,
            text="PRÉDIRE LA SURVIE",
            command=self.predire,
            bg=self.vert,
            fg="white",
            font=("Arial", 11, "bold"),
            relief="flat",
            cursor="hand2",
            pady=10
        ).pack(
            fill="x",
            pady=(0, 8)
        )

        tk.Button(
            actions,
            text="Consulter l'historique",
            command=self.voir_historique,
            bg="#E8EFF5",
            fg=self.bleu,
            font=("Arial", 10),
            relief="flat",
            cursor="hand2",
            pady=8
        ).pack(fill="x")

        # Partie droite
        droite = tk.Frame(
            contenu,
            bg=self.blanc
        )
        droite.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(10, 0)
        )

        tk.Label(
            droite,
            text="Analyse et résultats",
            font=("Arial", 15, "bold"),
            bg=self.blanc,
            fg=self.bleu
        ).pack(
            anchor="w",
            padx=20,
            pady=(18, 10)
        )

        self.resultat = tk.Label(
            droite,
            text="En attente d'une prédiction",
            bg="#E8F4F1",
            fg=self.bleu,
            font=("Arial", 13, "bold"),
            padx=15,
            pady=18,
            wraplength=500,
            justify="center"
        )
        self.resultat.pack(
            fill="x",
            padx=20,
            pady=8
        )

        self.zone_graphique = tk.Frame(
            droite,
            bg=self.blanc
        )
        self.zone_graphique.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        tk.Label(
            droite,
            text=(
                "Outil pédagogique : les prédictions "
                "ne constituent pas une garantie."
            ),
            bg=self.blanc,
            fg="#66788A",
            font=("Arial", 9),
            wraplength=550
        ).pack(pady=12)

    def defiler_souris(self, event) -> None:
        """Défile avec la molette dans le formulaire seulement."""

        if self.defilement is None:
            return

        # Trouver la position actuelle de la souris
        x = self.winfo_pointerx()
        y = self.winfo_pointery()

        # Trouver le widget sous le pointeur
        widget = self.winfo_containing(x, y)

        if widget is None:
            return

        # Vérifier si le pointeur se trouve dans le formulaire
        courant = widget

        while courant is not None:
            if courant == self.zone_souris:
                break

            courant = courant.master

        if courant is None:
            return

        # Windows : delta positif vers le haut
        if event.delta > 0:
            deplacement = -1
        elif event.delta < 0:
            deplacement = 1
        else:
            return

        self.defilement.yview_scroll(
            deplacement, "units"
        )

    def ajouter_champ(
        self,
        parent: tk.Frame,
        nom: str,
        definition: dict
    ) -> None:
        """Ajoute un champ à la zone défilante."""

        groupe = tk.Frame(
            parent,
            bg=self.blanc
        )
        groupe.pack(
            fill="x",
            padx=12,
            pady=6
        )

        tk.Label(
            groupe,
            text=NOMS.get(nom, nom),
            bg=self.blanc,
            fg=self.bleu,
            font=("Arial", 10, "bold")
        ).pack(
            anchor="w",
            pady=(0, 4)
        )

        variable = tk.StringVar()
        self.variables[nom] = variable

        if definition["type"] == "categorie":

            valeurs = definition["choix"]

            ttk.Combobox(
                groupe,
                textvariable=variable,
                values=valeurs,
                state="readonly"
            ).pack(fill="x")

            if valeurs:
                variable.set(valeurs[0])

        else:
            ttk.Entry(
                groupe,
                textvariable=variable
            ).pack(fill="x")

            tk.Label(
                groupe,
                text=(
                    f"Valeurs observées : "
                    f"{definition['min']:g} à "
                    f"{definition['max']:g}"
                ),
                bg=self.blanc,
                fg="#798899",
                font=("Arial", 8)
            ).pack(anchor="w")

    def lire_valeurs(self) -> dict:
        """Valide les champs et retourne les valeurs."""

        valeurs = {}

        for nom, definition in self.definitions.items():

            texte = self.variables[nom].get().strip()

            if not texte:
                raise ValueError(
                    f"Veuillez remplir : "
                    f"{NOMS.get(nom, nom)}"
                )

            if definition["type"] == "categorie":

                if texte not in definition["choix"]:
                    raise ValueError(
                        f"Catégorie invalide : {nom}"
                    )

                if nom in (
                    "is_franchise",
                    "has_marketing"
                ):
                    valeurs[nom] = int(texte)
                else:
                    valeurs[nom] = texte

            else:
                try:
                    if definition["type"] == "entier":
                        valeur = int(texte)
                    else:
                        valeur = float(texte)

                except ValueError:
                    raise ValueError(
                        f"Nombre invalide : "
                        f"{NOMS.get(nom, nom)}"
                    )

                if not math.isfinite(valeur):
                    raise ValueError(
                        f"Valeur non finie : {nom}"
                    )

                if not (
                    definition["min"]
                    <= valeur
                    <= definition["max"]
                ):
                    raise ValueError(
                        f"{NOMS.get(nom, nom)} doit être entre "
                        f"{definition['min']:g} et "
                        f"{definition['max']:g}"
                    )

                valeurs[nom] = valeur

        return valeurs

    def predire(self) -> None:
        """Vérifie les saisies et appelle le modèle."""

        try:
            valeurs = self.lire_valeurs()

        except ValueError as erreur:
            messagebox.showerror(
                "Erreur de saisie",
                str(erreur)
            )
            return

        if self.predicteur is None:
            self.resultat.config(
                text="Interface prête - modèle non connecté",
                bg="#FFF1DB"
            )

            messagebox.showinfo(
                "Modèle non disponible",
                "Les champs sont valides. "
                "Il reste à connecter le modèle entraîné."
            )
            return

        try:
            classe, probabilite = self.predicteur(valeurs)

            classe = int(classe)
            probabilite = float(probabilite)

            if classe not in (0, 1):
                raise ValueError(
                    "Classe prédite invalide."
                )

            if not (
                math.isfinite(probabilite)
                and 0 <= probabilite <= 1
            ):
                raise ValueError(
                    "Probabilité invalide."
                )

            if classe == 1:
                message = "Survie probable à 3 ans"
                couleur = "#DDF3EA"
            else:
                message = "Fermeture probable avant 3 ans"
                couleur = "#FCE8E4"

            self.resultat.config(
                text=(
                    f"{message}\n"
                    f"Probabilité de survie : "
                    f"{probabilite * 100:.1f} %"
                ),
                bg=couleur
            )

            self.afficher_graphique(
                graphique_prediction(probabilite)
            )

            self.sauvegarder_historique(
                valeurs,
                classe,
                probabilite
            )

        except Exception as erreur:
            messagebox.showerror(
                "Erreur du modèle",
                str(erreur)
            )

    def afficher_graphique(self, figure) -> None:
        """Affiche une figure dans la zone de résultats."""

        if self.canvas_graphique is not None:
            self.canvas_graphique.get_tk_widget().destroy()

        if self.figure is not None:
            plt.close(self.figure)

        self.figure = figure

        self.canvas_graphique = FigureCanvasTkAgg(
            figure,
            master=self.zone_graphique
        )

        self.canvas_graphique.draw()

        self.canvas_graphique.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    def sauvegarder_historique(
        self,
        valeurs: dict,
        classe: int,
        probabilite: float
    ) -> None:
        """Enregistre la prédiction dans un fichier CSV."""

        nouveau = (
            not HISTORIQUE.exists()
            or HISTORIQUE.stat().st_size == 0
        )

        with open(
            HISTORIQUE,
            "a",
            newline="",
            encoding="utf-8"
        ) as fichier:

            champs = [
                "date",
                "prediction",
                "probabilite_survie",
                "donnees"
            ]

            writer = csv.DictWriter(
                fichier,
                fieldnames=champs
            )

            if nouveau:
                writer.writeheader()

            writer.writerow({
                "date": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "prediction": classe,
                "probabilite_survie": probabilite,
                "donnees": json.dumps(
                    valeurs,
                    ensure_ascii=False
                )
            })

    def voir_historique(self) -> None:
        """Affiche les prédictions enregistrées."""

        fenetre = tk.Toplevel(self)
        fenetre.title("Historique des prédictions")
        fenetre.geometry("750x400")

        colonnes = (
            "Date",
            "Résultat",
            "Probabilité de survie"
        )

        tableau = ttk.Treeview(
            fenetre,
            columns=colonnes,
            show="headings"
        )

        for colonne in colonnes:
            tableau.heading(
                colonne,
                text=colonne
            )

            tableau.column(
                colonne,
                width=220,
                anchor="center"
            )

        tableau.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )

        if not HISTORIQUE.exists():
            return

        try:
            with open(
                HISTORIQUE,
                "r",
                newline="",
                encoding="utf-8"
            ) as fichier:

                lecteur = csv.DictReader(fichier)

                for ligne in lecteur:

                    classe = int(
                        ligne["prediction"]
                    )

                    resultat = (
                        "Survie"
                        if classe == 1
                        else "Fermeture"
                    )

                    probabilite = (
                        float(
                            ligne["probabilite_survie"]
                        ) * 100
                    )

                    tableau.insert(
                        "",
                        "end",
                        values=(
                            ligne["date"],
                            resultat,
                            f"{probabilite:.1f} %"
                        )
                    )

        except (
            OSError,
            ValueError,
            KeyError
        ) as erreur:
            messagebox.showerror(
                "Erreur d'historique",
                str(erreur)
            )


if __name__ == "__main__":
    application = Application()
    application.mainloop()
