"""Interface graphique Tkinter.

L'interface appelle la logique du projet sans la contenir : la validation est
faite par les objets Variable, la prédiction par ModeleSurvie et l'enregistrement
par Historique (composition : Application contient ces trois éléments).
"""
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from exceptions import DonneeInvalideError, FichierIntrouvableError, ModeleNonEntraineError
from historique import Historique
from modele import ModeleSurvie
from variables import Variable, VariableCategorielle
from visualisation import graphique_prediction, graphique_survie

AVERTISSEMENT = ("Outil pédagogique (projet INF1007) : les prédictions ne constituent "
                 "ni une garantie ni une aide à la décision financière.")


class Application(tk.Tk):
    """Fenêtre principale de prédiction de survie des restaurants."""

    FOND = "#F3F6FA"
    BLEU = "#17324D"
    VERT = "#16816F"
    BLANC = "#FFFFFF"

    def __init__(self, modele: ModeleSurvie, variables: list[Variable],
                 historique: Historique) -> None:
        super().__init__()
        self._modele = modele
        self._variables = list(variables)
        self._historique = historique
        self._champs: dict[str, tk.StringVar] = {}
        self._figure: Figure | None = None
        self._canvas_graphique: FigureCanvasTkAgg | None = None
        self._defilement: tk.Canvas | None = None
        self._zone_souris: tk.Frame | None = None

        self.title("Restaurant Predictor - INF1007")
        self.geometry("1200x800")
        self.minsize(950, 620)
        self.configure(bg=self.FOND)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TEntry", padding=5)
        style.configure("TCombobox", padding=5)

        self._creer_entete()
        contenu = self._creer_contenu()
        self._creer_formulaire(contenu)
        self._creer_zone_resultats(contenu)
        self._afficher_graphique_initial()
        self.bind_all("<MouseWheel>", self._defiler_souris)

    def _creer_entete(self) -> None:
        """Crée le bandeau supérieur avec le titre."""
        entete = tk.Frame(self, bg=self.BLEU, height=90)
        entete.pack(fill="x")
        entete.pack_propagate(False)
        tk.Label(entete, text="RESTAURANT SURVIVAL PREDICTOR", font=("Arial", 20, "bold"),
                 bg=self.BLEU, fg=self.BLANC).pack(anchor="w", padx=25, pady=(15, 2))
        tk.Label(entete, text="INF1007 | Prédiction de survie à 3 ans", font=("Arial", 11),
                 bg=self.BLEU, fg="#D8E5F0").pack(anchor="w", padx=25)

    def _creer_contenu(self) -> tk.Frame:
        """Crée la zone principale à deux colonnes et la renvoie."""
        contenu = tk.Frame(self, bg=self.FOND)
        contenu.pack(fill="both", expand=True, padx=18, pady=18)
        contenu.columnconfigure(0, weight=1)
        contenu.columnconfigure(1, weight=2)
        contenu.rowconfigure(0, weight=1)
        return contenu

    def _creer_formulaire(self, contenu: tk.Frame) -> None:
        """Crée la colonne de gauche : un champ par variable et les boutons."""
        gauche = tk.Frame(contenu, bg=self.BLANC)
        gauche.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tk.Label(gauche, text="Caractéristiques du restaurant", font=("Arial", 14, "bold"),
                 bg=self.BLANC, fg=self.BLEU).pack(anchor="w", padx=15, pady=15)

        zone = tk.Frame(gauche, bg=self.BLANC)
        zone.pack(fill="both", expand=True, padx=10)
        self._zone_souris = zone
        self._defilement = tk.Canvas(zone, bg=self.BLANC, highlightthickness=0)
        barre = ttk.Scrollbar(zone, orient="vertical", command=self._defilement.yview)
        formulaire = tk.Frame(self._defilement, bg=self.BLANC)
        fenetre = self._defilement.create_window((0, 0), window=formulaire, anchor="nw")
        formulaire.bind("<Configure>", lambda e: self._defilement.configure(
            scrollregion=self._defilement.bbox("all")))
        self._defilement.bind("<Configure>", lambda e: self._defilement.itemconfigure(
            fenetre, width=e.width))
        self._defilement.configure(yscrollcommand=barre.set)
        barre.pack(side="right", fill="y")
        self._defilement.pack(side="left", fill="both", expand=True)

        for variable in self._variables:
            self._ajouter_champ(formulaire, variable)

        actions = tk.Frame(gauche, bg=self.BLANC)
        actions.pack(fill="x", padx=15, pady=12)
        tk.Button(actions, text="PRÉDIRE LA SURVIE", command=self.predire, bg=self.VERT,
                  fg="white", font=("Arial", 11, "bold"), relief="flat", cursor="hand2",
                  pady=10).pack(fill="x", pady=(0, 8))
        tk.Button(actions, text="Consulter l'historique", command=self.voir_historique,
                  bg="#E8EFF5", fg=self.BLEU, font=("Arial", 10), relief="flat",
                  cursor="hand2", pady=8).pack(fill="x")

    def _creer_zone_resultats(self, contenu: tk.Frame) -> None:
        """Crée la colonne de droite : résultat, graphique et avertissement."""
        droite = tk.Frame(contenu, bg=self.BLANC)
        droite.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        tk.Label(droite, text="Analyse et résultats", font=("Arial", 15, "bold"),
                 bg=self.BLANC, fg=self.BLEU).pack(anchor="w", padx=20, pady=(18, 10))
        self._resultat = tk.Label(droite, text="En attente d'une prédiction", bg="#E8F4F1",
                                  fg=self.BLEU, font=("Arial", 13, "bold"), padx=15, pady=18,
                                  wraplength=500, justify="center")
        self._resultat.pack(fill="x", padx=20, pady=8)
        self._zone_graphique = tk.Frame(droite, bg=self.BLANC)
        self._zone_graphique.pack(fill="both", expand=True, padx=15, pady=10)
        tk.Label(droite, text=AVERTISSEMENT, bg=self.BLANC, fg="#66788A",
                 font=("Arial", 9), wraplength=550).pack(pady=12)

    def _ajouter_champ(self, parent: tk.Frame, variable: Variable) -> None:
        """Ajoute un champ adapté au type de la variable (liste ou champ texte)."""
        groupe = tk.Frame(parent, bg=self.BLANC)
        groupe.pack(fill="x", padx=12, pady=6)
        tk.Label(groupe, text=variable.libelle, bg=self.BLANC, fg=self.BLEU,
                 font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 4))
        texte = tk.StringVar()
        self._champs[variable.nom] = texte
        if isinstance(variable, VariableCategorielle):
            ttk.Combobox(groupe, textvariable=texte, values=variable.choix,
                         state="readonly").pack(fill="x")
            texte.set(variable.choix[0])
        else:
            ttk.Entry(groupe, textvariable=texte).pack(fill="x")
            tk.Label(groupe, text=variable.description(), bg=self.BLANC, fg="#798899",
                     font=("Arial", 8)).pack(anchor="w")

    def _defiler_souris(self, event: tk.Event) -> None:
        """Fait défiler le formulaire avec la molette, seulement au-dessus de lui."""
        if self._defilement is None or event.delta == 0:
            return
        widget = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
        while widget is not None and widget != self._zone_souris:
            widget = widget.master
        if widget is None:
            return
        self._defilement.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def lire_valeurs(self) -> dict:
        """Valide chaque champ avec sa Variable (polymorphisme).

        Renvoie : un dict {nom_colonne: valeur convertie}.
        Échoue : DonneeInvalideError au premier champ invalide.
        """
        return {v.nom: v.valider(self._champs[v.nom].get()) for v in self._variables}

    def predire(self) -> None:
        """Valide les saisies, appelle le modèle et affiche le résultat."""
        try:
            valeurs = self.lire_valeurs()
        except DonneeInvalideError as erreur:
            messagebox.showerror("Erreur de saisie", str(erreur))
            return
        try:
            classe, probabilite = self._modele.predire(valeurs)
        except (DonneeInvalideError, ModeleNonEntraineError, ValueError) as erreur:
            messagebox.showerror("Erreur du modèle", str(erreur))
            return
        self._afficher_resultat(classe, probabilite)
        try:
            self._historique.ajouter(valeurs, classe, probabilite)
        except (OSError, DonneeInvalideError) as erreur:
            messagebox.showwarning("Historique", f"Prédiction non enregistrée : {erreur}")

    def _afficher_resultat(self, classe: int, probabilite: float) -> None:
        """Affiche le résultat en langage clair et le graphique des probabilités."""
        if classe == 1:
            message, couleur = "Survie probable à 3 ans", "#DDF3EA"
        else:
            message, couleur = "Fermeture probable avant 3 ans", "#FCE8E4"
        self._resultat.config(text=f"{message}\nProbabilité de survie : {probabilite * 100:.1f} %",
                              bg=couleur)
        self._afficher_graphique(graphique_prediction(probabilite))

    def _afficher_graphique_initial(self) -> None:
        """Affiche le graphique d'exploration, ou un message si le CSV manque."""
        try:
            self._afficher_graphique(graphique_survie())
        except (FichierIntrouvableError, DonneeInvalideError, OSError) as erreur:
            tk.Label(self._zone_graphique, text=f"Graphique indisponible : {erreur}",
                     bg=self.BLANC, fg="#66788A", wraplength=500).pack(pady=20)

    def _afficher_graphique(self, figure: Figure) -> None:
        """Remplace le graphique affiché par une nouvelle figure."""
        if self._canvas_graphique is not None:
            self._canvas_graphique.get_tk_widget().destroy()
        if self._figure is not None:
            plt.close(self._figure)
        self._figure = figure
        self._canvas_graphique = FigureCanvasTkAgg(figure, master=self._zone_graphique)
        self._canvas_graphique.draw()
        self._canvas_graphique.get_tk_widget().pack(fill="both", expand=True)

    def voir_historique(self) -> None:
        """Ouvre une fenêtre listant les prédictions enregistrées."""
        try:
            entrees = self._historique.lire()
        except (OSError, DonneeInvalideError) as erreur:
            messagebox.showerror("Erreur d'historique", str(erreur))
            return
        fenetre = tk.Toplevel(self)
        fenetre.title("Historique des prédictions")
        fenetre.geometry("750x400")
        colonnes = ("Date", "Résultat", "Probabilité de survie")
        tableau = ttk.Treeview(fenetre, columns=colonnes, show="headings")
        for colonne in colonnes:
            tableau.heading(colonne, text=colonne)
            tableau.column(colonne, width=220, anchor="center")
        tableau.pack(fill="both", expand=True, padx=15, pady=15)
        for entree in entrees:
            tableau.insert("", "end", values=(entree["date"], entree["resultat"],
                                              f"{entree['probabilite'] * 100:.1f} %"))
        if not entrees:
            tableau.insert("", "end", values=("Aucune prédiction", "", ""))
