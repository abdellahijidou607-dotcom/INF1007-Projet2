
import tkinter as tk
from tkinter import messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from visualisation import graphique_survie


class Application(tk.Tk):

    def __init__(self):
        super().__init__()

        # Configuration de la fenêtre
        self.title("Prédiction de survie des restaurants")
        self.geometry("900x700")

        # Titre
        titre = tk.Label(
            self,
            text="Prédiction de survie des restaurants",
            font=("Arial", 18, "bold")
        )
        titre.pack(pady=20)

        # Bouton pour afficher le graphique
        bouton_graphique = tk.Button(
            self,
            text="Afficher le graphique",
            font=("Arial", 12),
            command=self.afficher_graphique
        )
        bouton_graphique.pack(pady=10)

        # Zone qui contiendra le graphique
        self.zone_graphique = tk.Frame(self)
        self.zone_graphique.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )

        self.canvas = None
        self.figure = None

    def afficher_graphique(self):

        try:
            # Supprimer l'ancien graphique
            if self.canvas is not None:
                self.canvas.get_tk_widget().destroy()
                plt.close(self.figure)

            # Créer le graphique
            self.figure = graphique_survie()

            # Intégrer le graphique dans Tkinter
            self.canvas = FigureCanvasTkAgg(
                self.figure,
                master=self.zone_graphique
            )

            self.canvas.draw()

            self.canvas.get_tk_widget().pack(
                fill="both",
                expand=True
            )

        except Exception as erreur:
            messagebox.showerror(
                "Erreur",
                str(erreur)
            )


if __name__ == "__main__":
    application = Application()
    application.mainloop()
