import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

chemin = Path(__file__).resolve().parent / "donnees" / "restaurant_survival.csv"
donnees = pd.read_csv(chemin)

print(donnees.head())
print("Nombre de restaurants :", len(donnees))
print("Colonnes :", donnees.columns.tolist())
