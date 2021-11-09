# pandas : panel data , permet de manipuler et l'analyser des données
from datetime import time
import pandas as pd
# Seaborn est une bibliothèque permettant de créer des graphiques statistiques en Python.
import seaborn as sns
# numpy : destinée à manipuler des matrices ou tableaux multidimensionnels et des fonctions mathématiques opérant
#        sur ces tableaux.

import numpy as np  # numpy : destinée à manipuler des matrices ou tableaux multidimensionnels et des fonctions mathématiques
#                  opérant sur ces tableaux.

# preprocessing : package pour centrer et normaliser les données avant d'effectuer l'ACP.
from sklearn.preprocessing import StandardScaler

# decomposition : Package pour l'analyse en composantes principales de scikit learn.
from sklearn.decomposition import PCA

# matplotlib : pour dessiner des graphiques, des tracés ...
import matplotlib.pyplot as plt
# importer les données de notre fichier dataset.
data = pd.read_csv("./bd-tp1.csv")
bd = pd.read_csv("./bd-tp1.csv")
# informations sur les données avec Pandas
# data.info()

# L'attribut shape renvoie les dimensions de notre data
print(data.shape)
data.set_index('ShortDescrip', inplace=True)
# il y a 8618 échantillons et 40 variables, (Etat : étiquette )

scaled_data = StandardScaler().fit_transform(data)
#pca = pca_var.fit_transform(scaled_data)
pca_var = PCA(n_components=2)
pca = pca_var.fit_transform(scaled_data)
plt.figure(figsize=(10, 7))
sns.scatterplot(data=data, x=pca[:, 0], y=pca[:, 1], s=70,
                palette=['green', 'blue'])
plt.ylabel('deuxième composante principale')
plt.xlabel('premier composant principal')
plt.title('observations actives')
plt.show()
""" plt.figure(figsize=(10, 7))
plt.plot(pca)
plt.ylabel('données transformées')
plt.xlabel('observation')
plt.title(
    'données transformées par les composantes principales (variabilité de 95 %)')
plt.show() """
