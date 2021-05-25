import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# On charge le dataset
house_data = pd.read_csv("house.csv")

# On enleve les données  
house_data = house_data[house_data['loyer'] < 10000]
print(house_data.head())


# On décompose le dataset et on le transforme en matrices pour pouvoir effectuer notre calcul
X = np.matrix([np.ones(house_data.shape[0]), house_data['surface'].values]).T
y = np.matrix(house_data['loyer']).T

"""
# apercu de X
# print(X[1:3,:])

print(np.matrix(house_data['loyer']).T)
print(type(np.matrix(house_data['loyer']).T))

plt.plot(house_data['surface'], house_data['loyer'],'ro', markersize =3)
plt.show()
"""

# Ici on aplique la formule pour le calcul de theta en dimension 1
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)



print('Notre modele est donc: y = {} * x + {}'.format(theta.item(1),theta.item(0)))

# Preparation du graphique
plt.title("Prix des loyers parisien en fonction de la surface d'habitation")
plt.xlabel('Surface')
plt.ylabel('Loyer')
# parametrage de l'affichage du nuage de points :
plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=3)

# parametrage de l'affichage de la droite de regression linéaire de 0 à 250 :
plt.plot([0,250], [theta.item(0), theta.item(0) + 250 * theta.item(1)], linestyle='--', c='black',
	label= "Estimation loyer/surface")

plt.legend(loc='lower right');

plt.show()