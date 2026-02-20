import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#chargement des fichiers 
df = pd.read_csv("titanic.csv") 

#Affichons les 5 premieres lignes des données 
print(f"les cinq premieres lignes du dataset: \n{df.head(5)}")

#Affichons les informations generales sur le dataset
print(f"{df.info()}") #pour afficher les informations generales sur les colonnes
print(f"{df.describe()}") #pour afficher les statistiques du dataset 

#les dimensions du dataset
print(f"le dataset comporte {df.shape[0]} lignes et {df.shape[1]} colonnes ")

#identifions les colonnes contenant des valeurs manquantes
print(f"{df.isnull()}")

#Remplacons les valeurs manquantes de la colonne Age par l'age median
age_median = df["Age"].median()
print(f"l'age median: {age_median}")
df = df["Age"].fillna(age_median)

#supprimons les lignes ou la colonne Embarked est manquante
df = df.dropna(subset="Embarked")

#creons une nouvelle colonne FamilySize qui represente la taille de la famille (SibSp+Parch+1)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
print(df)

#Analyse statistique avec Numpy:
#calculons l'age moyen des passagers
age_moyen = np.nanmean(df["Age"])
print(f"l'age moyen des passagers est de: {age_moyen}")

#calculons l'age median des passgers 
age_medians = np.nanmedian(df["Age"])
print(f"l'age median des passagers est de: {age_medians}")

#calculons l'ecart-type de l'age 
ecart_type = np.nanstd(df["Age"])
print(f"l'ecart-type de l'age est: {ecart_type}")

#le prix maximum et minimum des billets
prix_max = np.nanmax(df["Fare"])
prix_min = np.nanmin(df["Fare"])
print(f"le prix maximum des billets est de: {prix_max} et le prix minimum est de: {prix_min}")

#calculons le taux de survie global 
total_survie = df["Survived"].sum()
moyen_survecu = (total_survie/df.shape[0])*100
print(f"{total_survie} personnes ont survecu sur {df.shape[0]}")
print(f"le taux de survie global est de: {moyen_survecu}")

#calculons le taux de survie par sexe 
taux_survie_sexe = df.groupby("Sex")["Survived"].mean() * 100
print(taux_survie_sexe)

#calculons le taux de survie par classe 
taux_survie_class = df.groupby("Pclass")["Survived"].mean() * 100
print(taux_survie_class)

#Visulation avec Matplotlib:
#creons l'histogramme de la distribution des ages et celui des prix des billets
plt.subplot(1, 2, 1)
plt.hist(df["Age"], color="red", bins=30, label="Age")
plt.title("Distribution des ages")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(df["Fare"], color="blue", bins=30, label="Prix billets")
plt.title("Distribution des prix des billets")
plt.legend()
plt.grid(True)
plt.show()

#creons le graphe en barre du nombre des passagers par classe
compte = df["Pclass"].value_counts() 
plt.bar(compte.index, compte.values)
plt.show()

#creons le graphe de taux de survie par classe 
taux_survie_classe = df.groupby("Pclass")["Survived"].mean()
plt.bar(taux_survie_classe.index, taux_survie_classe.values)
plt.show()

#ccreons le graphue de taux de survie par sexe
taux_survie_sex = df.groupby("Sex")["Survived"].mean()
plt.bar(taux_survie_sex.index, taux_survie_sex.values)
plt.show()