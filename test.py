import pandas as pd
df = pd.read_csv("clients.csv") 
print(df) #Affiche le fichier csv 
print(df.info()); #donne les informations sur les colonnes 
print(df.describe()); #donne une statistiques d'ensemble 
print(df.head(3)); #affiche les 3 premieres lignes 
print(df.tail(3)); #affiche les 3 dernières lignes 
# df.mean(); #donne la moyenne 
# df.max(): donne la valeur maximale 
# df.min(): donne la valeur minimale 
# df.count(): affiche le nombre des valeurs non nulles 
print(df["Index"]) #selectionne une colonne particulière  
print(df[["First Name", "Last Name", "City", "Phone 1", "Email"]]) #selectionne plusieures colonnes particulieres 
print(df.drop(columns=["Index"])) #supprime une colonne particuliere 
print(df.isnull()) #Affiche les valeurs manquantes 
print(df.dropna()) #supprime les valeurs manquantes 
print(df.fillna("---")) #Remplace toutes les valeurs manquantes par --- 
print(df.replace(1, "one")) #Remplace les valeurs par des valeurs quelconques 
print(df[df["Index"] >= 50]) #faire des selections sur de colonnes 
#print(df[(df["Index"] >= 50) & (df["Customer Id"] >= 50)]) 