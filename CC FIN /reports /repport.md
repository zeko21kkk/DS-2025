# Dataset Adult (Census Income) - 10.24432/C5XW20

## Créateurs et Date

- **Créateurs** : Barry Becker (Silicon Graphics) et Ronny Kohavi (Consultant/Stanford)
- **Date de création** : 1996
- **Date de donation au UCI ML Repository** : 30 avril 1996

## Origine des Données

- **Source** : Base de données du recensement américain de 1994
- **Enquête spécifique** : Current Population Survey (CPS) Annual Social and Economic Supplement (ASEC) de 1994
- **Extraction** : Réalisée par Barry Becker

### Critères de sélection des enregistrements
Les données ont été extraites avec les conditions suivantes :
- `(AAGE > 16)` : Âge supérieur à 16 ans
- `(AGI > 100)` : Revenu ajusté supérieur à 100$
- `(AFNLWGT > 1)` : Poids final supérieur à 1
- `(HRSWK > 0)` : Au moins 1 heure de travail par semaine

## Thématique et Objectif

**Objectif principal** : Prédire si le revenu annuel d'un individu dépasse 50 000 dollars par an, basé sur des données de recensement.

**Nom alternatif** : "Census Income dataset"

## Caractéristiques Techniques

| Caractéristique | Valeur |
|----------------|--------|
| Nombre d'instances | 48 842 |
| Nombre de features | 14 |
| Variable cible | 1 (>50K ou <=50K) |
| Types de données | Catégorielles et Entiers |
| Valeurs manquantes | Oui |
| Type de tâche | Classification |
| Domaine | Sciences Sociales |

## Variables du Dataset

### 1. age (Âge)
- **Type** : Integer (continu)
- **Valeurs manquantes** : Non

### 2. workclass (Classe de travail)
- **Type** : Catégoriel
- **Valeurs** : Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
- **Valeurs manquantes** : Oui

### 3. fnlwgt (Poids final)
- **Type** : Integer (continu)
- **Valeurs manquantes** : Non

### 4. education (Éducation)
- **Type** : Catégoriel
- **Valeurs** : Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool
- **Valeurs manquantes** : Non

### 5. education-num (Nombre d'années d'éducation)
- **Type** : Integer (continu)
- **Valeurs manquantes** : Non

### 6. marital-status (Statut matrimonial)
- **Type** : Catégoriel
- **Valeurs** : Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
- **Valeurs manquantes** : Non

### 7. occupation (Profession)
- **Type** : Catégoriel
- **Valeurs** : Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces
- **Valeurs manquantes** : Oui

### 8. relationship (Relations familiales)
- **Type** : Catégoriel
- **Valeurs** : Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
- **Valeurs manquantes** : Non

### 9. race (Race)
- **Type** : Catégoriel
- **Valeurs** : White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
- **Valeurs manquantes** : Non

### 10. sex (Sexe)
- **Type** : Binaire
- **Valeurs** : Female, Male
- **Valeurs manquantes** : Non

### 11. capital-gain (Gains en capital)
- **Type** : Integer (continu)
- **Valeurs manquantes** : Non

### 12. capital-loss (Pertes en capital)
- **Type** : Integer (continu)
- **Valeurs manquantes** : Non

### 13. hours-per-week (Heures par semaine)
- **Type** : Integer (continu)
- **Valeurs manquantes** : Non

### 14. native-country (Pays d'origine)
- **Type** : Catégoriel
- **Valeurs** : United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands
- **Valeurs manquantes** : Non

### Variable Cible
- **Nom** : income
- **Valeurs** : >50K, <=50K

## Fichiers Disponibles

| Fichier | Taille |
|---------|--------|
| adult.data | 3.8 MB |
| adult.test | 1.9 MB |
| adult.names | 5.1 KB |
| old.adult.names | 4.2 KB |
| Index | 140 Bytes |

## Utilisation et Impact

- **Articles de recherche** : Plus de 300 articles utilisant ce dataset
- **Domaine principal** : Équité algorithmique (algorithmic fairness)
- **Premier article sur l'équité** : 2009, par Calders et al.
- **Statut** : Un des datasets de référence les plus utilisés pour le développement et la comparaison d'interventions en matière d'équité algorithmique

## Licence

**Type** : Creative Commons Attribution 4.0 International (CC BY 4.0)

**Permissions** :
- Partage autorisé
- Adaptation autorisée
- Usage commercial autorisé
- Attribution requise

## Citation
```
Becker, B. & Kohavi, R. (1996). Adult [Dataset]. 
UCI Machine Learning Repository. 
https://doi.org/10.24432/C5XW20
```

## Accès au Dataset

### Via Python (ucimlrepo)
```python
pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

# Télécharger le dataset
adult = fetch_ucirepo(id=2)

# Données (pandas dataframes)
X = adult.data.features
y = adult.data.targets

# Métadonnées
print(adult.metadata)

# Informations sur les variables
print(adult.variables)
```

### Téléchargement direct
- Lien : https://archive.ics.uci.edu/dataset/2/adult
- Taille totale : 605.7 KB (zippé)

## Contact

**Ronny Kohavi** : ronnyk@live.com
