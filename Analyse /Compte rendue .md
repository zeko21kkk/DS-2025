    Titanic - Machine Learning from Disaster

Titanic - Machine Learning from Disaster
Rapport d'Analyse Approfondie du Dataset Titanic
📋 Table des Matières

Vue d'ensemble du dataset
Description des variables
Analyse exploratoire des données (EDA)
Nettoyage et prétraitement
Analyse statistique
Visualisations recommandées
Insights et découvertes
Modélisation et prédiction
Conclusions et recommandations
[titanic_analysis_report.md]  Rapport d'Analyse Approfondie du Dataset Titanic

## 📋 Table des Matières

1. [Vue d'ensemble du dataset](#vue-densemble-du-dataset)
2. [Description des variables](#description-des-variables)
3. [Analyse exploratoire des données (EDA)](#analyse-exploratoire-des-données-eda)
4. [Nettoyage et prétraitement](#nettoyage-et-prétraitement)
5. [Analyse statistique](#analyse-statistique)
6. [Visualisations recommandées](#visualisations-recommandées)
7. [Insights et découvertes](#insights-et-découvertes)
8. [Modélisation et prédiction](#modélisation-et-prédiction)
9. [Conclusions et recommandations](#conclusions-et-recommandations)

---

## 📊 Vue d'ensemble du dataset

Le dataset Titanic est une compétition classique de Kaggle qui vise à prédire la survie des passagers du Titanic lors de son naufrage en 1912. Ce dataset est idéal pour l'apprentissage du machine learning et de l'analyse de données.

### Fichiers disponibles

- **train.csv** : Données d'entraînement (891 passagers)
- **test.csv** : Données de test (418 passagers)
- **gender_submission.csv** : Exemple de soumission basé sur le genre

### Objectif

Prédire si un passager a survécu (Survived = 1) ou non (Survived = 0) en fonction de ses caractéristiques.

---

## 🔍 Description des variables

| Variable | Type | Description | Valeurs |
|----------|------|-------------|---------|
| **PassengerId** | int | Identifiant unique du passager | 1-891 (train) |
| **Survived** | int | Variable cible : survie | 0 = Non, 1 = Oui |
| **Pclass** | int | Classe du billet | 1 = 1ère, 2 = 2ème, 3 = 3ème |
| **Name** | str | Nom complet du passager | Nom, Prénom (titre) |
| **Sex** | str | Sexe du passager | male, female |
| **Age** | float | Âge en années | 0.42-80 |
| **SibSp** | int | Nombre de frères/sœurs/conjoints à bord | 0-8 |
| **Parch** | int | Nombre de parents/enfants à bord | 0-6 |
| **Ticket** | str | Numéro de billet | Alphanumerique |
| **Fare** | float | Prix du billet | 0-512.33 |
| **Cabin** | str | Numéro de cabine | A-G + numéro |
| **Embarked** | str | Port d'embarquement | C=Cherbourg, Q=Queenstown, S=Southampton |

---

## 🔬 Analyse exploratoire des données (EDA)

### 1. Chargement des données

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Aperçu des données
print(train.head())
print(train.info())
print(train.describe())
```

### 2. Vérification des valeurs manquantes

```python
# Valeurs manquantes
missing = train.isnull().sum()
missing_percent = (missing / len(train)) * 100
missing_df = pd.DataFrame({
    'Colonne': missing.index,
    'Valeurs manquantes': missing.values,
    'Pourcentage': missing_percent.values
})
print(missing_df[missing_df['Valeurs manquantes'] > 0])
```

**Résultats attendus :**
- Age : ~20% de valeurs manquantes
- Cabin : ~77% de valeurs manquantes
- Embarked : 2 valeurs manquantes

### 3. Distribution des variables

```python
# Distribution de la variable cible
print(train['Survived'].value_counts())
print(f"Taux de survie : {train['Survived'].mean():.2%}")

# Distribution par classe
print(train['Pclass'].value_counts().sort_index())

# Distribution par sexe
print(train['Sex'].value_counts())

# Distribution par port d'embarquement
print(train['Embarked'].value_counts())
```

---

## 🧹 Nettoyage et prétraitement

### 1. Gestion des valeurs manquantes

```python
# Age : imputation par la médiane selon Pclass et Sex
train['Age'].fillna(train.groupby(['Pclass', 'Sex'])['Age'].transform('median'), inplace=True)

# Embarked : imputation par le mode
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# Cabin : création d'une variable binaire
train['Has_Cabin'] = train['Cabin'].notna().astype(int)

# Fare : imputation par la médiane (pour test set)
test['Fare'].fillna(test['Fare'].median(), inplace=True)
```

### 2. Feature Engineering

```python
# Extraction du titre depuis Name
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Simplification des titres
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 
    'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
    'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
    'Capt': 'Rare', 'Sir': 'Rare'
}
train['Title'] = train['Title'].map(title_mapping)

# Taille de la famille
train['Family_Size'] = train['SibSp'] + train['Parch'] + 1

# Catégorie de famille
train['Family_Category'] = pd.cut(train['Family_Size'], 
                                    bins=[0, 1, 4, 11], 
                                    labels=['Alone', 'Small', 'Large'])

# Catégorie d'âge
train['Age_Category'] = pd.cut(train['Age'], 
                                bins=[0, 12, 18, 35, 60, 100], 
                                labels=['Child', 'Teen', 'Adult', 'Middle_Age', 'Senior'])

# Est enfant
train['Is_Child'] = (train['Age'] < 18).astype(int)

# Fare par personne
train['Fare_Per_Person'] = train['Fare'] / train['Family_Size']
```

### 3. Encodage des variables catégorielles

```python
# Label Encoding pour Sex
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Sex_encoded'] = le.fit_transform(train['Sex'])

# One-Hot Encoding pour Embarked
train = pd.get_dummies(train, columns=['Embarked'], prefix='Emb', drop_first=True)

# One-Hot Encoding pour Pclass
train = pd.get_dummies(train, columns=['Pclass'], prefix='Class', drop_first=True)
```

---

## 📈 Analyse statistique

### 1. Taux de survie par caractéristiques

```python
# Survie par sexe
survival_by_sex = train.groupby('Sex')['Survived'].agg(['mean', 'count'])
print("Taux de survie par sexe :")
print(survival_by_sex)

# Survie par classe
survival_by_class = train.groupby('Pclass')['Survived'].agg(['mean', 'count'])
print("\nTaux de survie par classe :")
print(survival_by_class)

# Survie par port d'embarquement
survival_by_embarked = train.groupby('Embarked')['Survived'].agg(['mean', 'count'])
print("\nTaux de survie par port d'embarquement :")
print(survival_by_embarked)

# Survie par catégorie d'âge
survival_by_age = train.groupby('Age_Category')['Survived'].mean()
print("\nTaux de survie par catégorie d'âge :")
print(survival_by_age)

# Survie par taille de famille
survival_by_family = train.groupby('Family_Category')['Survived'].mean()
print("\nTaux de survie par taille de famille :")
print(survival_by_family)
```

### 2. Corrélations

```python
# Matrice de corrélation
numeric_features = ['Survived', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 
                    'Fare', 'Family_Size', 'Is_Child', 'Has_Cabin']
correlation_matrix = train[numeric_features].corr()

print("Corrélations avec Survived :")
print(correlation_matrix['Survived'].sort_values(ascending=False))
```

### 3. Tests statistiques

```python
from scipy.stats import chi2_contingency, ttest_ind

# Test Chi² pour Sex vs Survived
contingency_table = pd.crosstab(train['Sex'], train['Survived'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Test Chi² (Sex vs Survived) : p-value = {p_value:.4f}")

# Test t pour Age (Survived vs Not Survived)
age_survived = train[train['Survived'] == 1]['Age'].dropna()
age_not_survived = train[train['Survived'] == 0]['Age'].dropna()
t_stat, p_value = ttest_ind(age_survived, age_not_survived)
print(f"Test t (Age) : p-value = {p_value:.4f}")
```

---

## 📊 Visualisations recommandées

### 1. Distribution de la survie

```python
plt.figure(figsize=(12, 4))

# Graphique 1 : Répartition de la survie
plt.subplot(1, 3, 1)
train['Survived'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Répartition de la survie')
plt.xlabel('Survived (0=Non, 1=Oui)')
plt.ylabel('Nombre de passagers')
plt.xticks(rotation=0)

# Graphique 2 : Taux de survie
plt.subplot(1, 3, 2)
survival_rate = train['Survived'].value_counts(normalize=True)
plt.pie(survival_rate, labels=['Décédé', 'Survécu'], autopct='%1.1f%%', colors=['red', 'green'])
plt.title('Taux de survie global')

# Graphique 3 : Survie par sexe
plt.subplot(1, 3, 3)
pd.crosstab(train['Sex'], train['Survived']).plot(kind='bar', color=['red', 'green'])
plt.title('Survie par sexe')
plt.xlabel('Sexe')
plt.ylabel('Nombre de passagers')
plt.legend(['Décédé', 'Survécu'])
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()
```

### 2. Analyse multivariée

```python
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Survie par classe et sexe
pd.crosstab([train['Pclass'], train['Sex']], train['Survived']).plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('Survie par classe et sexe')
axes[0, 0].set_xlabel('Classe et Sexe')
axes[0, 0].legend(['Décédé', 'Survécu'])

# Distribution de l'âge par survie
axes[0, 1].hist([train[train['Survived']==0]['Age'].dropna(), 
                 train[train['Survived']==1]['Age'].dropna()], 
                bins=20, label=['Décédé', 'Survécu'], color=['red', 'green'])
axes[0, 1].set_title('Distribution de l\'âge par survie')
axes[0, 1].set_xlabel('Âge')
axes[0, 1].legend()

# Survie par taille de famille
survival_by_family_size = train.groupby('Family_Size')['Survived'].mean()
survival_by_family_size.plot(kind='bar', ax=axes[1, 0], color='blue')
axes[1, 0].set_title('Taux de survie par taille de famille')
axes[1, 0].set_xlabel('Taille de la famille')
axes[1, 0].set_ylabel('Taux de survie')
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)

# Boxplot du tarif par classe et survie
train.boxplot(column='Fare', by=['Pclass', 'Survived'], ax=axes[1, 1])
axes[1, 1].set_title('Distribution du tarif par classe et survie')
axes[1, 1].set_xlabel('Classe et Survie')
axes[1, 1].set_ylabel('Tarif')

plt.tight_layout()
plt.show()
```

### 3. Heatmap de corrélation

```python
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matrice de corrélation des variables numériques')
plt.tight_layout()
plt.show()
```

---

## 💡 Insights et découvertes

### Découvertes principales

1. **Le sexe est le facteur le plus déterminant**
   - Taux de survie des femmes : ~74%
   - Taux de survie des hommes : ~19%
   - Principe "les femmes et les enfants d'abord"

2. **La classe sociale a un impact significatif**
   - 1ère classe : ~63% de survie
   - 2ème classe : ~47% de survie
   - 3ème classe : ~24% de survie
   - Les passagers de 1ère classe avaient un meilleur accès aux canots de sauvetage

3. **L'âge influence la survie**
   - Les enfants (< 18 ans) avaient un taux de survie plus élevé
   - Les enfants de moins de 10 ans : ~61% de survie
   - Impact du principe "les femmes et les enfants d'abord"

4. **La taille de la famille est importante**
   - Les familles de 2-4 personnes avaient le meilleur taux de survie
   - Les personnes seules : ~30% de survie
   - Les grandes familles (5+) : ~16% de survie
   - Explication : entraide familiale vs difficulté à coordonner de grands groupes

5. **Le port d'embarquement révèle des patterns**
   - Cherbourg (C) : ~55% de survie (plus de passagers de 1ère classe)
   - Queenstown (Q) : ~39% de survie
   - Southampton (S) : ~34% de survie

6. **Le tarif corrèle avec la survie**
   - Les passagers ayant payé plus cher avaient plus de chances de survivre
   - Corrélation avec la classe et l'accès aux canots de sauvetage

### Patterns intéressants

- Les femmes de 1ère classe avaient un taux de survie de ~97%
- Les hommes de 3ème classe avaient un taux de survie de ~14%
- Avoir une cabine (information disponible) est un indicateur positif
- Le titre (Mr, Mrs, Miss, Master) encode des informations sur l'âge et le statut marital

---

## 🤖 Modélisation et prédiction

### 1. Préparation des données

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Sélection des features
features = ['Sex_encoded', 'Age', 'Fare', 'Family_Size', 'Is_Child', 
            'Has_Cabin', 'Emb_Q', 'Emb_S', 'Class_2', 'Class_3']

X = train[features]
y = train['Survived']

# Division train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

### 2. Modèles à tester

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dictionnaire de modèles
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier()
}

# Entraînement et évaluation
results = {}
for name, model in models.items():
    # Entraînement
    model.fit(X_train_scaled, y_train)
    
    # Prédiction
    y_pred = model.predict(X_val_scaled)
    
    # Évaluation
    accuracy = accuracy_score(y_val, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    results[name] = {
        'Accuracy': accuracy,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    }
    
    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
```

### 3. Optimisation des hyperparamètres

```python
from sklearn.model_selection import GridSearchCV

# Exemple avec Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("Meilleurs paramètres:", grid_search.best_params_)
print("Meilleur score:", grid_search.best_score_)

# Prédiction avec le meilleur modèle
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val_scaled)
print("Accuracy sur validation:", accuracy_score(y_val, y_pred))
```

### 4. Feature Importance

```python
# Pour Random Forest
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Importance des features (Random Forest)')
plt.tight_layout()
plt.show()

print("\nImportance des features:")
print(feature_importance)
```

### 5. Prédiction sur le test set

```python
# Préparation du test set (appliquer les mêmes transformations)
# ... [même preprocessing que train]

# Prédiction
test_predictions = best_model.predict(test[features])

# Création du fichier de soumission
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("Fichier de soumission créé avec succès!")
```

---

## 🎯 Conclusions et recommandations

### Résumé des résultats

1. **Taux de survie global** : ~38% des passagers ont survécu
2. **Facteurs les plus importants** : Sexe, Classe, Âge, Taille de famille
3. **Meilleur modèle** : Random Forest ou Gradient Boosting (~80-82% d'accuracy)
4. **Score Kaggle attendu** : Entre 0.75 et 0.82

### Recommandations pour améliorer le modèle

1. **Feature Engineering avancé**
   - Interactions entre variables (ex: Sex × Pclass)
   - Extraction d'informations du numéro de cabine (pont)
   - Analyse des noms pour identifier les familles
   - Regroupement de billets (passagers voyageant ensemble)

2. **Gestion des valeurs manquantes**
   - Utiliser des algorithmes de prédiction pour l'âge (KNN, Random Forest)
   - Exploiter les corrélations entre variables

3. **Ensemble methods**
   - Stacking de plusieurs modèles
   - Voting Classifier
   - Blending

4. **Hyperparameter tuning**
   - Utiliser RandomizedSearchCV pour explorer plus de combinaisons
   - Optimisation bayésienne (Optuna, Hyperopt)

5. **Cross-validation stratifiée**
   - Assurer une distribution équilibrée des classes
   - K-Fold avec stratification

### Limitations

- Dataset relativement petit (891 observations)
- Beaucoup de valeurs manquantes pour Cabin
- Possibilité d'overfitting avec des modèles complexes
- Certaines informations historiques ne sont pas dans le dataset

### Prochaines étapes

1. Implémenter le feature engineering avancé
2. Tester des modèles d'ensemble
3. Effectuer une validation croisée rigoureuse
4. Soumettre les prédictions à Kaggle
5. Analyser les erreurs de prédiction pour améliorer le modèle

---

## 📚 Ressources supplémentaires

### Code complet disponible

Le code complet avec toutes les analyses est disponible sur GitHub :
```bash
git clone https://github.com/your-username/titanic-analysis.git
```

### Notebooks Kaggle recommandés

- [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
- [A Data Science Framework: To Achieve 99% Accuracy](https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy)
- [Titanic: Advanced Feature Engineering Tutorial](https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial)

### Bibliothèques utilisées

```python
pandas==2.0.0
numpy==1.24.0
matplotlib==3.7.0
seaborn==0.12.0
scikit-learn==1.2.0
scipy==1.10.0
```

---

**Auteur** : Votre Nom  
**Date** : 2025-11-27  
**Version** : 1.0  
**Licence** : MIT

---

*Ce rapport a été généré pour l'analyse du dataset Titanic de Kaggle. Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.*


📊 Vue d'ensemble du dataset
Le dataset Titanic est une compétition classique de Kaggle qui vise à prédire la survie des passagers du Titanic lors de son naufrage en 1912. Ce dataset est idéal pour l'apprentissage du machine learning et de l'analyse de données.
Fichiers disponibles

train.csv : Données d'entraînement (891 passagers)
test.csv : Données de test (418 passagers)
gender_submission.csv : Exemple de soumission basé sur le genre

Objectif
Prédire si un passager a survécu (Survived = 1) ou non (Survived = 0) en fonction de ses caractéristiques.

