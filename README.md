![Page de couverture du projet](images/couverture.png)

#  Ames Housing Price Prediction - ML Fundamentals Checkpoint

Ce projet est un checkpoint d'évaluation axé sur les **fondations du Machine Learning**. L'objectif principal est de construire, d'entraîner et d'évaluer un modèle de régression régularisée (**Ridge Regression**) afin de prédire le prix de vente des maisons (`SalePrice`) en utilisant le jeu de données *Ames Housing*.

---

##  Objectifs Pédagogiques & Techniques
Le projet couvre les étapes fondamentales du flux de travail en science des données [1] :
- **Séparation des données** : Validation croisée via un découpage entraînement/test (*train-test split*) pour évaluer la capacité de généralisation.
- **Prétraitement des données** : Standardisation des caractéristiques numériques à l'aide de `StandardScaler` pour éviter les biais liés aux échelles.
- **Régularisation (L2)** : Application de la régression Ridge pour limiter le surapprentissage (*overfitting*).
- **Évaluation** : Analyse de la performance du modèle à l'aide des métriques **R² (Coefficient de détermination)** et **RMSE (Root Mean Squared Error)**.

---

##  Stack Technique
Le projet est développé en **Python 3** au sein d'un environnement Jupyter Notebook (`.ipynb`) [1] et s'appuie sur les bibliothèques standards suivantes :
* **Pandas & NumPy** : Pour l'exploration, le nettoyage et la manipulation des données.
* **Scikit-Learn (sklearn)** : 
  * `model_selection.train_test_split` [1]
  * `preprocessing.StandardScaler` [1]
  * `linear_model.Ridge` [1]
  * `metrics` (`mean_squared_error`, `r2_score`) [1]

---

##  Structure et Pipeline du Code

Le notebook est structuré en 5 étapes clés guidées par des tests de validation intégrés (`assert`) [1] :

### 1. Chargement et Nettoyage Initial
Seules les données numériques sans valeurs manquantes sont conservées pour l'exercice. Le dataset est ensuite séparé en variables explicatives (`X`) et variable cible (`y = SalePrice`), suivi d'un découpage à **40% pour le jeu de test** (Random State: 42).

### 2. Standardisation (Scaling)
Instanciation d'un `StandardScaler` ajusté uniquement sur le jeu d'entraînement (`X_train`) afin d'éviter les fuites de données (*data leakage*), puis appliqué sur `X_train` et `X_test`.

### 3. Entraînement du Modèle Ridge
Configuration et ajustement du modèle avec les hyperparamètres spécifiques [1] :
* `alpha = 100` (Force de la pénalité de régularisation) [1]
* `solver = "sag"` (Stochastic Average Gradient Descent) [1]
* `random_state = 1` [1]

### 4. Évaluation des Performances
Génération des prédictions et calcul des scores d'erreurs (RMSE et R²) sur les deux ensembles de données pour mesurer précisément la précision et détecter un éventuel surapprentissage.

### 5. Interprétation & Comparaison
Analyse comparative finale entre un modèle classique de Régression Linéaire et la Régression Ridge pour déterminer la meilleure approche en contexte prédictif [1] :

| Modèle | RMSE Entraînement | RMSE Test |
| :--- | :---: | :---: |
| **Linear Regression** | \$33,633.14 | \$39,255.80 |
| **Ridge Regression** | \$33,910.84 | \$39,213.66 |

---
