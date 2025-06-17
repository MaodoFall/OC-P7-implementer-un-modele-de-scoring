# 💳 Implémentation et déploiement d’un modèle de scoring crédit

**Projet 7 – Mastère Spécialisé Data Science – OpenClassrooms**  
**Client : Prêt à dépenser (institution financière)**  
**Rôle : Data Scientist / MLOps**

---

## 🎯 Objectif du projet

Développer et mettre en production un **modèle de scoring crédit** capable de prédire la probabilité de défaut d’un client sans historique de prêt, pour automatiser l’acceptation de dossiers de crédit.

---

## 🗂️ Données

📦 Source : [Kaggle – Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)  
- 307 511 clients  
- 121 variables (comportement, historique, finances, etc.)
- Données réparties sur **10 fichiers relationnels**  
- Variable cible : `TARGET` (0 = remboursé, 1 = défaut)

---

## 🧪 Étapes du projet

### 🔍 1. Analyse exploratoire
- Nettoyage, imputation, encodage
- Agrégation des tables secondaires
- Création de features par domaine d’expertise

### 🤖 2. Modélisation & MLflow
- Entraînement de plusieurs modèles : **LogReg, RandomForest, LGBM**
- Essais avec des approches de rééchantillonnage et d'optimisation par recherche d'hyperparamètres 
- Suivi des expériences avec **MLflow**
- Sélection du **LGBMClassifier** basé sur un **score métier optimisé**

### 🔬 3. Explicabilité du modèle
- Analyse **globale** des features avec SHAP
- Analyse **locale** pour chaque prédiction avec LIME

### ☁️ 4. Mise en production
- Création d’une **API FastAPI** encapsulant le modèle
- Conteneurisation via **Docker**
- Déploiement automatisé via **GitHub Actions** vers **AWS ECS**

### 🛡️ 5. Maintenance & suivi
- Implémentation d’un **notebook d’analyse du data drift**
- Recommandations pour le monitoring en production

---

## 🚀 Infrastructure & MLOps

| Élément         | Technologie utilisée                     |
|-----------------|-------------------------------------------|
| Déploiement API | FastAPI + Docker + AWS ECS               |
| CI/CD           | GitHub Actions (`.yml`)                  |
| Tracking        | MLflow + MLflow UI (expériences modèles) |
| Tests API       | `pytest` sur les endpoints REST          |
| Monitoring      | Analyse du **drift de données**          |


---

## 🌐 API en production

✅ Déployée sur AWS ECS 

---

## 🙋‍♂️ Réalisé par

**Maodo FALL**  
Data Scientist*  

---

