from ultralytics import YOLO
import os
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Charger le modèle pré-entraîné
model = YOLO('/content/animal_classification/best.pt')  # Mettez le chemin de votre fichier best.pt

# Définir le chemin du dossier de validation et les classes
val_dir = '/content/animal_classification/val'
classes = ['bird', 'cat', 'dog']

# Initialiser les listes pour les vraies étiquettes et les prédictions
y_true = []
y_pred = []

# Parcourir les images de validation et effectuer des prédictions
for i, cls in enumerate(classes):
    class_dir = os.path.join(val_dir, cls)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)

        # Effectuer la prédiction
        results = model(img_path)
        # Récupérer les prédictions
        predictions = results.pred[0] if isinstance(results.pred, list) else results.pred
        confidence, class_pred = torch.max(predictions, dim=1)

        # Ajouter la vraie classe et la classe prédite aux listes
        y_true.append(i)
        y_pred.append(class_pred.item())

# Calculer les métriques globales
accuracy = accuracy_score(y_true, y_pred)
precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

# Afficher les métriques
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {fscore:.2f}')

# Afficher la matrice de confusion
conf_mat = confusion_matrix(y_true, y_pred, target_names=classes)
print('Confusion Matrix:')
print(conf_mat)

# Pour afficher la matrice de confusion plus visuellement
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
