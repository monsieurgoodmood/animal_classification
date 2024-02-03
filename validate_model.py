from ultralytics import YOLO
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Charger le modèle pré-entraîné
model = YOLO('/content/animal_classification/runs/classify/train/weights/best.pt')  # Assurez-vous du chemin

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

        # Récupérer la classe prédite (indice avec la plus haute confiance)
        pred_index, _ = max(enumerate(results), key=lambda x: x[1][1])  # Prendre l'indice avec le max de confiance
        y_pred.append(pred_index)

        # Ajouter la vraie classe
        y_true.append(i)

# Calculer les métriques globales
accuracy = accuracy_score(y_true, y_pred)
precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

# Afficher les métriques
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {fscore:.2f}')

# Afficher la matrice de confusion
conf_mat = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(conf_mat)

# Pour afficher la matrice de confusion plus visuellement
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
