from ultralytics import YOLO
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from pathlib import Path

# Charger le modèle
model = YOLO('/content/drive/MyDrive/ColabNotebooks/best.pt')  # Remplacez par le chemin réel de votre fichier best.pt

# Charger les données de validation
data_yaml = '/content/animal_classification/animal_data.yaml'  # Remplacez par le chemin réel vers votre fichier YAML de données
with open(data_yaml) as f:
    data = data_yaml.safe_load(f)  # Charger le fichier YAML
val_dir = data['val']  # Récupérer le chemin vers les données de validation

# Initialiser les listes pour les étiquettes réelles et prédites
y_true = []
y_pred = []

# Charger les images de validation
val_images = list(Path(val_dir).rglob('*.jpg')) + list(Path(val_dir).rglob('*.png'))  # Ajouter d'autres formats si nécessaire

# Prédire les étiquettes pour les images de validation
for img_path in val_images:
    # Extraire les étiquettes réelles à partir des noms de fichiers ou d'un fichier d'annotation
    # Ceci est un exemple et doit être adapté à la façon dont vos données sont structurées
    label = img_path.parts[-2]  # Adaptez ceci pour extraire l'étiquette réelle
    true_label = data['names'].index(label)  # Convertir le nom de la classe en index
    y_true.append(true_label)

    # Effectuer la prédiction
    results = model(img_path)
    pred = results.pred[0]
    if pred.shape[0]:
        pred_classes = pred[:, -1].numpy()  # Récupérer les classes prédites
        y_pred.append(int(pred_classes[0]))  # Prendre la classe avec la plus haute confiance
    else:
        y_pred.append(None)  # Aucune prédiction pour cette image

# Supprimer les cas où le modèle n'a pas fait de prédiction
valid_indices = [i for i, x in enumerate(y_pred) if x is not None]
y_true = [y_true[i] for i in valid_indices]
y_pred = [y_pred[i] for i in valid_indices]

# Calculer les métriques
precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)
conf_mat = confusion_matrix(y_true, y_pred)

# Afficher les résultats
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {fscore:.2f}')
print('Confusion Matrix:')
print(conf_mat)
