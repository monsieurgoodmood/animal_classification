import os
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Chemin vers le dossier contenant les images de validation classées par dossiers de classe
val_dir = '/content/animal_classification/val'
# Liste des classes dans l'ordre utilisé lors de l'entraînement
classes = ['bird', 'cat', 'dog']

# Charger le modèle YOLOv8 pré-entraîné ou personnalisé
model_path = '/content/animal_classification/runs/classify/train/weights/best.pt'  # Chemin vers votre modèle entraîné
model = YOLO(model_path)

# Fonction pour évaluer le modèle sur l'ensemble de validation
def evaluate_model(data_dir, classes, model):
    y_true, y_pred = [], []
    for i, cls in enumerate(classes):  # Parcourir chaque classe
        class_dir = os.path.join(data_dir, cls)
        for img_file in os.listdir(class_dir):  # Parcourir chaque image
            img_path = os.path.join(class_dir, img_file)
            results = model(img_path)  # Prédire avec le modèle

            # S'assurer qu'il y a une prédiction et extraire les probabilités
            if results and hasattr(results[0], 'probs') and results[0].probs is not None:
                pred_probs = results[0].probs
                pred_class_id = np.argmax(pred_probs)  # Obtenir l'ID de classe avec la probabilité la plus élevée
                y_true.append(i)
                y_pred.append(pred_class_id)
            else:
                # Gérer les cas où aucune prédiction de classe n'est faite
                y_pred.append(-1)  # Utiliser -1 pour les prédictions manquantes
                y_true.append(i)

    return np.array(y_true), np.array(y_pred)

# Évaluer le modèle et calculer les métriques
y_true_val, y_pred_val = evaluate_model(val_dir, classes, model)
# Filtrer les prédictions manquantes
valid_indices = y_pred_val != -1
y_true_val = y_true_val[valid_indices]
y_pred_val = y_pred_val[valid_indices]

# Calculer les métriques si des prédictions valides ont été faites
if len(y_true_val) > 0 and len(y_pred_val) > 0:
    accuracy_val = accuracy_score(y_true_val, y_pred_val)
    precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted')
    print(f"Accuracy: {accuracy_val:.2f}, Precision: {precision_val:.2f}, Recall: {recall_val:.2f}, F-score: {fscore_val:.2f}")
else:
    print("Aucune prédiction valide n'a été faite.")
