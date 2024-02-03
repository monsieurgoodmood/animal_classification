import os
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(data_dir, classes, model):
    y_true = []
    y_pred = []

    # Parcourir chaque classe pour lire les images et faire des prédictions
    for class_id, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            results = model(img_path)

            # Adapter la récupération des prédictions selon la structure de l'objet results
            # Vérifier d'abord si results contient des attributs attendus
            if hasattr(results, 'pred') and len(results.pred) > 0 and len(results.pred[0]) > 0:
                # Accéder à la première prédiction la plus confiante
                best_pred = results.pred[0][0]
                pred_class = int(best_pred[-1])
            elif hasattr(results, 'xyxy') and len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                # Ou utiliser xyxy si pred n'est pas disponible
                best_pred = results.xyxy[0][0]
                pred_class = int(best_pred[-1])
            else:
                # Si aucune prédiction, utiliser -1 ou une valeur spéciale pour indiquer l'absence de prédiction
                pred_class = -1

            y_true.append(class_id)
            y_pred.append(pred_class)

    # Filtrer les prédictions manquantes si nécessaire
    y_true_filtered, y_pred_filtered = zip(*[(t, p) for t, p in zip(y_true, y_pred) if p != -1])

    return np.array(y_true_filtered), np.array(y_pred_filtered)

# Configuration initiale
val_dir = '/content/animal_classification/val'
classes = ['bird', 'cat', 'dog']
model_path = '/content/animal_classification/runs/classify/train/weights/best.pt'  # Mettez à jour avec le chemin de votre modèle
model = YOLO(model_path)

# Évaluation
y_true_val, y_pred_val = evaluate_model(val_dir, classes, model)
accuracy_val = accuracy_score(y_true_val, y_pred_val)
precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted')

# Affichage des résultats
print(f'Validation Accuracy: {accuracy_val:.2f}')
print(f'Precision: {precision_val:.2f}, Recall: {recall_val:.2f}, F-Score: {fscore_val:.2f}')
