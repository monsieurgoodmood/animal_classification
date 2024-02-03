import os
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(data_dir, classes, model):
    y_true = []
    y_pred = []
    for i, cls in enumerate(classes):  # Pour chaque classe
        class_dir = os.path.join(data_dir, cls)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            results = model(img_path)

            # Vérifier si des détections ont été faites
            detected = False
            if results.pred is not None:
                for pred in results.pred:
                    if len(pred) > 0:
                        detected = True
                        # Trouver la prédiction avec la plus haute confiance
                        best_pred = pred[0]  # Prendre la première prédiction (la plus confiante)
                        pred_class = int(best_pred[-1])
                        y_pred.append(pred_class)
                        break
            if not detected:
                y_pred.append(-1)  # -1 pour les prédictions manquantes/non détectées

            y_true.append(i)  # L'index de la classe actuelle

    # Filtrer les prédictions manquantes
    y_true_filtered, y_pred_filtered = zip(*[(t, p) for t, p in zip(y_true, y_pred) if p != -1])

    return np.array(y_true_filtered), np.array(y_pred_filtered)

# Charger le modèle
model_path = '/content/animal_classification/runs/classify/train/weights/best.pt'  # Mettez à jour avec le chemin vers votre modèle YOLOv8
model = YOLO(model_path)

val_dir = '/content/animal_classification/val'  # Chemin vers le dossier de validation
classes = ['bird', 'cat', 'dog']  # Mettez à jour avec vos classes

# Évaluation sur l'ensemble de validation
y_true_val, y_pred_val = evaluate_model(val_dir, classes, model)
if len(y_true_val) > 0 and len(y_pred_val) > 0:
    accuracy_val = accuracy_score(y_true_val, y_pred_val)
    precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted')
    print(f"Validation Accuracy: {accuracy_val:.4f}")
    print(f"Validation Precision: {precision_val:.4f}")
    print(f"Validation Recall: {recall_val:.4f}")
    print(f"Validation F1-Score: {fscore_val:.4f}")
else:
    print("Aucune prédiction valide n'a été effectuée.")
