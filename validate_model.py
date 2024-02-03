import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ultralytics import YOLO  # Assurez-vous d'avoir installé le package ultralytics

# Définir les chemins vers les répertoires de données et le modèle
val_dir = '/content/animal_classification/val'  # Chemin vers le répertoire de validation
classes = ['bird', 'cat', 'dog']  # Classes
model_path = '/content/animal_classification/runs/classify/train/weights/best.pt'  # Chemin vers le modèle YOLOv8 pré-entraîné

# Charger le modèle YOLOv8
model = YOLO(model_path)

def evaluate_model(data_dir, classes, model):
    y_true = []
    y_pred = []
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            results = model(img_path)

            # Correction: Vérifier si results contient des prédictions et traiter correctement
            if len(results) > 0 and hasattr(results[0], 'pred') and len(results[0].pred) > 0:
                pred = results[0].pred[0]  # Récupérer les prédictions
                # Sélectionner la prédiction avec la plus haute confiance
                if len(pred) > 0:
                    best_pred = pred[pred[:, 4].argmax()]
                    pred_class = classes[int(best_pred[-1])]
                    y_true.append(class_name)
                    y_pred.append(pred_class)
                else:
                    # Si aucune prédiction n'est faite pour l'image, on peut choisir de l'ignorer ou d'assigner une classe par défaut
                    pass
            else:
                # Gérer le cas où aucune prédiction n'est retournée
                pass

    return np.array(y_true), np.array(y_pred)

# Évaluer le modèle sur l'ensemble de validation
y_true_val, y_pred_val = evaluate_model(val_dir, classes, model)
if len(y_true_val) > 0 and len(y_pred_val) > 0:
    accuracy_val = accuracy_score(y_true_val, y_pred_val)
    precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted')
    print(f"Validation Accuracy: {accuracy_val}")
    print(f"Precision: {precision_val}, Recall: {recall_val}, F-score: {fscore_val}")
else:
    print("Aucune prédiction valide n'a été faite.")
