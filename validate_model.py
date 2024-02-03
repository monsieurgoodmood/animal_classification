from ultralytics import YOLO  # Assurez-vous d'avoir installé le package ultralytics
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(data_dir, classes, model):
    y_true = []
    y_pred = []

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            results = model(img_path)

            # Vérifier si des prédictions ont été faites
            if results.pred is not None and len(results.pred[0]) > 0:
                preds = results.pred[0]
                # Trouver la prédiction avec la plus haute confiance
                best_pred = preds[preds[:, 4].argmax()]
                predicted_class_id = best_pred[5].item()
                y_pred.append(int(predicted_class_id))
            else:
                # Aucune prédiction, on pourrait choisir de mettre une valeur spéciale ou d'ignorer cette image
                y_pred.append(-1)  # -1 pour indiquer aucune prédiction
            y_true.append(i)

    # Filtrer les valeurs -1 pour les prédictions manquantes
    y_true_filtered, y_pred_filtered = zip(*[(t, p) for t, p in zip(y_true, y_pred) if p != -1])

    return np.array(y_true_filtered), np.array(y_pred_filtered)

# Chemins et classes
val_dir = "/content/animal_classification/val"
classes = ["bird", "cat", "dog"]
model_path = "/content/animal_classification/runs/classify/train/weights/best.pt"  # Assurez-vous d'avoir le bon chemin

# Charger le modèle
model = YOLO(model_path)

# Évaluer le modèle
y_true_val, y_pred_val = evaluate_model(val_dir, classes, model)
if len(y_true_val) > 0 and len(y_pred_val) > 0:
    accuracy_val = accuracy_score(y_true_val, y_pred_val)
    precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted')
    print(f"Validation Accuracy: {accuracy_val:.4f}")
    print(f"Validation Precision: {precision_val:.4f}")
    print(f"Validation Recall: {recall_val:.4f}")
    print(f"Validation F-score: {fscore_val:.4f}")
else:
    print("Aucune prédiction n'a été effectuée sur l'ensemble de validation.")
