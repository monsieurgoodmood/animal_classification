from ultralytics import YOLO
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Assumption: The YOLO model has been correctly loaded elsewhere in the script
# model = YOLO('/path/to/your/model.pt')

def evaluate_model(data_dir, classes, model):
    y_true, y_pred = [], []

    # Itérer sur chaque classe et ses images dans le répertoire de données
    for class_id, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            results = model(img_path)

            # La prédiction est renvoyée dans results.xyxy[0] pour la version actuelle de Ultralytics YOLO
            # Vérifier que des prédictions ont été renvoyées
            if len(results.pred) and len(results.pred[0]):
                # Prédiction avec la plus haute confiance
                best_pred = results.pred[0][0]  # Prendre la première prédiction (la plus confiante)
                y_pred_class = int(best_pred[-1])  # Le dernier élément est l'ID de classe
                y_pred.append(y_pred_class)
            else:
                # Aucune prédiction, marquer comme classe inconnue (-1 ou un autre identifiant spécifique)
                y_pred.append(-1)

            y_true.append(class_id)

    # Filtrer les prédictions manquantes si nécessaire
    y_true_filtered, y_pred_filtered = zip(*[(t, p) for t, p in zip(y_true, y_pred) if p != -1])

    return np.array(y_true_filtered), np.array(y_pred_filtered)

# Définition des chemins et des classes
val_dir = '/content/animal_classification/val'
classes = ['bird', 'cat', 'dog']
model_path = '/content/animal_classification/runs/train/exp/weights/best.pt'  # Ajuster selon le chemin correct du modèle
model = YOLO(model_path)

y_true_val, y_pred_val = evaluate_model(val_dir, classes, model)

if len(y_true_val) > 0 and len(y_pred_val) > 0:
    accuracy_val = accuracy_score(y_true_val, y_pred_val)
    precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted')
    print(f'Validation Accuracy: {accuracy_val:.2f}')
    print(f'Precision: {precision_val:.2f}, Recall: {recall_val:.2f}, F-Score: {fscore_val:.2f}')
else:
    print("No valid predictions were made.")
