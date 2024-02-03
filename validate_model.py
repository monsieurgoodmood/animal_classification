from ultralytics import YOLO
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(data_dir, classes, model):
    y_true = []
    y_pred = []

    for i, cls in enumerate(classes):
        class_dir = os.path.join(data_dir, cls)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            results = model(img_path)  # Effectuer la prédiction avec le chemin de l'image

            # Adapter la vérification aux erreurs observées
            if results:  # Si `results` n'est pas vide
                predictions = results.pred[0]  # Accéder aux prédictions
                if predictions.shape[0]:  # Si des prédictions sont présentes
                    best_pred = predictions[predictions[:, 4].argmax()]  # Sélectionner la prédiction avec la plus haute confiance
                    y_pred.append(int(best_pred[-1].item()))  # Ajouter la classe prédite
                else:
                    y_pred.append(-1)  # Aucune prédiction valide
            else:
                y_pred.append(-1)  # Aucun résultat

            y_true.append(i)  # Ajouter la classe réelle

    # Filtrer les valeurs -1 pour les prédictions manquantes
    y_true_filtered, y_pred_filtered = zip(*[(t, p) for t, p in zip(y_true, y_pred) if p != -1])

    if not y_true_filtered:
        return np.array([]), np.array([])  # Retourne des tableaux vides si aucune prédiction valide
    return np.array(y_true_filtered), np.array(y_pred_filtered)

# Initialisation et évaluation du modèle
model_path = '/content/animal_classification/runs/classify/train/weights/best.pt'  # Assurez-vous que le chemin est correct
val_dir = '/content/animal_classification/val'
classes = ['bird', 'cat', 'dog']
model = YOLO(model_path)

print("Évaluation sur l'ensemble de validation:")
y_true_val, y_pred_val = evaluate_model(val_dir, classes, model)
if y_true_val.size > 0 and y_pred_val.size > 0:
    accuracy_val = accuracy_score(y_true_val, y_pred_val)
    precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted')
    print(f'Accuracy: {accuracy_val:.2f}, Precision: {precision_val:.2f}, Recall: {recall_val:.2f}, F1 Score: {fscore_val:.2f}')
else:
    print("Aucune prédiction valide n'a été faite.")
