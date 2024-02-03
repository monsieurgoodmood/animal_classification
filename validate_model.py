from ultralytics import YOLO
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
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

            # S'assurer que les résultats sont sous la forme attendue
            if results.xyxy[0].shape[0] > 0:  # Vérifiez si des prédictions ont été faites
                for pred in results.xyxy[0]:
                    y_pred.append(int(pred[-1].item()))  # Ajouter la classe prédite
            else:
                y_pred.append(-1)  # Indiquer l'absence de détection

            y_true.append(i)  # Ajouter la classe réelle

    # Filtrer les valeurs -1 pour les prédictions manquantes
    y_true_filtered, y_pred_filtered = zip(*[(t, p) for t, p in zip(y_true, y_pred) if p != -1])

    if not y_true_filtered:
        return np.array([]), np.array([])  # Retourne des tableaux vides si aucune prédiction valide
    return np.array(y_true_filtered), np.array(y_pred_filtered)

# Chemin vers le modèle et les dossiers de données
model_path = '/content/animal_classification/runs/train/weights/best.pt'
val_dir = '/content/animal_classification/val'
classes = ['bird', 'cat', 'dog']

# Initialisation du modèle
model = YOLO(model_path)

# Évaluation sur l'ensemble de validation
print("Évaluation sur l'ensemble de validation:")
y_true_val, y_pred_val = evaluate_model(val_dir, classes, model)
if y_true_val.size > 0 and y_pred_val.size > 0:
    accuracy_val = accuracy_score(y_true_val, y_pred_val)
    precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted')
    print(f'Validation - Accuracy: {accuracy_val:.2f}, Precision: {precision_val:.2f}, Recall: {recall_val:.2f}, F1 Score: {fscore_val:.2f}')

    # Affichage de la matrice de confusion
    conf_mat_val = confusion_matrix(y_true_val, y_pred_val)
    sns.heatmap(conf_mat_val, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Validation Set')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
else:
    print("Aucune prédiction valide n'a été faite sur l'ensemble de validation.")
