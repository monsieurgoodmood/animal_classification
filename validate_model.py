# Importation des bibliothèques nécessaires
from ultralytics import YOLO
import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Fonction pour évaluer le modèle sur un ensemble de données spécifié
def evaluate_model(data_dir, classes, model):
    y_true = []  # Liste pour stocker les vraies classes
    y_pred = []  # Liste pour stocker les classes prédites

    # Itération sur chaque classe dans l'ensemble de données
    for i, cls in enumerate(classes):
        class_dir = os.path.join(data_dir, cls)  # Chemin du dossier pour la classe actuelle
        # Itération sur chaque image dans le dossier de classe
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)  # Chemin de l'image
            img = cv2.imread(img_path)  # Chargement de l'image
            results = model(img)  # Prédiction sur l'image

            # Extraction des prédictions
            preds = results.pred[0] if results.pred and len(results.pred[0]) else []

            if len(preds) > 0:
                # Extraction des classes prédites et des confiances
                pred_classes = preds[:, -1].int().tolist()
                confidences = preds[:, 4].tolist()
                best_pred_idx = np.argmax(confidences)  # Index de la prédiction avec la plus haute confiance
                pred_class = pred_classes[best_pred_idx]  # Classe prédite avec la plus haute confiance
                y_pred.append(pred_class)
            else:
                y_pred.append(None)  # Aucune prédiction

            y_true.append(i)

    # Filtrer les None pour éviter les erreurs dans les métriques
    y_true_filtered, y_pred_filtered = zip(*[(true, pred) for true, pred in zip(y_true, y_pred) if pred is not None])

    return y_true_filtered, y_pred_filtered

# Chargement du modèle pré-entraîné
model_path = '/content/animal_classification/runs/classify/train/weights/best.pt'  # Assurez-vous que le chemin est correct
model = YOLO(model_path)

# Chemins vers les dossiers de validation et de test
val_dir = '/content/animal_classification/val'
test_dir = '/content/animal_classification/test'
classes = ['bird', 'cat', 'dog']  # Liste des classes

# Évaluation sur l'ensemble de validation
print("Évaluation sur l'ensemble de validation:")
y_true_val, y_pred_val = evaluate_model(val_dir, classes, model)
accuracy_val = accuracy_score(y_true_val, y_pred_val)
precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted')
print(f'Validation - Accuracy: {accuracy_val:.2f}, Precision: {precision_val:.2f}, Recall: {recall_val:.2f}, F1 Score: {fscore_val:.2f}')

# Évaluation sur l'ensemble de test
print("\nÉvaluation sur l'ensemble de test:")
y_true_test, y_pred_test = evaluate_model(test_dir, classes, model)
accuracy_test = accuracy_score(y_true_test, y_pred_test)
precision_test, recall_test, fscore_test, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average='weighted')
print(f'Test - Accuracy: {accuracy_test:.2f}, Precision: {precision_test:.2f}, Recall: {recall_test:.2f}, F1 Score: {fscore_test:.2f}')

# Affichage de la matrice de confusion pour l'ensemble de test
conf_mat_test = confusion_matrix(y_true_test, y_pred_test)
sns.heatmap(conf_mat_test, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Test Set')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
