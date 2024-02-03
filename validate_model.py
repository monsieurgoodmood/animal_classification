from ultralytics import YOLO
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate_model(data_dir, classes, model):
    y_true = []
    y_pred = []

    for i, cls in enumerate(classes):
        class_dir = os.path.join(data_dir, cls)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            results = model(img_path)  # Effectuer la prédiction avec le chemin de l'image

            # S'assurer que les résultats sont sous la forme attendue
            if len(results) > 0 and hasattr(results[0], 'xyxy'):
                preds = results[0].xyxy[0]  # Prendre les prédictions du premier lot si disponible

                if preds.shape[0] > 0:
                    best_pred_idx = preds[:, 4].argmax()  # Trouver l'indice avec la confiance la plus élevée
                    pred_class = int(preds[best_pred_idx, -1])  # Obtenir la classe de la meilleure prédiction
                    y_pred.append(pred_class)
                else:
                    y_pred.append(-1)  # Indiquer l'absence de détection par -1
            else:
                y_pred.append(-1)  # Indiquer l'absence de détection par -1

            y_true.append(i)

    # Filtrer les valeurs -1 pour les prédictions manquantes
    y_true, y_pred = zip(*[(t, p) for t, p in zip(y_true, y_pred) if p != -1])

    return np.array(y_true), np.array(y_pred)

# Chemin vers le modèle et les dossiers de données
model_path = '/content/animal_classification/runs/classify/train/weights/best.pt'
val_dir = '/content/animal_classification/val'
test_dir = '/content/animal_classification/test'
classes = ['bird', 'cat', 'dog']

# Initialisation du modèle
model = YOLO(model_path)

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

# Affichage de la matrice de confusion
conf_mat_test = confusion_matrix(y_true_test, y_pred_test)
sns.heatmap(conf_mat_test, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Test Set')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
