from ultralytics import YOLO
import os
import cv2
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
            img = cv2.imread(img_path)  # Chargement de l'image
            results = model(img)  # Effectuer la prédiction

            # Vérification si des objets sont détectés
            if results.pred[0].shape[0] > 0:
                preds = results.pred[0]
                # Obtenir les indices des classes avec la plus grande confiance
                max_conf_indices = preds[:, 5:].max(1).indices.cpu().numpy()
                # Obtenir l'indice de la classe prédite avec la plus haute confiance pour la première détection
                pred_class = max_conf_indices[0]
                y_pred.append(pred_class)
            else:
                y_pred.append(-1)  # -1 pour indiquer qu'aucune classe n'a été détectée

            y_true.append(i)

    return y_true, y_pred

# Assurez-vous que le chemin vers le modèle et les dossiers de données sont corrects
model_path = '/content/animal_classification/runs/classify/train/weights/best.pt'
model = YOLO(model_path)

val_dir = '/content/animal_classification/val'
test_dir = '/content/animal_classification/test'
classes = ['bird', 'cat', 'dog']

print("Évaluation sur l'ensemble de validation:")
y_true_val, y_pred_val = evaluate_model(val_dir, classes, model)
accuracy_val = accuracy_score(y_true_val, y_pred_val)
precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted')
print(f'Validation - Accuracy: {accuracy_val:.2f}, Precision: {precision_val:.2f}, Recall: {recall_val:.2f}, F1 Score: {fscore_val:.2f}')

print("\nÉvaluation sur l'ensemble de test:")
y_true_test, y_pred_test = evaluate_model(test_dir, classes, model)
accuracy_test = accuracy_score(y_true_test, y_pred_test)
precision_test, recall_test, fscore_test, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average='weighted')
print(f'Test - Accuracy: {accuracy_test:.2f}, Precision: {precision_test:.2f}, Recall: {recall_test:.2f}, F1 Score: {fscore_test:.2f}')

conf_mat_test = confusion_matrix(y_true_test, y_pred_test)
sns.heatmap(conf_mat_test, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Test Set')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
