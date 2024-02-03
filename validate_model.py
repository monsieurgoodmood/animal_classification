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
            results = model(img_path)
            preds = results.pred[0]

            if len(preds) > 0:
                # Prendre la classe avec la confiance la plus élevée
                pred_classes = preds[:, -1].cpu().numpy()
                confidences = preds[:, -2].cpu().numpy()
                best_pred_idx = np.argmax(confidences)
                pred_class = int(pred_classes[best_pred_idx])
                y_pred.append(pred_class)
            else:
                y_pred.append(None)  # Ou une valeur spéciale indiquant 'aucune détection'

            y_true.append(i)

    # Filtrer les None pour éviter les erreurs dans les métriques
    y_true_filtered, y_pred_filtered = zip(*[(true, pred) for true, pred in zip(y_true, y_pred) if pred is not None])

    return y_true_filtered, y_pred_filtered

# Charger le modèle pré-entraîné
model = YOLO('/content/animal_classification/runs/classify/train/weights/best.pt')

# Définir les chemins des dossiers de validation et de test, et les classes
val_dir = '/content/animal_classification/val'
test_dir = '/content/animal_classification/test'
classes = ['bird', 'cat', 'dog']

# Évaluer sur l'ensemble de validation
print("Évaluation sur l'ensemble de validation:")
y_true_val, y_pred_val = evaluate_model(val_dir, classes, model)
accuracy_val = accuracy_score(y_true_val, y_pred_val)
precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted')
print(f'Validation - Accuracy: {accuracy_val:.2f}, Precision: {precision_val:.2f}, Recall: {recall_val:.2f}, F1 Score: {fscore_val:.2f}')

# Évaluer sur l'ensemble de test
print("\nÉvaluation sur l'ensemble de test:")
y_true_test, y_pred_test = evaluate_model(test_dir, classes, model)
accuracy_test = accuracy_score(y_true_test, y_pred_test)
precision_test, recall_test, fscore_test, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average='weighted')
print(f'Test - Accuracy: {accuracy_test:.2f}, Precision: {precision_test:.2f}, Recall: {recall_test:.2f}, F1 Score: {fscore_test:.2f}')

# Afficher la matrice de confusion pour l'ensemble de test
conf_mat_test = confusion_matrix(y_true_test, y_pred_test)
sns.heatmap(conf_mat_test, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Test Set')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
