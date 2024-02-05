# training_evaluation.py

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_model(model, train_generator, validation_generator, epochs=40):
    """
    Entrena el modelo y lo evalúa con el conjunto de validación.
    """
    history = model.fit(
        train_generator,
        steps_per_epoch=80,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=80)

    return history

def evaluate_model(model, validation_generator):
    """
    Evalúa el modelo con el conjunto de validación y calcula métricas clave.
    """
    validation_generator.reset()
    predictions = model.predict(validation_generator, steps=len(validation_generator))
    y_pred = np.argmax(predictions, axis=1)
    y_true = validation_generator.classes

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Dibuja la matriz de confusión utilizando Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
