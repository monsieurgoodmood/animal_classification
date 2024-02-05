# visualization.py

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from IPython.display import display, HTML
import base64

def image_to_base64(image_path):
    """
    Convierte una imagen en su ruta dada a una cadena base64.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_image_html(title, images, base_dir):
    """
    Crea HTML para mostrar un conjunto de imágenes con un título.
    """
    title_html = f"<div style='text-align: center; margin-bottom: 20px;'><h2>{title}</h2></div>"
    images_html = ' '.join([f"<img style='width: 150px; margin: 10px; display: inline-block;' src='data:image/jpeg;base64,{image_to_base64(os.path.join(base_dir, img))}' />" for img in images])
    return title_html + images_html

def plot_training_history(history):
    """
    Dibuja las gráficas de accuracy y loss de entrenamiento y validación.
    """
    acc = history['sparse_categorical_accuracy']
    val_acc = history['val_sparse_categorical_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Entrenamiento acc')
    plt.plot(epochs, val_acc, 'b', label='Validación acc')
    plt.title('Accuracy de entrenamiento y validación')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Entrenamiento loss')
    plt.plot(epochs, val_loss, 'b', label='Validación loss')
    plt.title('Loss de entrenamiento y validación')
    plt.legend()

    plt.show()

def display_misclassified_images(model, val_generator, class_names, num_images=10):
    """
    Muestra imágenes mal clasificadas por el modelo junto con las predicciones y etiquetas verdaderas.
    """
    for i, (x_batch, y_batch) in enumerate(val_generator):
        predictions = model.predict(x_batch)
        predicted_classes = np.argmax(predictions, axis=1)
        misclassified_idxs = np.where(predicted_classes != y_batch)[0]
        if len(misclassified_idxs) == 0:
            continue

        plt.figure(figsize=(15, 5))
        for j, idx in enumerate(misclassified_idxs[:num_images]):
            plt.subplot(1, num_images, j+1)
            plt.axis('off')
            plt.imshow(img_to_array(x_batch[idx]), cmap='gray', interpolation='nearest')
            predicted_label = class_names[predicted_classes[idx]]
            true_label = class_names[int(y_batch[idx])]
            plt.title(f"Pred: {predicted_label}\nTrue: {true_label}", color=("green" if predicted_label == true_label else "red"))

        if i * val_generator.batch_size >= num_images:
            break
        plt.tight_layout()
        plt.show()
