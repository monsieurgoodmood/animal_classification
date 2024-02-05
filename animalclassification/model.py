# model.py

from tensorflow.keras import layers, models, optimizers
import tensorflow as tf

def create_model():
    """Crea y compila un modelo de CNN para clasificación de imágenes."""
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  #Capa de Dropout para regularización

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(3, activation='softmax')
    ])

    # Compilar el modelo
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizers.Adam(0.005),  #hacer el optimizador más pequeño para buscar más lentamente el menor error
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return model
