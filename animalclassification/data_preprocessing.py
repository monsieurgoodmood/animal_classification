# data_preprocessing.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TRAIN_DIR, EVAL_DIR

def create_generators(batch_size=32, target_size=(150, 150)):
    """
    Crea generadores de imágenes para entrenamiento y validación.

    Parámetros:
    - batch_size: Tamaño del lote para el generador.
    - target_size: El tamaño al cual se redimensionarán las imágenes.

    Retorna:
    - train_generator: Generador para los datos de entrenamiento.
    - validation_generator: Generador para los datos de validación.
    """
    # Crear un generador de datos con aumento para el conjunto de entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Crear un generador de datos sin aumento para el conjunto de validación
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Crear el generador de entrenamiento
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse'
    )

    # Crear el generador de validación
    validation_generator = validation_datagen.flow_from_directory(
        EVAL_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse'
    )

    return train_generator, validation_generator
