# data_management.py

import os
import shutil
import zipfile
import tarfile
import random
from config import BASE_DIR, CATS_AND_DOGS_ZIP, CALTECH_BIRDS_TAR, TRAIN_DIR, EVAL_DIR

def extract_data():
    """Extrae los archivos zip y tar en el directorio base."""
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    with zipfile.ZipFile(CATS_AND_DOGS_ZIP, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)

    with tarfile.open(CALTECH_BIRDS_TAR, 'r') as tar_ref:
        tar_ref.extractall(BASE_DIR)

def prepare_directories():
    """Crea los directorios necesarios para el entrenamiento y la evaluación si no existen."""
    directories = [
        os.path.join(TRAIN_DIR, 'cats'),
        os.path.join(TRAIN_DIR, 'dogs'),
        os.path.join(TRAIN_DIR, 'birds'),
        os.path.join(EVAL_DIR, 'cats'),
        os.path.join(EVAL_DIR, 'dogs'),
        os.path.join(EVAL_DIR, 'birds')
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def organize_data():
    """Organiza los datos en directorios de entrenamiento y evaluación."""
    # Esta función asume que los datos extraídos tienen una estructura específica
    # y necesitará ser adaptada según cómo se extraen y organizan los datos reales

    for animal in ['Dog', 'Cat']:
        origin_dir = os.path.join(BASE_DIR, f'PetImages/{animal}')
        for image in os.listdir(origin_dir):
            destination_dir = os.path.join(TRAIN_DIR if random.random() < 0.7 else EVAL_DIR, animal.lower() + 's')
            shutil.move(os.path.join(origin_dir, image), destination_dir)

    # Las aves requieren un tratamiento especial si están en subdirectorios
    origin_dir = os.path.join(BASE_DIR, 'CUB_200_2011/images')
    for subdir in os.listdir(origin_dir):
        subdir_path = os.path.join(origin_dir, subdir)
        if os.path.isdir(subdir_path):
            for image in os.listdir(subdir_path):
                destination_dir = os.path.join(TRAIN_DIR if random.random() < 0.7 else EVAL_DIR, 'birds')
                shutil.move(os.path.join(subdir_path, image), destination_dir)

def clean_data():
    """Elimina imágenes corruptas o no deseadas."""
    for root, _, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) == 0:
                    os.remove(file_path)

def setup_data():
    """Ejecuta todos los pasos necesarios para preparar los datos."""
    extract_data()
    prepare_directories()
    organize_data()
    clean_data()

if __name__ == "__main__":
    setup_data()
