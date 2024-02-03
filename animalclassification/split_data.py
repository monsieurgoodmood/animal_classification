import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Définir le chemin de votre dossier de données
data_path = '/content/drive/MyDrive/ColabNotebooks/raw_data/kaggle_dataset'
classes = ['cat', 'bird', 'dog']
total_images_per_class = 150  # Total d'images à sélectionner par classe

# Créer des dossiers pour les ensembles d'entraînement, de validation et de test
for cls in classes:
    os.makedirs(f'/content/animal_classification/train/{cls}', exist_ok=True)
    os.makedirs(f'/content/animal_classification/val/{cls}', exist_ok=True)
    os.makedirs(f'/content/animal_classification/test/{cls}', exist_ok=True)

    # Ajuster le chemin pour accéder correctement aux images
    corrected_path = os.path.join(data_path, cls, cls)  # Notez le double cls pour le chemin correct

    # Récupérer toutes les images et mélanger aléatoirement
    images = os.listdir(corrected_path)
    images = list(set(images))  # Supprimer les doublons
    random.shuffle(images)  # Mélanger les images pour sélectionner aléatoirement

    # Sélectionner un nombre fixe d'images si disponible
    if len(images) >= total_images_per_class:
        selected_images = images[:total_images_per_class]
    else:
        print(f"Not enough images for class {cls}. Only {len(images)} available.")
        selected_images = images

    # Séparation en ensemble d'entraînement, de validation et de test
    train_imgs, test_val_imgs = train_test_split(selected_images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(test_val_imgs, test_size=1/3, random_state=42)  # 1/3 de 0.3 sera 0.1 du total

    # Copier les images sélectionnées dans les nouveaux dossiers
    for img in train_imgs:
        shutil.copy(os.path.join(corrected_path, img), f'/content/animal_classification/train/{cls}')
    for img in val_imgs:
        shutil.copy(os.path.join(corrected_path, img), f'/content/animal_classification/val/{cls}')
    for img in test_imgs:
        shutil.copy(os.path.join(corrected_path, img), f'/content/animal_classification/test/{cls}')
