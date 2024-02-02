import os
import shutil
from sklearn.model_selection import train_test_split

# Définir le chemin de votre dossier de données
data_path = '/content/drive/MyDrive/ColabNotebooks/raw_data/kaggle_animal'
classes = ['cat', 'bird', 'dog']

# Créer des dossiers pour les ensembles d'entraînement et de validation
for cls in classes:
    os.makedirs(f'/content/animal_classification/train/{cls}', exist_ok=True)
    os.makedirs(f'/content/animal_classification/val/{cls}', exist_ok=True)

    # Récupérer toutes les images
    images = os.listdir(os.path.join(data_path, cls))

    # Séparation en ensemble d'entraînement et de validation
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    # Copier les images dans les nouveaux dossiers
    for img in train_imgs:
        shutil.copy(os.path.join(data_path, cls, img), f'/content/animal_classification/train/{cls}')
    for img in val_imgs:
        shutil.copy(os.path.join(data_path, cls, img), f'/content/animal_classification/val/{cls}')
