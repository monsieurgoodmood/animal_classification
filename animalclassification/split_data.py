import os
import shutil
from sklearn.model_selection import train_test_split

# Définir le chemin de votre dossier de données
data_path = '/content/drive/MyDrive/ColabNotebooks/raw_data/kaggle_dataset'
classes = ['cat', 'bird', 'dog']

# Créer des dossiers pour les ensembles d'entraînement et de validation
for cls in classes:
    os.makedirs(f'/content/animal_classification/train/{cls}', exist_ok=True)
    os.makedirs(f'/content/animal_classification/val/{cls}', exist_ok=True)

    # Ajuster le chemin pour accéder correctement aux images
    corrected_path = os.path.join(data_path, cls, cls)  # Chemin corrigé

    # Récupérer toutes les images
    images = os.listdir(corrected_path)  # Utiliser le chemin corrigé

    # Vérifier s'il y a suffisamment d'images
    if len(images) > 1:
        # Séparation en ensemble d'entraînement et de validation
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        # Copier les images dans les nouveaux dossiers
        for img in train_imgs:
            shutil.copy(os.path.join(corrected_path, img), f'/content/animal_classification/train/{cls}')
        for img in val_imgs:
            shutil.copy(os.path.join(corrected_path, img), f'/content/animal_classification/val/{cls}')
    else:
        print(f"Not enough images for class {cls}.")
