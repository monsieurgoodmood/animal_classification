from ultralytics import YOLO  # Assurez-vous d'avoir installé le package ultralytics
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Charger le modèle entraîné
model_path = 'runs/train/exp/weights/best.pt'  # Ajustez le chemin selon votre configuration
model = YOLO(model_path)

# Liste des chemins des images de test
image_paths = ['path/to/cat.jpg', 'path/to/dog.jpg', 'path/to/bird.jpg']  # Ajustez les chemins

# Faire des prédictions et afficher les résultats
fig, axs = plt.subplots(4, 3, figsize=(20, 20))  # Ajustez la taille si nécessaire
axs = axs.flatten()

for i, image_path in enumerate(image_paths * 4):  # Multipliez la liste pour remplir la grille d'images
    results = model(image_path)
    img = results.render()[0]  # Rendu de l'image annotée
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir BGR en RGB

    axs[i].imshow(img)
    axs[i].axis('off')

plt.tight_layout()
plt.show()
