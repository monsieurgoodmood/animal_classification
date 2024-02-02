from ultralytics import YOLO

# Charger un modèle pré-entraîné pour la classification
model = YOLO('yolov8n-cls.pt')  # Le modèle sera téléchargé automatiquement s'il n'est pas trouvé localement

# Lancer l'entraînement
results = model.train(data='/content/animal_classification/animal_data.yaml', # Chemin vers votre fichier YAML
                      epochs=50,  # Nombre d'époques pour l'entraînement
                      imgsz=224,  # Taille des images pour l'entraînement
                      batch_size=16)  # Taille du lot pour l'entraînement
