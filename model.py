from ultralytics import YOLO  # Importez la classe YOLO

# Chargement et entraînement du modèle
model = YOLO('yolov8n-cls.pt')  # Créez un modèle à partir d'un fichier de configuration
results = model.train(data='/content/animal_classification/data.yaml',  # Spécifiez le chemin vers votre fichier de données
                      epochs=50,  # Définissez le nombre d'époques d'entraînement
                      imgsz=224,  # Définissez la taille des images
                      batch_size=16)  # Définissez la taille du lot
