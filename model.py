from ultralytics import YOLO

# Charger le modèle
model = YOLO('yolov8n-cls.pt')

# Configurer les paramètres d'entraînement et lancer l'entraînement
model.train(data='/content/animal_classification/data.yaml',
            epochs=50,
            imgsz=224,
            batch=16)  # Utilisez 'batch' au lieu de 'batch_size'
