from ultralytics import YOLO  # Importez la classe YOLO

# Charger le modèle pré-entraîné
model = YOLO('yolov8n-cls.pt')

# Lancer l'entraînement
model.train(data='data.yaml', epochs=100, imgsz=224, batch_size=16)
