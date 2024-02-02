from ultralytics import YOLO  # Importez la classe YOLO

# Assurez-vous que le modèle et le fichier de données YAML sont spécifiés correctement
model = YOLO('yolov8n-cls.pt')  # Spécifiez le chemin correct au modèle pré-entraîné si nécessaire

# Lancer l'entraînement
# Notez que vous devez vous assurer que 'data' pointe vers le chemin correct de votre fichier YAML
results = model.train(data='animal_data.yaml', epochs=50, imgsz=224, batch=16)
