# main.py

from config import *
from data_management import setup_data
from model import create_model
from data_preprocessing import create_generators
from training_evaluation import train_and_evaluate_model
from visualization import plot_train_eval, display_misclassified_images

def main():
    setup_data()
    model = create_model()
    print(model.summary())
    train_generator, validation_generator = create_generators(TRAIN_DIR, EVAL_DIR)
    history = train_and_evaluate_model(model, train_generator, validation_generator)
    plot_train_eval(history)
    display_misclassified_images(model, validation_generator)

if __name__ == "__main__":
    main()
