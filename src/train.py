from fastai.vision.all import *
from pathlib import Path

def load_data(path: Path):
    """Load and preprocess the MNIST dataset."""
    dblock = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name='train', valid_name='valid'),
        get_y=parent_label,
        item_tfms=Resize(28)
    )
    return dblock.dataloaders(path)

def train_model(dls, epochs=1, model_name="mnist_model.pkl"):
    """Train a model using the given DataLoader."""
    learn = vision_learner(dls, resnet18, metrics=accuracy)
    learn.fine_tune(epochs)
    learn.export(f"src/model/{model_name}")
    print(f"Model saved as src/model/{model_name}")
    return learn

if __name__ == "__main__":
    # Define paths
    dataset_path = untar_data(URLs.MNIST_SAMPLE)
    
    # Load data
    print("Loading data...")
    dls = load_data(dataset_path)
    
    # Train the model
    print("Training the model...")
    train_model(dls)
