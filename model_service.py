from pathlib import Path
import pickle as pk
import torch
from model import build_model
from model import load_model
from model import get_model
from torchvision import transforms
from PIL import Image
from config import settings
from loguru import logger
import pickle as pk

class ModelService:
    def __init__(self):
        self.model = None

    def load(self):
        logger.info(f"checking the existance of model config file at {settings.local_model_path}/{settings.local_model_name}")
        model_path = Path(f'{settings.local_model_path}/{settings.local_model_name}')

        if not model_path.exists():
            logger.warning(f"model at {settings.local_model_path}/{settings.local_model_name} was not found -> building {settings.local_model_name}")
            build_model()

        logger.info(f"model {settings.local_model_name} exists! -> loading model configuration file")

        self.model = torch.load(model_path)

        # load_model(model_path)
        

    def predict(self, image_path):

        cifar100_classes = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
            'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
            'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
            'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
            'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
            'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
            'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree',
            'wolf', 'woman', 'worm'
        ]

        logger.info(f"transforming image")
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])

        image = Image.open(image_path).convert('RGB')  
        image = transform(image).unsqueeze(0)

        logger.info(f"making prediction!")

        # Перевірка доступності CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        logger.add("predict_{time}.log", rotation="500 MB")  # Налаштування логування у файл

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = cifar100_classes[predicted.item()]
            
            return predicted_class
        
   