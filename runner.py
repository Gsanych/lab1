import torch
from model_service import ModelService
from loguru import logger
from config import settings


@logger.catch 
def main():
    logger.info("running the application...")
    ml_svc = ModelService()
    ml_svc.load()

    image_path = 'resources/test_mountain.jpg' 

    # pred = ml_svc.predict(image_path)
    # logger.info(f"prediction = {pred}")

if __name__ == '__main__':
    main()