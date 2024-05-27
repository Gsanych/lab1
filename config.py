from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import (
    FilePath,
    DirectoryPath,
    BaseModel,
    Field,
)
from loguru import logger 

class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, env_file='.env', env_file_encoding='utf-8')

    data_path: str = './data'
    log_level: str = 'INFO'
    local_model_path: DirectoryPath = 'artifacts'
    local_model_name: str = 'cifar100_resnet_10epochs.pth'
    batch_size: int = 64
    learning_rate: float = 0.01
    epochs: int = 10
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
settings = Settings()

logger.remove()
logger.add("app.log", rotation="1 day", retention="2 days", compression="zip", level=settings.log_level)

