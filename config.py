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
    local_model_path: DirectoryPath = 'artifacts_cifar_10_batch4_2'
    local_model_name: str = 'artifacts_cifar_10_batch4_2.pth'
    batch_size: int = 4
    learning_rate: float = 0.001
    epochs: int = 10
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    patience: int = 5
    
settings = Settings()

logger.remove()
logger.add("app.log", rotation="1 day", retention="2 days", compression="zip", level=settings.log_level)