"""
Конфигурация антиспам-бота
"""
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Токен Telegram бота
BOT_TOKEN = os.getenv("BOT_TOKEN")

# API ключ OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ID администратора
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))

# Белый список групп (только эти группы будут обрабатываться)
ALLOWED_GROUP_IDS = [
    -1002116322225,  # Группа 1
    -4952972324,     # спамтестгруппа (из логов)
    -1001508463207,  # Группа 3
    -1001342298943   # Группа 4
]

# Настройки базы данных
DATABASE_URL = os.getenv("DATABASE_URL")  # Для Railway PostgreSQL
DATABASE_PATH = os.getenv("DATABASE_PATH", "antispam.db")  # Локальная SQLite

# Настройки LLM
LLM_MODEL = "gpt-3.5-turbo"
LLM_MAX_TOKENS = 5
LLM_TEMPERATURE = 0
LLM_TIMEOUT = 10

# Логирование
LOG_LEVEL = "INFO"
