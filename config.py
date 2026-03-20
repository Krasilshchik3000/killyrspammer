"""
Конфигурация антиспам-бота
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Токен Telegram бота
BOT_TOKEN = os.getenv("BOT_TOKEN")

# API ключ OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ID администратора
ADMIN_ID = int(os.getenv("ADMIN_ID") or "0")
if ADMIN_ID <= 0:
    import logging
    logging.warning("⚠️ ADMIN_ID не установлен или некорректен!")
    ADMIN_ID = -1

# Белый список групп
ALLOWED_GROUP_IDS = [
    -1002116322225,
    -4952972324,
    -1001508463207,
    -1001342298943,
]

# Настройки базы данных
DATABASE_URL = os.getenv("DATABASE_URL")  # PostgreSQL (Railway)
DATABASE_PATH = os.getenv("DATABASE_PATH", "antispam.db")  # SQLite (локальная)

# Настройки LLM
# gpt-5.4-nano: самая быстрая/дешёвая модель ($0.20/1M input), заточена под классификацию
# gpt-5.4-mini: для анализа ошибок и улучшения промптов (мощнее, но всё ещё дешёвая)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-nano")
LLM_IMPROVEMENT_MODEL = os.getenv("LLM_IMPROVEMENT_MODEL", "gpt-5.4-mini")
LLM_MAX_TOKENS = 30
LLM_TEMPERATURE = 0
LLM_TIMEOUT = 10

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 5

# Few-shot: сколько примеров из training_examples подставлять в контекст
FEW_SHOT_EXAMPLES_COUNT = 10

# Combot Anti-Spam (CAS) — бесплатная база спамеров
CAS_API_URL = "https://api.cas.chat/check"

# Сколько сообщений в группе нужно, чтобы считать пользователя «своим» и не проверять через LLM
TRUSTED_USER_MESSAGES = int(os.getenv("TRUSTED_USER_MESSAGES", "3"))

# Автоматическое улучшение промпта
# После скольких ошибок запускать улучшение промпта
AUTO_IMPROVE_AFTER_ERRORS = int(os.getenv("AUTO_IMPROVE_AFTER_ERRORS", "3"))
# Минимум примеров для валидации нового промпта
MIN_VALIDATION_EXAMPLES = int(os.getenv("MIN_VALIDATION_EXAMPLES", "5"))
# Максимум примеров для валидации (больше = точнее, но дороже)
MAX_VALIDATION_EXAMPLES = int(os.getenv("MAX_VALIDATION_EXAMPLES", "30"))

# Логирование
LOG_LEVEL = "INFO"
