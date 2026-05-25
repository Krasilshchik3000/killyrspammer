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
# Список моделей в порядке предпочтения. Бот при старте автодетектит
# первую доступную модель из каждого списка и использует её.
# Можно переопределить через env переменную LLM_MODEL (одна модель).
LLM_MODEL_CANDIDATES = [
    "gpt-5.5-nano", "gpt-5.5-mini", "gpt-5.4-nano", "gpt-5.4-mini",
    "gpt-5-nano", "gpt-5-mini", "gpt-4.1-nano", "gpt-4o-mini",
]
LLM_IMPROVEMENT_MODEL_CANDIDATES = [
    "gpt-5.5", "gpt-5.5-mini", "gpt-5.4", "gpt-5.4-mini",
    "gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4o",
]

# Переопределение через env (если задано — используется именно эта модель, без автодетекта)
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_IMPROVEMENT_MODEL = os.getenv("LLM_IMPROVEMENT_MODEL", "")
LLM_VALIDATION_MODEL = os.getenv("LLM_VALIDATION_MODEL", "")
LLM_MAX_TOKENS = 150  # Enough for {"result":"...","reasoning":"..."}
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
AUTO_IMPROVE_AFTER_ERRORS = int(os.getenv("AUTO_IMPROVE_AFTER_ERRORS", "5"))
# Cooldown между автозапусками улучшения (минуты)
AUTO_IMPROVE_COOLDOWN_MINUTES = int(os.getenv("AUTO_IMPROVE_COOLDOWN_MINUTES", "60"))
# Минимум примеров для валидации нового промпта
MIN_VALIDATION_EXAMPLES = int(os.getenv("MIN_VALIDATION_EXAMPLES", "10"))
# Максимум spam-примеров для валидации
MAX_VALIDATION_EXAMPLES = int(os.getenv("MAX_VALIDATION_EXAMPLES", "200"))
# Сколько admin-reviewed правильных примеров включать в валидацию
REGRESSION_CHECK_SAMPLES = int(os.getenv("REGRESSION_CHECK_SAMPLES", "50"))
# Сколько обычных сообщений (без подозрений) включать в валидацию
ORDINARY_MESSAGES_SAMPLES = int(os.getenv("ORDINARY_MESSAGES_SAMPLES", "300"))
# Сколько попыток улучшить промпт за один цикл (каждая — вызов LLM)
MAX_IMPROVEMENT_ATTEMPTS = int(os.getenv("MAX_IMPROVEMENT_ATTEMPTS", "5"))
# Минимальный прирост точности для применения нового промпта (5%)
MIN_ACCURACY_GAIN = float(os.getenv("MIN_ACCURACY_GAIN", "0.05"))
# Максимум регрессий (раньше правильно → теперь неправильно)
MAX_REGRESSIONS = int(os.getenv("MAX_REGRESSIONS", "3"))

# Логирование
LOG_LEVEL = "INFO"
