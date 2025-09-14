"""
Модуль для работы с базой данных (SQLite или PostgreSQL)
"""
import sqlite3
import os
from datetime import datetime
from config import DATABASE_URL, DATABASE_PATH
import logging

logger = logging.getLogger(__name__)

def get_db_connection():
    """Получить подключение к базе данных"""
    if DATABASE_URL:
        # PostgreSQL для Railway
        import psycopg2
        return psycopg2.connect(DATABASE_URL)
    else:
        # SQLite для локальной разработки
        return sqlite3.connect(DATABASE_PATH)

def init_database():
    """Инициализация базы данных"""
    if DATABASE_URL:
        # PostgreSQL
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Таблица сообщений
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id BIGSERIAL PRIMARY KEY,
                message_id BIGINT,
                chat_id BIGINT,
                user_id BIGINT,
                username TEXT,
                text TEXT,
                created_at TIMESTAMP,
                llm_result TEXT,
                admin_decision TEXT,
                admin_decided_at TIMESTAMP
            )
        ''')
        
        # Таблица обучающих примеров
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_examples (
                id SERIAL PRIMARY KEY,
                text TEXT,
                is_spam BOOLEAN,
                source TEXT,
                created_at TIMESTAMP
            )
        ''')
        
        # Таблица промптов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                id SERIAL PRIMARY KEY,
                prompt_text TEXT,
                version INTEGER,
                created_at TIMESTAMP,
                is_active BOOLEAN DEFAULT FALSE,
                improvement_reason TEXT
            )
        ''')
        
        # Таблица состояний бота
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bot_state (
                id SERIAL PRIMARY KEY,
                admin_id BIGINT,
                awaiting_prompt_edit BOOLEAN DEFAULT FALSE,
                pending_prompt TEXT,
                updated_at TIMESTAMP
            )
        ''')
        
        # Вставляем базовый промпт, если таблица пустая
        cursor.execute("SELECT COUNT(*) FROM prompts")
        if cursor.fetchone()[0] == 0:
            base_prompt = """Проанализируй сообщение из телеграм-группы и ответь строго одним из трёх вариантов:
СПАМ
НЕ_СПАМ  
ВОЗМОЖНО_СПАМ

Считай особенно подозрительными: безадресные вакансии/работу "без опыта/высокий доход", призывы писать в ЛС/бота/внешние ссылки, сердечки 💘/💝 с намёком на интим-услуги. Если данных мало — выбирай ВОЗМОЖНО_СПАМ.

Сообщение: «{message_text}»

Ответ:"""
            cursor.execute('''
                INSERT INTO prompts (prompt_text, version, created_at, is_active, improvement_reason)
                VALUES (%s, 1, %s, TRUE, 'Базовый промпт')
            ''', (base_prompt, datetime.now()))
        
    else:
        # SQLite (старый код)
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Таблица сообщений
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                message_id INTEGER,
                chat_id INTEGER,
                user_id INTEGER,
                username TEXT,
                text TEXT,
                created_at TIMESTAMP,
                llm_result TEXT,
                admin_decision TEXT,
                admin_decided_at TIMESTAMP
            )
        ''')
        
        # Таблица обучающих примеров
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                is_spam BOOLEAN,
                source TEXT,
                created_at TIMESTAMP
            )
        ''')
        
        # Таблица промптов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_text TEXT,
                version INTEGER,
                created_at TIMESTAMP,
                is_active BOOLEAN DEFAULT FALSE,
                improvement_reason TEXT
            )
        ''')
        
        # Таблица состояний бота
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bot_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_id INTEGER,
                awaiting_prompt_edit BOOLEAN DEFAULT FALSE,
                pending_prompt TEXT,
                updated_at TIMESTAMP
            )
        ''')
        
        # Вставляем базовый промпт, если таблица пустая
        cursor.execute("SELECT COUNT(*) FROM prompts")
        if cursor.fetchone()[0] == 0:
            base_prompt = """Проанализируй сообщение из телеграм-группы и ответь строго одним из трёх вариантов:
СПАМ
НЕ_СПАМ  
ВОЗМОЖНО_СПАМ

Считай особенно подозрительными: безадресные вакансии/работу "без опыта/высокий доход", призывы писать в ЛС/бота/внешние ссылки, сердечки 💘/💝 с намёком на интим-услуги. Если данных мало — выбирай ВОЗМОЖНО_СПАМ.

Сообщение: «{message_text}»

Ответ:"""
            cursor.execute('''
                INSERT INTO prompts (prompt_text, version, created_at, is_active, improvement_reason)
                VALUES (?, 1, ?, TRUE, 'Базовый промпт')
            ''', (base_prompt, datetime.now()))
    
    conn.commit()
    conn.close()
    logger.info("✅ База данных инициализирована")

def execute_query(query, params=None, fetch=False):
    """Универсальное выполнение запроса"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if DATABASE_URL:
            # PostgreSQL - заменяем ? на %s
            query = query.replace('?', '%s')
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        result = None
        if fetch == 'one':
            result = cursor.fetchone()
        elif fetch == 'all':
            result = cursor.fetchall()
        
        conn.commit()
        conn.close()
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Ошибка выполнения запроса: {e}")
        logger.error(f"📝 Запрос: {query}")
        logger.error(f"📝 Параметры: {params}")
        raise

def set_bot_state(admin_id, awaiting_prompt_edit=False, pending_prompt=None):
    """Сохранить состояние бота"""
    try:
        # Удаляем старое состояние для этого админа
        execute_query("DELETE FROM bot_state WHERE admin_id = ?", (admin_id,))
        
        # Добавляем новое состояние
        execute_query('''
            INSERT INTO bot_state (admin_id, awaiting_prompt_edit, pending_prompt, updated_at)
            VALUES (?, ?, ?, ?)
        ''', (admin_id, awaiting_prompt_edit, pending_prompt, datetime.now()))
        
        logger.info(f"💾 Состояние бота сохранено: awaiting_prompt_edit={awaiting_prompt_edit}")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения состояния: {e}")

def get_bot_state(admin_id):
    """Получить состояние бота"""
    try:
        result = execute_query(
            "SELECT awaiting_prompt_edit, pending_prompt FROM bot_state WHERE admin_id = ? ORDER BY updated_at DESC LIMIT 1",
            (admin_id,), fetch='one'
        )
        if result:
            awaiting_edit, pending = result
            logger.info(f"📖 Загружено состояние: awaiting_prompt_edit={awaiting_edit}")
            return awaiting_edit, pending
        return False, None
    except Exception as e:
        logger.error(f"❌ Ошибка получения состояния: {e}")
        return False, None
