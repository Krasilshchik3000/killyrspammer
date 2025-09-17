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
        try:
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL)
            logger.info("✅ Подключение к PostgreSQL установлено")
            return conn
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к PostgreSQL: {e}")
            raise
    else:
        # SQLite для локальной разработки
        conn = sqlite3.connect(DATABASE_PATH)
        logger.info("✅ Подключение к SQLite установлено")
        return conn

def init_database():
    """Инициализация базы данных"""
    logger.info("🔄 Инициализация БД - удаляю старые таблицы промптов")
    
    if DATABASE_URL:
        # PostgreSQL
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # УДАЛЯЕМ старую таблицу промптов если существует
        try:
            cursor.execute("DROP TABLE IF EXISTS prompts")
            logger.info("🗑️ Удалена старая таблица prompts")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось удалить старую таблицу: {e}")
        
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
        
        # Таблица промпта (только один активный)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS current_prompt (
                id SERIAL PRIMARY KEY,
                prompt_text TEXT,
                updated_at TIMESTAMP,
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
        
        # НЕ СОЗДАЕМ ПРОМПТ ПРИ ИНИЦИАЛИЗАЦИИ!
        # Промпт должен создаваться ТОЛЬКО админом через команды
        cursor.execute("SELECT COUNT(*) FROM current_prompt")
        if cursor.fetchone()[0] == 0:
            logger.warning("⚠️ PostgreSQL: Таблица current_prompt пустая - промпт должен быть установлен админом")
            # НЕ ВСТАВЛЯЕМ НИЧЕГО!
        
    else:
        # SQLite (старый код)
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # УДАЛЯЕМ старую таблицу промптов если существует
        try:
            cursor.execute("DROP TABLE IF EXISTS prompts")
            logger.info("🗑️ Удалена старая таблица prompts из SQLite")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось удалить старую таблицу из SQLite: {e}")
        
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
        
        # Таблица промпта (только один активный)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS current_prompt (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_text TEXT,
                updated_at TIMESTAMP,
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
        
        # НЕ СОЗДАЕМ ПРОМПТ ПРИ ИНИЦИАЛИЗАЦИИ!
        # Промпт должен создаваться ТОЛЬКО админом через команды
        cursor.execute("SELECT COUNT(*) FROM current_prompt")
        if cursor.fetchone()[0] == 0:
            logger.warning("⚠️ SQLite: Таблица current_prompt пустая - промпт должен быть установлен админом")
            # НЕ ВСТАВЛЯЕМ НИЧЕГО!
    
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
