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
        
        # Вставляем базовый промпт, если таблица пустая
        cursor.execute("SELECT COUNT(*) FROM prompts")
        if cursor.fetchone()[0] == 0:
            from main import SPAM_CHECK_PROMPT
            cursor.execute('''
                INSERT INTO prompts (prompt_text, version, created_at, is_active, improvement_reason)
                VALUES (%s, 1, %s, TRUE, 'Базовый промпт')
            ''', (SPAM_CHECK_PROMPT, datetime.now()))
        
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
        
        # Вставляем базовый промпт, если таблица пустая
        cursor.execute("SELECT COUNT(*) FROM prompts")
        if cursor.fetchone()[0] == 0:
            from main import SPAM_CHECK_PROMPT
            cursor.execute('''
                INSERT INTO prompts (prompt_text, version, created_at, is_active, improvement_reason)
                VALUES (?, 1, ?, TRUE, 'Базовый промпт')
            ''', (SPAM_CHECK_PROMPT, datetime.now()))
    
    conn.commit()
    conn.close()
    logger.info("✅ База данных инициализирована")

def execute_query(query, params=None, fetch=False):
    """Универсальное выполнение запроса"""
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
