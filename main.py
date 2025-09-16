import asyncio
import logging
import os
from datetime import datetime
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, BotCommand
from aiogram import F
import sqlite3
import re
from enum import Enum
from openai import AsyncOpenAI
from config import BOT_TOKEN, OPENAI_API_KEY, ADMIN_ID, ALLOWED_GROUP_IDS

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные
bot = None
dp = Dispatcher()

# Глобальная переменная для хранения предложенного промпта
pending_prompt = None
awaiting_prompt_edit = False
openai_client = None

class SpamResult(Enum):
    SPAM = "СПАМ"
    NOT_SPAM = "НЕ_СПАМ"  
    MAYBE_SPAM = "ВОЗМОЖНО_СПАМ"

# Промпт для проверки спама
SPAM_CHECK_PROMPT = """Проанализируй сообщение из телеграм-группы и ответь строго одним из трёх вариантов:
СПАМ
НЕ_СПАМ  
ВОЗМОЖНО_СПАМ

Считай особенно подозрительными: безадресные вакансии/работу "без опыта/высокий доход", призывы писать в ЛС/бота/внешние ссылки, сердечки 💘/💝 с намёком на интим-услуги. Если данных мало — выбирай ВОЗМОЖНО_СПАМ.

Сообщение: «{message_text}»

Ответ:"""

def init_database():
    """Инициализация базы данных"""
    conn = sqlite3.connect('antispam.db')
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
        cursor.execute('''
            INSERT INTO prompts (prompt_text, version, created_at, is_active, improvement_reason)
            VALUES (?, 1, ?, TRUE, 'Базовый промпт')
        ''', (SPAM_CHECK_PROMPT, datetime.now()))
    
    conn.commit()
    conn.close()

def save_message_to_db(message: types.Message, llm_result: SpamResult = None):
    """Сохранение сообщения в базу данных"""
    try:
        from database import execute_query
        execute_query('''
            INSERT INTO messages 
            (message_id, chat_id, user_id, username, text, created_at, llm_result)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (message_id) DO UPDATE SET
            llm_result = EXCLUDED.llm_result
        ''', (
            message.message_id,
            message.chat.id,
            message.from_user.id,
            message.from_user.username or '',
            message.text,
            datetime.now(),
            llm_result.value if llm_result else None
        ))
    except:
        # Fallback к SQLite
        conn = sqlite3.connect('antispam.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO messages 
            (message_id, chat_id, user_id, username, text, created_at, llm_result)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            message.message_id,
            message.chat.id,
            message.from_user.id,
            message.from_user.username or '',
            message.text,
            datetime.now(),
            llm_result.value if llm_result else None
        ))
        conn.commit()
        conn.close()

def save_message_to_db_direct(message_id: int, chat_id: int, user_id: int, username: str, text: str, llm_result: str):
    """Прямое сохранение сообщения в БД (для восстановления)"""
    try:
        from database import execute_query
        execute_query('''
            INSERT INTO messages 
            (message_id, chat_id, user_id, username, text, created_at, llm_result)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (message_id) DO UPDATE SET
            llm_result = EXCLUDED.llm_result
        ''', (message_id, chat_id, user_id, username, text, datetime.now(), llm_result))
    except:
        conn = sqlite3.connect('antispam.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO messages 
            (message_id, chat_id, user_id, username, text, created_at, llm_result)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (message_id, chat_id, user_id, username, text, datetime.now(), llm_result))
        conn.commit()
        conn.close()

def update_admin_decision(message_id: int, decision: str):
    """Обновление решения администратора"""
    try:
        from database import execute_query
        execute_query('''
            UPDATE messages 
            SET admin_decision = ?, admin_decided_at = ?
            WHERE message_id = ?
        ''', (decision, datetime.now(), message_id))
    except:
        conn = sqlite3.connect('antispam.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE messages 
            SET admin_decision = ?, admin_decided_at = ?
            WHERE message_id = ?
        ''', (decision, datetime.now(), message_id))
        conn.commit()
        conn.close()

def add_training_example(text: str, is_spam: bool, source: str):
    """Добавление примера для обучения"""
    try:
        from database import execute_query
        execute_query('''
            INSERT INTO training_examples (text, is_spam, source, created_at)
            VALUES (?, ?, ?, ?)
        ''', (text, is_spam, source, datetime.now()))
    except:
        conn = sqlite3.connect('antispam.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO training_examples (text, is_spam, source, created_at)
            VALUES (?, ?, ?, ?)
        ''', (text, is_spam, source, datetime.now()))
        conn.commit()
        conn.close()

def get_current_prompt():
    """Получить текущий активный промпт"""
    # КРИТИЧЕСКИ ВАЖНО: Всегда логируем откуда берем промпт
    logger.info("🔍 Запрос текущего промпта...")
    
    try:
        from database import execute_query
        result = execute_query("SELECT prompt_text, improvement_reason FROM current_prompt ORDER BY id DESC LIMIT 1", fetch='one')
        
        if result:
            prompt, reason = result
            logger.info(f"📖 ЗАГРУЖЕН ПРОМПТ ИЗ POSTGRESQL:")
            logger.info(f"   Причина: {reason}")
            logger.info(f"   Содержит пункты 1-5: {'1.' in prompt and '2.' in prompt and '3.' in prompt}")
            logger.info(f"   Содержит исключения: {'Исключения' in prompt}")
            logger.info(f"   Середина: {prompt[200:400]}...")
            return prompt
        else:
            logger.error("❌ ПРОМПТ НЕ НАЙДЕН В POSTGRESQL!")
            
    except Exception as e:
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА POSTGRESQL: {e}")
    
    # Fallback к SQLite только в крайнем случае
    logger.warning("⚠️ ПЕРЕХОД К SQLITE FALLBACK")
    try:
        conn = sqlite3.connect('antispam.db')
        cursor = conn.cursor()
        cursor.execute("SELECT prompt_text, improvement_reason FROM current_prompt ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        
        if result:
            prompt, reason = result
            logger.warning(f"⚠️ ЗАГРУЖЕН ПРОМПТ ИЗ SQLITE:")
            logger.warning(f"   Причина: {reason}")
            logger.warning(f"   Содержит пункты 1-5: {'1.' in prompt and '2.' in prompt and '3.' in prompt}")
            logger.warning(f"   Содержит исключения: {'Исключения' in prompt}")
            return prompt
        else:
            logger.error("❌ ПРОМПТ НЕ НАЙДЕН ДАЖЕ В SQLITE!")
            
    except Exception as e2:
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА SQLITE: {e2}")
    
    logger.error("❌ ИСПОЛЬЗУЮ БАЗОВЫЙ ПРОМПТ - ЭТО ПРОБЛЕМА!")
    return SPAM_CHECK_PROMPT

def save_new_prompt(prompt_text: str, reason: str):
    """Сохранить новый промпт (заменяет предыдущий)"""
    try:
        from database import execute_query
        
        # Удаляем ВСЕ старые промпты
        execute_query("DELETE FROM current_prompt")
        
        # Добавляем новый промпт (единственный)
        execute_query('''
            INSERT INTO current_prompt (prompt_text, updated_at, improvement_reason)
            VALUES (?, ?, ?)
        ''', (prompt_text, datetime.now(), reason))
        
        logger.info(f"✅ ПРОМПТ ЗАМЕНЕН В POSTGRESQL:")
        logger.info(f"   Причина: {reason}")
        logger.info(f"   Длина: {len(prompt_text)} символов")
        logger.info(f"   Содержит пункты: {'1.' in prompt_text and '2.' in prompt_text}")
        logger.info(f"   Содержит исключения: {'Исключения' in prompt_text}")
        
    except Exception as e:
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА СОХРАНЕНИЯ В POSTGRESQL: {e}")
        
        # Fallback к SQLite
        conn = sqlite3.connect('antispam.db')
        cursor = conn.cursor()
        
        # Удаляем все старые
        cursor.execute("DELETE FROM current_prompt")
        
        cursor.execute('''
            INSERT INTO current_prompt (prompt_text, updated_at, improvement_reason)
            VALUES (?, ?, ?)
        ''', (prompt_text, datetime.now(), reason))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Промпт заменен через SQLite fallback: {reason}")

def get_recent_mistakes(limit=10):
    """Получить недавние ошибки бота для улучшения промпта"""
    try:
        from database import execute_query
        mistakes = execute_query('''
            SELECT text, llm_result, admin_decision, created_at
            FROM messages 
            WHERE admin_decision IS NOT NULL 
            AND ((llm_result = 'НЕ_СПАМ' AND admin_decision = 'СПАМ') 
                 OR (llm_result IN ('СПАМ', 'ВОЗМОЖНО_СПАМ') AND admin_decision = 'НЕ_СПАМ'))
            ORDER BY admin_decided_at DESC 
            LIMIT ?
        ''', (limit,), fetch='all')
        return mistakes or []
    except:
        conn = sqlite3.connect('antispam.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT text, llm_result, admin_decision, created_at
            FROM messages 
            WHERE admin_decision IS NOT NULL 
            AND ((llm_result = 'НЕ_СПАМ' AND admin_decision = 'СПАМ') 
                 OR (llm_result IN ('СПАМ', 'ВОЗМОЖНО_СПАМ') AND admin_decision = 'НЕ_СПАМ'))
            ORDER BY admin_decided_at DESC 
            LIMIT ?
        ''', (limit,))
        mistakes = cursor.fetchall()
        conn.close()
        return mistakes

def parse_llm_response(response_text: str) -> SpamResult:
    """Парсинг ответа от LLM"""
    cleaned = re.sub(r'[^\w\s_]', '', response_text.strip().upper())
    
    # Добавляем обработку обрезанных ответов
    maybe_spam_keywords = [
        'ВОЗМОЖНО_СПАМ', 'ВОЗМОЖНО СПАМ', 'ВОЗМОЖНОСПАМ', 
        'MAYBE_SPAM', 'MAYBE SPAM', 'MAYBEСПАМ',
        'ВОЗМО', 'ВОЗМОЖ'  # Обрезанные варианты
    ]
    not_spam_keywords = [
        'НЕ_СПАМ', 'НЕ СПАМ', 'НЕСПАМ', 
        'NOT_SPAM', 'NOT SPAM', 'NOTSPAM',
        'НЕ_СП', 'НЕ_С'  # Обрезанные варианты
    ]
    spam_keywords = ['СПАМ', 'SPAM']
    
    # Проверяем точные совпадения сначала
    if cleaned in ['СПАМ', 'SPAM']:
        return SpamResult.SPAM
    elif cleaned in ['НЕ_СПАМ', 'НЕ СПАМ', 'НЕСПАМ', 'NOT_SPAM', 'NOT SPAM', 'NOTSPAM']:
        return SpamResult.NOT_SPAM
    elif cleaned in ['ВОЗМОЖНО_СПАМ', 'ВОЗМОЖНО СПАМ', 'ВОЗМОЖНОСПАМ', 'MAYBE_SPAM', 'MAYBE SPAM']:
        return SpamResult.MAYBE_SPAM
    
    # Проверяем частичные совпадения
    if any(keyword in cleaned for keyword in maybe_spam_keywords):
        return SpamResult.MAYBE_SPAM
    elif any(keyword in cleaned for keyword in not_spam_keywords):
        return SpamResult.NOT_SPAM
    elif any(keyword in cleaned for keyword in spam_keywords):
        return SpamResult.SPAM
    
    logger.warning(f"Не удалось распарсить ответ LLM: '{response_text}' (очищенный: '{cleaned}')")
    return SpamResult.MAYBE_SPAM

async def improve_prompt_with_ai(mistakes):
    """Улучшение промпта с помощью ChatGPT на основе ошибок"""
    current_prompt = get_current_prompt()
    
    mistakes_text = ""
    for text, bot_decision, admin_decision, created_at in mistakes:
        mistakes_text += f"Сообщение: '{text}'\nБот решил: {bot_decision}\nПравильно: {admin_decision}\n\n"
    
    improvement_prompt = f"""
Ты эксперт по созданию промптов для определения спама в Telegram.

ТЕКУЩИЙ ПРОМПТ:
{current_prompt}

ОШИБКИ БОТА (последние):
{mistakes_text}

Проанализируй ошибки и улучши промпт, чтобы бот лучше определял спам. 
Сохрани структуру (три варианта ответа), но добавь более точные критерии на основе ошибок.

ОТВЕТЬ ТОЛЬКО УЛУЧШЕННЫМ ПРОМПТОМ, БЕЗ ДОПОЛНИТЕЛЬНЫХ ОБЪЯСНЕНИЙ:
"""
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4",  # Используем GPT-4 для улучшения промптов
            messages=[{"role": "user", "content": improvement_prompt}],
            max_tokens=1000,
            temperature=0.3,
            timeout=30
        )
        
        improved_prompt = response.choices[0].message.content.strip()
        logger.info("Промпт улучшен через AI")
        return improved_prompt
        
    except Exception as e:
        logger.error(f"Ошибка улучшения промпта: {e}")
        return None

async def check_message_with_llm(message_text: str) -> SpamResult:
    """Проверка сообщения через LLM"""
    current_prompt = get_current_prompt()
    
    # КРИТИЧЕСКОЕ ЛОГИРОВАНИЕ: какой промпт используется для анализа
    logger.info(f"🎯 ИСПОЛЬЗУЕТСЯ ПРОМПТ ДЛЯ АНАЛИЗА:")
    logger.info(f"   Содержит пункты 1-5: {'1.' in current_prompt and '2.' in current_prompt}")
    logger.info(f"   Содержит исключения: {'Исключения' in current_prompt}")
    logger.info(f"   Длина: {len(current_prompt)} символов")
    
    prompt = current_prompt.format(message_text=message_text)
    
    logger.info(f"🤖 Отправляю в ChatGPT: '{message_text[:50]}...'")
    logger.debug(f"📝 Полный промпт: {prompt}")
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,  # Увеличиваем лимит токенов
            temperature=0,
            timeout=10
        )
        
        llm_answer = response.choices[0].message.content.strip()
        result = parse_llm_response(llm_answer)
        
        logger.info(f"🎯 ChatGPT ответил: '{llm_answer}' (длина: {len(llm_answer)}) → {result.value}")
        
        # Если ответ слишком короткий, это подозрительно
        if len(llm_answer) < 3:
            logger.warning(f"⚠️ Подозрительно короткий ответ от ChatGPT: '{llm_answer}'")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Ошибка LLM: {e}")
        return SpamResult.MAYBE_SPAM

async def send_suspicious_message_to_admin(message: types.Message, result: SpamResult):
    """Отправка подозрительного сообщения админу"""
    result_emoji = "🔴" if result == SpamResult.SPAM else "🟡"
    
    admin_text = f"""{result_emoji} <b>{result.value}</b>

<b>От:</b> {message.from_user.full_name} (@{message.from_user.username or 'нет username'})
<b>Группа:</b> {message.chat.title}
<b>Время:</b> {message.date.strftime('%H:%M:%S')}

<b>Сообщение:</b>
<code>{message.text}</code>"""

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🔴 СПАМ", callback_data=f"spam_{message.message_id}"),
            InlineKeyboardButton(text="🟢 НЕ СПАМ", callback_data=f"not_spam_{message.message_id}")
        ]
    ])
    
    try:
        logger.info(f"📤 Отправляю подозрительное сообщение админу {ADMIN_ID}")
        logger.info(f"🔘 Кнопки: spam_{message.message_id}, not_spam_{message.message_id}")
        
        sent_message = await bot.send_message(
            ADMIN_ID, 
            admin_text, 
            reply_markup=keyboard,
            parse_mode='HTML'
        )
        logger.info(f"✅ Сообщение отправлено админу (ID: {sent_message.message_id})")
        
    except Exception as e:
        logger.error(f"❌ Ошибка отправки админу: {e}")


async def analyze_bot_error(message_text: str, error_type: str):
    """Анализ ошибки бота через ChatGPT"""
    if not openai_client:
        logger.error("❌ OpenAI клиент не инициализирован")
        return None, None
    
    # Принудительно получаем актуальный промпт
    current_prompt = get_current_prompt()
    logger.info(f"🧠 Для анализа ошибки используется промпт с пунктами: {'1.' in current_prompt and '2.' in current_prompt}")
        
    logger.info(f"🧠 Анализирую ошибку типа '{error_type}' для сообщения: '{message_text[:50]}...'")
    logger.info(f"🔍 Текущий промпт содержит: {current_prompt[100:200]}...")
    
    if error_type == "missed_spam":
        analysis_prompt = f"""У тебя есть промпт, по которому ты определяешь спам в Telegram. Вот он:

{current_prompt}

Но это сообщение ты НЕ определил как спам, хотя это спам:
"{message_text}"

Почему ты не определил это как спам? 

ВАЖНО: НЕ создавай новые критерии с нуля! ДОПОЛНИ существующие критерии, сохранив ВСЕ предыдущие знания.

ОБЯЗАТЕЛЬНО сохрани в итоговом промпте:
- Все существующие пункты 1-6
- Все существующие исключения и уточнения
- Весь контекст про аббревиатуры и политические партии

ЗАДАЧА: Добавь к существующим критериям новое правило, которое поможет определять такие сообщения как СПАМ.

Ответь в формате:
АНАЛИЗ: [причина ошибки]
ИТОГОВЫЙ_ПРОМПТ: [полный промпт с ВСЕМИ старыми критериями + новыми дополнениями]"""

    elif error_type == "uncertain_spam":
        analysis_prompt = f"""У тебя есть промпт, по которому ты определяешь спам в Telegram. Вот он:

{current_prompt}

Это сообщение ты определил как ВОЗМОЖНО_СПАМ, но это точно СПАМ:
"{message_text}"

Почему ты был неуверен? 

ВАЖНО: НЕ создавай новые критерии с нуля! ДОПОЛНИ существующие критерии, сохранив ВСЕ предыдущие знания.

ОБЯЗАТЕЛЬНО сохрани в итоговом промпте:
- Все существующие пункты 1-6
- Все существующие исключения и уточнения
- Весь контекст про аббревиатуры и политические партии

ЗАДАЧА: Добавь к существующим критериям новое правило или уточнение, которое поможет определять такие сообщения как СПАМ.

Ответь в формате:
АНАЛИЗ: [почему был неуверен]
ИТОГОВЫЙ_ПРОМПТ: [полный промпт с ВСЕМИ старыми критериями + новыми дополнениями]"""

    else:  # false_positive
        analysis_prompt = f"""У тебя есть промпт, по которому ты определяешь спам в Telegram. Вот он:

{current_prompt}

Но это сообщение ты определил как спам, хотя это НЕ спам:
"{message_text}"

Почему ты определил это как спам?

ВАЖНО: НЕ создавай новые критерии с нуля! ДОПОЛНИ существующие критерии исключением или уточнением.

ОБЯЗАТЕЛЬНО сохрани в итоговом промпте:
- Все существующие пункты 1-6
- Все существующие исключения и уточнения
- Весь контекст про аббревиатуры и политические партии

ЗАДАЧА: Добавь к существующим критериям исключение или уточнение, которое поможет НЕ считать такие сообщения спамом.

Ответь в формате:
АНАЛИЗ: [причина ошибки]
ИТОГОВЫЙ_ПРОМПТ: [полный промпт с ВСЕМИ старыми критериями + новыми исключениями/уточнениями]"""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=1500,
            temperature=0.3,
            timeout=30
        )
        
        analysis = response.choices[0].message.content.strip()
        logger.info(f"🧠 ChatGPT проанализировал ошибку: {analysis[:100]}...")
        
        # Извлекаем готовый итоговый промпт
        if "ИТОГОВЫЙ_ПРОМПТ:" in analysis:
            improved_prompt = analysis.split("ИТОГОВЫЙ_ПРОМПТ:")[1].strip()
            
            # Проверяем критически важные элементы
            checks = [
                ("{message_text}" in improved_prompt, "шаблон {message_text}"),
                ("Проанализируй сообщение из телеграм-группы" in improved_prompt, "системное начало"),
                ("безадресные вакансии" in improved_prompt, "знания о вакансиях"),
                ("сердечки 💘/💝" in improved_prompt, "знания о сердечках"),
                ("аббревиатуры" in improved_prompt, "знания об аббревиатурах")
            ]
            
            missing_elements = []
            for check, description in checks:
                if not check:
                    missing_elements.append(description)
                    logger.warning(f"⚠️ ChatGPT потерял: {description}")
            
            if missing_elements:
                logger.error(f"❌ ChatGPT потерял важные элементы: {missing_elements}")
                logger.error("🔄 Пытаюсь исправить промпт...")
                
                # Принудительно добавляем потерянные элементы
                if "{message_text}" not in improved_prompt:
                    if "Сообщение:" not in improved_prompt:
                        improved_prompt += "\n\nСообщение: «{message_text}»\n\nОтвет:"
            else:
                logger.info("✅ ChatGPT сохранил все важные элементы")
            
            return analysis, improved_prompt
        
        return analysis, None
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа: {e}")
        return None, None

@dp.message(F.content_type == 'text', F.forward_from)
async def handle_forwarded_spam(message: types.Message):
    """Обработка пересланных сообщений как примеров спама (ошибки бота)"""
    if message.from_user.id != ADMIN_ID:
        return
    
    # Добавляем как пример спама
    add_training_example(message.text, True, 'FORWARDED_MISTAKE')
    
    await message.reply("🔄 Анализирую, почему бот пропустил этот спам...")
    
    # Анализируем ошибку через ChatGPT
    analysis, improved_prompt = await analyze_bot_error(message.text, "missed_spam")
    
    if improved_prompt:
        # Отправляем админу предложение по улучшению
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Применить", callback_data="apply_prompt"),
                InlineKeyboardButton(text="✏️ Редактировать", callback_data="edit_prompt"),
                InlineKeyboardButton(text="❌ Отклонить", callback_data="reject_prompt")
            ]
        ])
        
        prompt_message = f"""🤖 <b>Анализ ошибки и улучшенный промпт:</b>

{analysis}

<b>Пропущенное сообщение:</b> "{message.text}"

<code>{improved_prompt}</code>"""
        
        # Сохраняем предложенный промпт
        global pending_prompt
        pending_prompt = improved_prompt
        
        await bot.send_message(ADMIN_ID, prompt_message, reply_markup=keyboard, parse_mode='HTML')
    else:
        await message.reply("❌ Не удалось проанализировать ошибку автоматически")

# ВАЖНО: Обработчики команд должны быть ПЕРЕД общим обработчиком текста!

@dp.message(Command("start"))
async def start_command(message: types.Message):
    """Команда /start"""
    logger.info(f"Команда /start от пользователя {message.from_user.id}")
    
    start_text = """🤖 <b>Kill Yr Spammers</b> - умный антиспам-бот!

🎯 <b>Что я умею:</b>
• Анализирую каждое сообщение через ИИ
• Отправляю подозрительные сообщения админу
• Учусь на ваших решениях и улучшаюсь
• Работаю только в разрешенных группах

📋 <b>Команды:</b>
/help - показать все команды
/stats - статистика работы (админ)
/editprompt - редактировать промпт (админ)
/groups - список разрешенных групп (админ)

💡 <b>Для обучения:</b> пересылайте мне примеры спама"""
    
    await message.reply(start_text, parse_mode='HTML')

@dp.message(Command("help"))
async def help_command(message: types.Message):
    """Команда /help - показать все команды"""
    logger.info(f"Команда /help от пользователя {message.from_user.id}")
    
    help_text = """📚 <b>Справка по командам Kill Yr Spammers</b>

🔹 <b>Общие команды:</b>
/start - информация о боте
/help - эта справка

🔹 <b>Команды администратора:</b>
/stats - статистика работы бота
/editprompt - редактировать промпт для анализа
/groups - список разрешенных групп
/cancel - отменить редактирование промпта

🎯 <b>Как работает бот:</b>
1️⃣ Анализирует каждое сообщение в группе через ChatGPT
2️⃣ Подозрительные сообщения отправляет админу с кнопками
3️⃣ Учится на ваших решениях (СПАМ/НЕ СПАМ)
4️⃣ Автоматически улучшает свой промпт

💡 <b>Обучение бота:</b>
• Пересылайте примеры спама боту в личку
• Используйте кнопки под подозрительными сообщениями
• После каждой ошибки бот предложит улучшенный промпт

🔐 <b>Безопасность:</b>
• Работает только в разрешенных группах
• API защищен от несанкционированного использования"""
    
    await message.reply(help_text, parse_mode='HTML')

@dp.message(Command("stats"))
async def stats_command(message: types.Message):
    """Статистика работы бота"""
    logger.info(f"Команда /stats от пользователя {message.from_user.id}")
    if message.from_user.id != ADMIN_ID:
        await message.reply("❌ Команда только для администратора")
        return
        
    try:
        from database import execute_query
        total_messages = execute_query("SELECT COUNT(*) FROM messages", fetch='one')[0]
        spam_count = execute_query("SELECT COUNT(*) FROM messages WHERE llm_result = 'СПАМ'", fetch='one')[0]
        maybe_spam_count = execute_query("SELECT COUNT(*) FROM messages WHERE llm_result = 'ВОЗМОЖНО_СПАМ'", fetch='one')[0]
        reviewed_count = execute_query("SELECT COUNT(*) FROM messages WHERE admin_decision IS NOT NULL", fetch='one')[0]
        training_count = execute_query("SELECT COUNT(*) FROM training_examples", fetch='one')[0]
    except:
        conn = sqlite3.connect('antispam.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        total_messages = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages WHERE llm_result = 'СПАМ'")
        spam_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages WHERE llm_result = 'ВОЗМОЖНО_СПАМ'")
        maybe_spam_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages WHERE admin_decision IS NOT NULL")
        reviewed_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM training_examples")
        training_count = cursor.fetchone()[0]
        
        conn.close()
    
    stats_text = f"""📊 <b>Статистика антиспам-бота</b>

📝 Всего сообщений: {total_messages}
🔴 Определено как спам: {spam_count}
🟡 Возможно спам: {maybe_spam_count}
✅ Проверено админом: {reviewed_count}
🧠 Примеров для обучения: {training_count}"""
    
    await message.reply(stats_text, parse_mode='HTML')

@dp.message(Command("editprompt"))
async def edit_prompt_command(message: types.Message):
    """Команда для редактирования промпта"""
    logger.info(f"Команда /editprompt от пользователя {message.from_user.id}")
    
    if message.from_user.id != ADMIN_ID:
        await message.reply("❌ Команда только для администратора")
        return
    
    # Сохраняем состояние в БД вместо глобальной переменной
    from database import set_bot_state
    set_bot_state(ADMIN_ID, awaiting_prompt_edit=True)
    
    global awaiting_prompt_edit
    awaiting_prompt_edit = True
    logger.info(f"Установлен режим редактирования в БД и памяти")
    
    current_prompt = get_current_prompt()
    edit_message = f"""✏️ <b>Редактирование промпта</b>

<b>Текущий промпт:</b>
<code>{current_prompt}</code>

<b>Отправьте новый промпт.</b> Должен содержать:
• Три варианта ответа: СПАМ, НЕ_СПАМ, ВОЗМОЖНО_СПАМ
• Место для подстановки: {{message_text}}

Для отмены: /cancel"""
    
    await message.reply(edit_message, parse_mode='HTML')

@dp.message(Command("groups"))
async def show_allowed_groups(message: types.Message):
    """Показать список разрешенных групп"""
    if message.from_user.id != ADMIN_ID:
        await message.reply("❌ Команда только для администратора")
        return
    
    groups_text = "🔐 <b>Разрешенные группы:</b>\n\n"
    for group_id in ALLOWED_GROUP_IDS:
        groups_text += f"• ID: <code>{group_id}</code>\n"
    
    groups_text += f"\n<b>Всего групп:</b> {len(ALLOWED_GROUP_IDS)}"
    groups_text += "\n\n💡 Только эти группы могут использовать API OpenAI"
    
    await message.reply(groups_text, parse_mode='HTML')

@dp.message(Command("version"))
async def show_prompt_version(message: types.Message):
    """Показать текущую версию промпта"""
    if message.from_user.id != ADMIN_ID:
        await message.reply("❌ Команда только для администратора")
        return
    
    # Проверяем PostgreSQL
    try:
        from database import execute_query
        result = execute_query("SELECT version, improvement_reason, created_at, substr(prompt_text, 1, 200) FROM prompts WHERE is_active = TRUE", fetch='one')
        if result:
            version, reason, created_at, prompt_preview = result
            version_info = f"🗄️ <b>PostgreSQL (основная БД):</b>\n📝 Версия: {version}\n🔄 Изменение: {reason}\n📅 Дата: {created_at}\n\n<code>{prompt_preview}...</code>"
        else:
            version_info = "🗄️ <b>PostgreSQL:</b> Промпт не найден"
    except Exception as e:
        version_info = f"🗄️ <b>PostgreSQL:</b> Ошибка подключения - {e}"
    
    # Проверяем SQLite fallback
    try:
        conn = sqlite3.connect('antispam.db')
        cursor = conn.cursor()
        cursor.execute("SELECT version, improvement_reason, created_at, substr(prompt_text, 1, 200) FROM prompts WHERE is_active = TRUE")
        result = cursor.fetchone()
        conn.close()
        
        if result:
            version, reason, created_at, prompt_preview = result
            version_info += f"\n\n💾 <b>SQLite (fallback):</b>\n📝 Версия: {version}\n🔄 Изменение: {reason}\n📅 Дата: {created_at}\n\n<code>{prompt_preview}...</code>"
        else:
            version_info += "\n\n💾 <b>SQLite:</b> Промпт не найден"
    except Exception as e:
        version_info += f"\n\n💾 <b>SQLite:</b> Ошибка - {e}"
    
    await message.reply(version_info, parse_mode='HTML')

@dp.message(Command("logs"))
async def show_action_logs(message: types.Message):
    """Показать последние действия"""
    if message.from_user.id != ADMIN_ID:
        await message.reply("❌ Команда только для администратора")
        return
    
    try:
        from action_logger import get_recent_actions
        actions = get_recent_actions(10)  # Последние 10 действий
        
        if not actions:
            await message.reply("📝 Логи действий пусты")
            return
        
        logs_text = "📋 <b>Последние действия:</b>\n\n"
        
        for action in reversed(actions[-10:]):  # Показываем в обратном порядке (новые сверху)
            timestamp = action["timestamp"][:19].replace('T', ' ')
            action_type = action["action_type"]
            
            if action_type == "message_analysis":
                result = action.get("result", {})
                logs_text += f"🔍 <b>Анализ сообщения</b> ({timestamp})\n"
                logs_text += f"   Результат: {result.get('llm_result', 'N/A')}\n"
                logs_text += f"   Текст: {action['details'].get('text', '')[:50]}...\n\n"
                
            elif action_type == "button_click":
                logs_text += f"🔘 <b>Кнопка: {action['details'].get('button', 'N/A')}</b> ({timestamp})\n"
                logs_text += f"   Исходный результат: {action['details'].get('original_llm_result', 'N/A')}\n"
                logs_text += f"   Текст: {action['details'].get('text', '')[:50]}...\n\n"
                
            elif action_type == "prompt_improvement":
                result = action.get("result", {})
                logs_text += f"🧠 <b>Улучшение промпта</b> ({timestamp})\n"
                logs_text += f"   Тип ошибки: {action['details'].get('error_type', 'N/A')}\n"
                logs_text += f"   Успешно: {action['details'].get('prompt_improved', False)}\n\n"
                
            elif action_type.startswith("error_"):
                logs_text += f"❌ <b>Ошибка: {action_type}</b> ({timestamp})\n"
                logs_text += f"   Сообщение: {action.get('error', 'N/A')[:100]}...\n\n"
        
        # Разбиваем на части если слишком длинное
        if len(logs_text) > 4000:
            logs_text = logs_text[:4000] + "\n\n... (обрезано)"
        
        await message.reply(logs_text, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"❌ Ошибка показа логов: {e}")
        await message.reply(f"❌ Ошибка получения логов: {e}")

@dp.message(Command("cancel"))
async def cancel_command(message: types.Message):
    """Команда для отмены редактирования"""
    global awaiting_prompt_edit
    
    if message.from_user.id != ADMIN_ID:
        return
    
    if awaiting_prompt_edit:
        awaiting_prompt_edit = False
        await message.reply("❌ Редактирование промпта отменено")
    else:
        await message.reply("ℹ️ Нет активного редактирования")


@dp.message(F.text & (F.chat.type == "private"))
async def handle_admin_text(message: types.Message):
    """Обработка текстовых сообщений от админа в ЛИЧКЕ (только для редактирования промпта в режиме ожидания)"""
    global awaiting_prompt_edit, pending_prompt
    
    # Проверяем, что это админ
    if message.from_user.id != ADMIN_ID:
        return
    
    # Пропускаем команды - они должны обрабатываться другими хендлерами
    if message.text and message.text.startswith('/'):
        return
    
    logger.info(f"🔍 handle_admin_text вызван с сообщением: '{message.text[:50]}...'")
    
    # Загружаем состояние из БД
    from database import get_bot_state, set_bot_state
    db_awaiting_edit, db_pending_prompt = get_bot_state(ADMIN_ID)
    
    # Синхронизируем с глобальной переменной
    awaiting_prompt_edit = db_awaiting_edit
    pending_prompt = db_pending_prompt
    
    logger.info(f"handle_admin_text: состояние из БД awaiting_prompt_edit = {awaiting_prompt_edit}")
    
    # Обрабатываем ТОЛЬКО если находимся в режиме редактирования промпта
    if awaiting_prompt_edit:
        
        # Проверяем базовую структуру промпта
        if "{message_text}" not in message.text:
            await message.reply("❌ Промпт должен содержать {message_text} для подстановки сообщения")
            return
        
        required_words = ["СПАМ", "НЕ_СПАМ", "ВОЗМОЖНО_СПАМ"]
        if not all(word in message.text.upper() for word in required_words):
            await message.reply("❌ Промпт должен содержать все три варианта ответа: СПАМ, НЕ_СПАМ, ВОЗМОЖНО_СПАМ")
            return
        
        # Сохраняем новый промпт
        logger.info(f"💾 Сохраняю новый промпт от админа (длина: {len(message.text)} символов)")
        save_new_prompt(message.text, "Ручное редактирование администратором")
        
        # Сбрасываем состояние в БД
        set_bot_state(ADMIN_ID, awaiting_prompt_edit=False)
        awaiting_prompt_edit = False
        pending_prompt = None
        
        # Получаем информацию о новом промпте
        try:
            from database import execute_query
            result = execute_query("SELECT version, improvement_reason, created_at FROM prompts WHERE is_active = TRUE", fetch='one')
        except:
            conn = sqlite3.connect('antispam.db')
            cursor = conn.cursor()
            cursor.execute("SELECT version, improvement_reason, created_at FROM prompts WHERE is_active = TRUE")
            result = cursor.fetchone()
            conn.close()
        
        if result:
            version, reason, created_at = result
            new_prompt_info = f"✅ <b>Новый промпт сохранен и активирован!</b>\n\n📝 <b>Версия {version}</b>\n\n<code>{message.text}</code>\n\n<b>Изменение:</b> {reason}\n<b>Дата:</b> {created_at}"
        else:
            new_prompt_info = f"✅ <b>Новый промпт сохранен и активирован!</b>\n\n<code>{message.text}</code>"
        
        await message.reply(new_prompt_info, parse_mode='HTML')
    else:
        # Если не в режиме редактирования, обрабатываем как обычное сообщение
        # Передаем дальше в общий обработчик
        return

@dp.message(F.content_type == 'text')
async def handle_message(message: types.Message):
    """Основная обработка сообщений"""
    # Логируем все сообщения для отладки
    logger.info(f"🔍 ПОЛУЧЕНО СООБЩЕНИЕ: от {message.from_user.id} (@{message.from_user.username}) в чате '{message.chat.title}' (тип: {message.chat.type}, ID: {message.chat.id})")
    logger.info(f"📝 Текст: '{message.text[:100]}...'")
    
    # Пропускаем сообщения от бота
    if message.from_user.is_bot:
        logger.info("🤖 Пропускаем сообщение от бота")
        return
    
    # В личных чатах обрабатываем только пересланные сообщения от админа
    if message.chat.type == 'private':
        if message.from_user.id != ADMIN_ID:
            return  # Не админ - игнорируем
        if not message.forward_from and not message.forward_from_chat:
            return  # Админ, но НЕ пересланное сообщение - игнорируем
    
    # В группах проверяем белый список
    elif message.chat.type in ['group', 'supergroup']:
        if message.chat.id not in ALLOWED_GROUP_IDS:
            logger.warning(f"🚫 ГРУППА НЕ В БЕЛОМ СПИСКЕ: {message.chat.title} (ID: {message.chat.id}) - игнорируем сообщение")
            return
    
    # Пропускаем команды
    if message.text and message.text.startswith('/'):
        return
        
    logger.info(f"Проверяю сообщение от {message.from_user.username}: {message.text[:50]}...")
    
    # Проверяем через LLM
    spam_result = await check_message_with_llm(message.text)
    
    # Логируем анализ сообщения
    try:
        from action_logger import log_message_analysis
        log_message_analysis(
            message.message_id,
            message.text,
            {
                "user_id": message.from_user.id,
                "username": message.from_user.username,
                "chat_title": message.chat.title,
                "chat_id": message.chat.id
            },
            spam_result.value
        )
    except Exception as e:
        logger.error(f"❌ Ошибка логирования анализа: {e}")
    
    # Сохраняем в БД
    save_message_to_db(message, spam_result)
    
    # Дублируем в backup файл
    try:
        from backup_messages import backup_message
        backup_message({
            "message_id": message.message_id,
            "chat_id": message.chat.id,
            "user_id": message.from_user.id,
            "username": message.from_user.username or "",
            "text": message.text,
            "llm_result": spam_result.value
        })
    except Exception as e:
        logger.error(f"❌ Ошибка backup: {e}")
    
    # Если подозрительное - отправляем админу
    if spam_result in [SpamResult.SPAM, SpamResult.MAYBE_SPAM]:
        logger.info(f"🚨 Подозрительное сообщение ({spam_result.value}), отправляю админу...")
        await send_suspicious_message_to_admin(message, spam_result)
    else:
        logger.info(f"✅ Сообщение чистое ({spam_result.value}), не отправляю админу")

@dp.callback_query(F.data.startswith("spam_") | F.data.startswith("not_spam_"))
async def handle_admin_feedback(callback: types.CallbackQuery):
    """Обработка обратной связи от администратора"""
    logger.info(f"🔘 Нажата кнопка: {callback.data} от пользователя {callback.from_user.id}")
    
    if callback.from_user.id != ADMIN_ID:
        await callback.answer("❌ Только для администратора")
        logger.warning(f"⚠️ Неавторизованный доступ к кнопке от {callback.from_user.id}")
        return
    
    if callback.data.startswith("not_spam_"):
        action = "not_spam"
        message_id = int(callback.data.replace("not_spam_", ""))
    elif callback.data.startswith("spam_"):
        action = "spam"
        message_id = int(callback.data.replace("spam_", ""))
    else:
        await callback.answer("❌ Неизвестная команда")
        return
    
    logger.info(f"🔍 Обработка кнопки: action={action}, message_id={message_id}")
    
    # Получаем текст сообщения и результат LLM из БД
    try:
        from database import execute_query
        result = execute_query("SELECT text, llm_result FROM messages WHERE message_id = ?", (message_id,), fetch='one')
    except:
        conn = sqlite3.connect('antispam.db')
        cursor = conn.cursor()
        cursor.execute("SELECT text, llm_result FROM messages WHERE message_id = ?", (message_id,))
        result = cursor.fetchone()
        conn.close()
    
    if not result:
        logger.warning(f"⚠️ Сообщение {message_id} не найдено в БД")
        
        # Пытаемся восстановить сообщение из самого callback
        try:
            # Извлекаем текст из сообщения callback
            original_text = callback.message.text
            if "Сообщение:" in original_text:
                # Извлекаем текст между <code> тегами
                import re
                code_match = re.search(r'<code>(.*?)</code>', original_text, re.DOTALL)
                if code_match:
                    message_text = code_match.group(1).strip()
                    
                    # Определяем llm_result из эмодзи в сообщении
                    if "🔴" in original_text:
                        llm_result = "СПАМ"
                    elif "🟡" in original_text:
                        llm_result = "ВОЗМОЖНО_СПАМ"
                    else:
                        llm_result = "ВОЗМОЖНО_СПАМ"  # По умолчанию
                    
                    # Сохраняем восстановленное сообщение в БД
                    save_message_to_db_direct(message_id, 0, 0, "unknown", message_text, llm_result)
                    
                    logger.info(f"🔄 Восстановлено сообщение из callback: '{message_text[:50]}...'")
                    result = (message_text, llm_result)
                else:
                    await callback.answer("❌ Не удалось восстановить текст сообщения")
                    return
            else:
                await callback.answer("❌ Сообщение не найдено в базе данных")
                return
        except Exception as e:
            logger.error(f"❌ Ошибка восстановления сообщения: {e}")
            await callback.answer("❌ Ошибка обработки сообщения")
            return
    
    message_text, llm_result = result
    decision = "СПАМ" if action == "spam" else "НЕ_СПАМ"
    is_spam = (action == "spam")
    
    # Логируем нажатие кнопки
    try:
        from action_logger import log_button_click
        log_button_click(callback.from_user.id, action, message_id, message_text, llm_result)
    except Exception as e:
        logger.error(f"❌ Ошибка логирования кнопки: {e}")
    
    # Обновляем решение админа
    update_admin_decision(message_id, decision)
    
    # Добавляем в обучающие примеры
    add_training_example(message_text, is_spam, 'ADMIN_FEEDBACK')
    
    # Обновляем сообщение
    decision_emoji = "❌" if is_spam else "✅"
    new_text = f"{callback.message.text}\n\n{decision_emoji} <b>Решение: {decision}</b>"
    
    await callback.message.edit_text(new_text, parse_mode='HTML')
    
    # Проверяем, нужно ли обучение
    logger.info(f"🔍 Проверяю необходимость обучения: action={action}, llm_result={llm_result}")
    
    needs_learning = False
    error_type = None
    
    if action == "not_spam" and llm_result in ['СПАМ', 'ВОЗМОЖНО_СПАМ']:
        needs_learning = True
        error_type = "false_positive"
    elif action == "spam" and llm_result == 'НЕ_СПАМ':
        needs_learning = True
        error_type = "missed_spam"
    elif action == "spam" and llm_result == 'ВОЗМОЖНО_СПАМ':
        needs_learning = True
        error_type = "uncertain_spam"
    
    if needs_learning:
        logger.info(f"🚨 Запускаю обучение! Тип: {error_type}")
        await callback.answer(f"✅ Отмечено как {decision}. Улучшаю промпт...")
        
        logger.info(f"📊 Тип обучения: {error_type}")
        
        # Анализируем ошибку через ChatGPT
        try:
            analysis, improved_prompt = await analyze_bot_error(message_text, error_type)
            logger.info(f"🧠 Результат анализа: analysis={analysis is not None}, prompt={improved_prompt is not None}")
            
            # Логируем результат улучшения промпта
            from action_logger import log_prompt_improvement
            log_prompt_improvement(callback.from_user.id, error_type, message_text, analysis, improved_prompt)
            
        except Exception as e:
            logger.error(f"❌ Ошибка в analyze_bot_error: {e}")
            
            # Логируем ошибку
            from action_logger import log_error
            log_error("prompt_improvement", callback.from_user.id, str(e), {
                "error_type": error_type,
                "message_text": message_text[:100]
            })
            
            analysis, improved_prompt = None, None
        
        if improved_prompt:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="✅ Применить", callback_data="apply_prompt"),
                    InlineKeyboardButton(text="✏️ Редактировать", callback_data="edit_prompt"),
                    InlineKeyboardButton(text="❌ Отклонить", callback_data="reject_prompt")
                ]
            ])
            
            global pending_prompt
            pending_prompt = improved_prompt
            
            error_description = "ложно определил как спам" if error_type == "false_positive" else "пропустил спам"
            
            prompt_message = f"""🤖 <b>Анализ ошибки бота:</b>

<b>Ошибка:</b> Бот {error_description}
<b>Сообщение:</b> "{message_text}"

{analysis}

<code>{improved_prompt}</code>"""
            
            await bot.send_message(ADMIN_ID, prompt_message, reply_markup=keyboard, parse_mode='HTML')
            logger.info("✅ Анализ ошибки отправлен админу")
        else:
            logger.warning("⚠️ Не удалось получить улучшенный промпт")
            await bot.send_message(ADMIN_ID, f"❌ Не удалось проанализировать ошибку автоматически\n\nСообщение: '{message_text}'\nОшибка: {error_type}")
    else:
        logger.info(f"ℹ️ Не ошибка бота: action={action}, llm_result={llm_result}")
        await callback.answer(f"✅ Отмечено как {decision}")

@dp.callback_query(F.data.in_(["apply_prompt", "edit_prompt", "reject_prompt", "edit_current_prompt"]))
async def handle_prompt_management(callback: types.CallbackQuery):
    """Обработка управления промптами"""
    global pending_prompt, awaiting_prompt_edit
    
    if callback.from_user.id != ADMIN_ID:
        await callback.answer("❌ Только для администратора")
        return
    
    if callback.data == "apply_prompt":
        if pending_prompt:
            save_new_prompt(pending_prompt, "Автоматическое улучшение на основе ошибок")
            await callback.message.edit_text(
                f"{callback.message.text}\n\n✅ <b>Промпт применен!</b>",
                parse_mode='HTML'
            )
            await callback.answer("✅ Новый промпт активирован!")
            pending_prompt = None
        else:
            await callback.answer("❌ Нет предложенного промпта")
    
    elif callback.data == "edit_prompt" or callback.data == "edit_current_prompt":
        awaiting_prompt_edit = True
        
        if callback.data == "edit_current_prompt":
            # Показываем текущий промпт для редактирования
            current_prompt = get_current_prompt()
            edit_message = f"✏️ <b>Редактирование текущего промпта</b>\n\n<b>Текущий промпт:</b>\n<code>{current_prompt}</code>\n\n"
        else:
            edit_message = "✏️ <b>Редактирование предложенного промпта</b>\n\n"
        
        edit_message += """Отправьте новый текст промпта. Должен содержать:
• Три варианта ответа: СПАМ, НЕ_СПАМ, ВОЗМОЖНО_СПАМ
• Место для подстановки сообщения: {message_text}

Для отмены отправьте /cancel"""
        
        await callback.message.reply(edit_message, parse_mode='HTML')
        await callback.answer("✏️ Жду новый промпт")
    
    elif callback.data == "reject_prompt":
        pending_prompt = None
        await callback.message.edit_text(
            f"{callback.message.text}\n\n❌ <b>Промпт отклонен</b>",
            parse_mode='HTML'
        )
        await callback.answer("❌ Промпт отклонен")

async def main():
    """Запуск бота"""
    global openai_client, bot
    
    # Проверяем наличие необходимых переменных окружения
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN не найден в переменных окружения!")
        return
    
    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY не найден в переменных окружения!")
        return
    
    if ADMIN_ID == 0:
        logger.error("❌ ADMIN_ID не найден в переменных окружения!")
        return
    
    # Инициализация бота
    try:
        bot = Bot(token=BOT_TOKEN)
        logger.info("✅ Telegram бот инициализирован")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации Telegram бота: {e}")
        return
    
    # Инициализация OpenAI клиента
    try:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        logger.info("✅ OpenAI клиент инициализирован")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации OpenAI: {e}")
        return
    
    # Инициализация БД
    from database import init_database as db_init
    db_init()
    
    # Восстанавливаем сообщения из backup файла
    try:
        from backup_messages import restore_messages_from_backup
        restored_count = restore_messages_from_backup()
        if restored_count > 0:
            logger.info(f"🔄 Восстановлено {restored_count} сообщений из backup")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось восстановить backup: {e}")
    
    # Настройка меню команд
    commands = [
        BotCommand(command="start", description="🤖 Информация о боте"),
        BotCommand(command="help", description="📚 Справка по командам"),
        BotCommand(command="stats", description="📊 Статистика работы (админ)"),
        BotCommand(command="editprompt", description="✏️ Редактировать промпт (админ)"),
        BotCommand(command="groups", description="🔐 Список разрешенных групп (админ)"),
        BotCommand(command="version", description="📋 Версия промпта (админ)"),
        BotCommand(command="logs", description="📝 Логи действий (админ)"),
        BotCommand(command="cancel", description="❌ Отменить редактирование (админ)")
    ]
    
    try:
        await bot.set_my_commands(commands)
        logger.info("✅ Меню команд настроено")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось настроить меню команд: {e}")
    
    logger.info("🤖 Kill Yr Spammers запускается...")
    logger.info(f"👤 Администратор: {ADMIN_ID}")
    logger.info(f"🔐 Разрешенных групп: {len(ALLOWED_GROUP_IDS)}")
    
    # КРИТИЧЕСКАЯ ПРОВЕРКА: какой промпт активен при старте
    logger.info("🔍 ПРОВЕРКА ПРОМПТА ПРИ СТАРТЕ:")
    startup_prompt = get_current_prompt()
    logger.info(f"🎯 СТАРТОВЫЙ ПРОМПТ:")
    logger.info(f"   Содержит пункты 1-5: {'1.' in startup_prompt and '2.' in startup_prompt}")
    logger.info(f"   Содержит исключения: {'Исключения' in startup_prompt}")
    logger.info(f"   Это базовый промпт: {startup_prompt == SPAM_CHECK_PROMPT}")
    
    # Запуск polling
    await dp.start_polling(bot)

if __name__ == "__main__":
    # Предупреждение о локальном запуске
    if not os.getenv("RAILWAY_ENVIRONMENT"):
        print("⚠️  ВНИМАНИЕ: Локальный запуск может привести к конфликтам с Railway ботом!")
        print("🚀 Рекомендуется использовать только Railway для продакшена.")
        print("🛑 Для остановки нажмите Ctrl+C")
        print("=" * 60)
    
    asyncio.run(main())
