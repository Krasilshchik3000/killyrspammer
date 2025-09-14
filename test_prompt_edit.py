#!/usr/bin/env python3
"""
Тест редактирования промпта
"""
import sqlite3
from datetime import datetime

def test_prompt_functions():
    # Подключаемся к локальной БД
    conn = sqlite3.connect('antispam.db')
    cursor = conn.cursor()
    
    print("🧪 Тестирование функций промпта")
    
    # Проверяем текущий промпт
    cursor.execute("SELECT version, prompt_text, is_active FROM prompts ORDER BY version DESC")
    prompts = cursor.fetchall()
    
    print(f"\n📋 Всего промптов в БД: {len(prompts)}")
    for version, text, is_active in prompts:
        status = "✅ АКТИВНЫЙ" if is_active else "❌ неактивный"
        print(f"   Версия {version}: {status} - {text[:50]}...")
    
    # Тестируем сохранение нового промпта
    print(f"\n💾 Тестирую сохранение нового промпта...")
    
    # Деактивируем старые
    cursor.execute("UPDATE prompts SET is_active = FALSE")
    
    # Получаем следующую версию
    cursor.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM prompts")
    next_version = cursor.fetchone()[0]
    
    # Добавляем тестовый промпт
    test_prompt = """ТЕСТОВЫЙ ПРОМПТ для проверки:
СПАМ
НЕ_СПАМ
ВОЗМОЖНО_СПАМ

Тестовые критерии спама.

Сообщение: «{message_text}»

Ответ:"""
    
    cursor.execute('''
        INSERT INTO prompts (prompt_text, version, created_at, is_active, improvement_reason)
        VALUES (?, ?, ?, TRUE, ?)
    ''', (test_prompt, next_version, datetime.now(), "Тестовое сохранение"))
    
    conn.commit()
    
    # Проверяем результат
    cursor.execute("SELECT version, is_active, improvement_reason FROM prompts WHERE version = ?", (next_version,))
    result = cursor.fetchone()
    
    if result:
        version, is_active, reason = result
        print(f"✅ Промпт сохранен: версия {version}, активный: {is_active}, причина: {reason}")
    else:
        print("❌ Промпт НЕ сохранен!")
    
    # Проверяем получение текущего промпта
    cursor.execute("SELECT prompt_text FROM prompts WHERE is_active = TRUE ORDER BY version DESC LIMIT 1")
    current = cursor.fetchone()
    
    if current and "ТЕСТОВЫЙ ПРОМПТ" in current[0]:
        print("✅ get_current_prompt() работает правильно")
    else:
        print("❌ get_current_prompt() НЕ работает!")
    
    conn.close()

if __name__ == "__main__":
    test_prompt_functions()
