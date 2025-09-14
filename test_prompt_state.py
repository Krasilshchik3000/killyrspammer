#!/usr/bin/env python3
"""
Тест состояния редактирования промпта
"""
import sys
sys.path.insert(0, '.')

from database import init_database, set_bot_state, get_bot_state

def test_prompt_state():
    print("🧪 Тестирование состояния редактирования промпта")
    
    # Инициализируем БД
    init_database()
    
    admin_id = 869587
    
    # Тест 1: Сохранение состояния
    print("\n1️⃣ Тестирую сохранение состояния...")
    set_bot_state(admin_id, awaiting_prompt_edit=True, pending_prompt="test")
    
    # Тест 2: Загрузка состояния
    print("2️⃣ Тестирую загрузку состояния...")
    awaiting_edit, pending = get_bot_state(admin_id)
    
    if awaiting_edit:
        print("✅ Состояние сохранено и загружено правильно")
    else:
        print("❌ Состояние НЕ сохранилось!")
    
    # Тест 3: Сброс состояния
    print("3️⃣ Тестирую сброс состояния...")
    set_bot_state(admin_id, awaiting_prompt_edit=False)
    
    awaiting_edit, pending = get_bot_state(admin_id)
    if not awaiting_edit:
        print("✅ Состояние сброшено правильно")
    else:
        print("❌ Состояние НЕ сбросилось!")

if __name__ == "__main__":
    test_prompt_state()
