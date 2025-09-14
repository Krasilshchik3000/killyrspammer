#!/usr/bin/env python3
"""
Полный тест редактирования промпта
"""
import sys
sys.path.insert(0, '.')

from database import init_database, set_bot_state, get_bot_state
from main import get_current_prompt, save_new_prompt

def test_full_edit():
    print("🧪 Полный тест редактирования промпта")
    
    # Инициализируем БД
    init_database()
    
    admin_id = 869587
    
    print("\n1️⃣ Получаю текущий промпт...")
    current_prompt = get_current_prompt()
    print(f"Текущий промпт: {current_prompt[:100]}...")
    
    print("\n2️⃣ Устанавливаю режим редактирования...")
    set_bot_state(admin_id, awaiting_prompt_edit=True)
    
    print("\n3️⃣ Проверяю состояние...")
    awaiting_edit, pending = get_bot_state(admin_id)
    print(f"awaiting_prompt_edit = {awaiting_edit}")
    
    if awaiting_edit:
        print("\n4️⃣ Сохраняю новый промпт...")
        new_prompt = """ТЕСТОВЫЙ ПРОМПТ:
СПАМ
НЕ_СПАМ
ВОЗМОЖНО_СПАМ

Тестовые критерии с намеком вместо намёк.

Сообщение: «{message_text}»

Ответ:"""
        
        save_new_prompt(new_prompt, "Тестовое редактирование")
        
        print("\n5️⃣ Сбрасываю состояние...")
        set_bot_state(admin_id, awaiting_prompt_edit=False)
        
        print("\n6️⃣ Проверяю новый промпт...")
        updated_prompt = get_current_prompt()
        
        if "ТЕСТОВЫЙ ПРОМПТ" in updated_prompt:
            print("✅ Редактирование промпта работает!")
        else:
            print("❌ Промпт НЕ изменился!")
            print(f"Получен: {updated_prompt[:100]}...")
    else:
        print("❌ Состояние не установилось!")

if __name__ == "__main__":
    test_full_edit()
