"""
Модуль для резервного копирования сообщений
"""
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

BACKUP_FILE = "messages_backup.json"

def backup_message(message_data):
    """Сохранить сообщение в резервный файл"""
    try:
        # Загружаем существующие данные
        if os.path.exists(BACKUP_FILE):
            with open(BACKUP_FILE, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
        else:
            backup_data = {"messages": []}
        
        # Добавляем новое сообщение
        backup_data["messages"].append({
            "message_id": message_data["message_id"],
            "chat_id": message_data["chat_id"],
            "user_id": message_data["user_id"],
            "username": message_data.get("username", ""),
            "text": message_data["text"],
            "llm_result": message_data.get("llm_result"),
            "admin_decision": message_data.get("admin_decision"),
            "timestamp": datetime.now().isoformat()
        })
        
        # Сохраняем обратно
        with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"💾 Сообщение сохранено в backup: {message_data['message_id']}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка backup: {e}")

def restore_messages_from_backup():
    """Восстановить сообщения из backup файла"""
    try:
        if not os.path.exists(BACKUP_FILE):
            logger.info("📁 Backup файл не найден")
            return 0
        
        with open(BACKUP_FILE, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        from main import save_message_to_db_direct, update_admin_decision
        
        restored_count = 0
        for msg in backup_data.get("messages", []):
            try:
                save_message_to_db_direct(
                    msg["message_id"],
                    msg["chat_id"], 
                    msg["user_id"],
                    msg["username"],
                    msg["text"],
                    msg["llm_result"]
                )
                
                if msg.get("admin_decision"):
                    update_admin_decision(msg["message_id"], msg["admin_decision"])
                
                restored_count += 1
                
            except Exception as e:
                logger.error(f"❌ Ошибка восстановления сообщения {msg['message_id']}: {e}")
        
        logger.info(f"✅ Восстановлено {restored_count} сообщений из backup")
        return restored_count
        
    except Exception as e:
        logger.error(f"❌ Ошибка восстановления из backup: {e}")
        return 0
