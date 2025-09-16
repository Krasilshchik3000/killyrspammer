"""
Модуль детального логирования всех действий
"""
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

ACTION_LOG_FILE = "action_log.json"

def log_action(action_type, user_id, details, result=None, error=None):
    """Логирование действия пользователя"""
    try:
        # Загружаем существующие логи
        if os.path.exists(ACTION_LOG_FILE):
            with open(ACTION_LOG_FILE, 'r', encoding='utf-8') as f:
                action_logs = json.load(f)
        else:
            action_logs = {"actions": []}
        
        # Добавляем новое действие
        action_entry = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "user_id": user_id,
            "details": details,
            "result": result,
            "error": error
        }
        
        action_logs["actions"].append(action_entry)
        
        # Сохраняем обратно (оставляем только последние 1000 записей)
        if len(action_logs["actions"]) > 1000:
            action_logs["actions"] = action_logs["actions"][-1000:]
        
        with open(ACTION_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(action_logs, f, ensure_ascii=False, indent=2)
            
        logger.info(f"📝 Действие записано: {action_type} от {user_id}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка записи действия: {e}")

def log_message_analysis(message_id, message_text, chat_info, llm_result, prompt_version=None):
    """Логирование анализа сообщения"""
    log_action(
        action_type="message_analysis",
        user_id=chat_info.get("user_id"),
        details={
            "message_id": message_id,
            "text": message_text[:200],  # Первые 200 символов
            "chat_title": chat_info.get("chat_title"),
            "chat_id": chat_info.get("chat_id"),
            "username": chat_info.get("username")
        },
        result={
            "llm_result": llm_result,
            "prompt_version": prompt_version
        }
    )

def log_button_click(user_id, button_action, message_id, message_text, llm_result):
    """Логирование нажатия кнопки"""
    log_action(
        action_type="button_click",
        user_id=user_id,
        details={
            "button": button_action,
            "message_id": message_id,
            "text": message_text[:100],
            "original_llm_result": llm_result
        }
    )

def log_prompt_improvement(user_id, error_type, message_text, analysis_result, improved_prompt):
    """Логирование улучшения промпта"""
    log_action(
        action_type="prompt_improvement",
        user_id=user_id,
        details={
            "error_type": error_type,
            "message_text": message_text[:100],
            "analysis_success": analysis_result is not None,
            "prompt_improved": improved_prompt is not None
        },
        result={
            "analysis": analysis_result[:200] if analysis_result else None,
            "new_prompt_preview": improved_prompt[:200] if improved_prompt else None
        }
    )

def log_error(action_type, user_id, error_message, details=None):
    """Логирование ошибки"""
    log_action(
        action_type=f"error_{action_type}",
        user_id=user_id,
        details=details or {},
        error=error_message
    )

def get_recent_actions(limit=50):
    """Получить последние действия"""
    try:
        if os.path.exists(ACTION_LOG_FILE):
            with open(ACTION_LOG_FILE, 'r', encoding='utf-8') as f:
                action_logs = json.load(f)
            return action_logs["actions"][-limit:]
        return []
    except Exception as e:
        logger.error(f"❌ Ошибка чтения логов действий: {e}")
        return []
