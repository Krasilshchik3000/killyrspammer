"""
Kill Yr Spammers — Telegram антиспам-бот с LLM-классификацией и автообучением.

Промпт улучшается автоматически:
1. Админ исправляет ошибку бота (кнопка или пересылка)
2. После N ошибок бот генерирует улучшенный промпт
3. Новый промпт проверяется на всех накопленных примерах
4. Применяется только если точность >= текущего, иначе откат
"""
import asyncio
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from enum import Enum
from functools import wraps

import html
import httpx
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, BotCommand
import json as json_module
from openai import AsyncOpenAI

from config import (
    BOT_TOKEN, OPENAI_API_KEY, ADMIN_ID, ALLOWED_GROUP_IDS,
    LLM_MODEL, LLM_IMPROVEMENT_MODEL, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, LLM_TIMEOUT, MAX_REQUESTS_PER_MINUTE,
    FEW_SHOT_EXAMPLES_COUNT, CAS_API_URL, TRUSTED_USER_MESSAGES,
    AUTO_IMPROVE_AFTER_ERRORS, MIN_VALIDATION_EXAMPLES, MAX_VALIDATION_EXAMPLES,
    MAX_IMPROVEMENT_ATTEMPTS,
)
import database as db
from text_normalize import normalize_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot: Bot = None
dp = Dispatcher()
openai_client: AsyncOpenAI = None

_user_request_times: dict[int, list[float]] = defaultdict(list)
_http_client: httpx.AsyncClient = None
# Блокировка чтобы не запускать два улучшения одновременно
_improvement_in_progress = False


def _token_limit_param(max_tokens: int) -> dict:
    """gpt-5+ требуют max_completion_tokens вместо max_tokens."""
    if LLM_MODEL.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def _token_limit_param_improvement(max_tokens: int) -> dict:
    if LLM_IMPROVEMENT_MODEL.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


class SpamResult(Enum):
    SPAM = "СПАМ"
    NOT_SPAM = "НЕ_СПАМ"
    MAYBE_SPAM = "ВОЗМОЖНО_СПАМ"


# ──────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────

def require_admin(func):
    @wraps(func)
    async def wrapper(message_or_callback, *args, **kwargs):
        user = getattr(message_or_callback, 'from_user', None)
        if not user or user.id != ADMIN_ID:
            if isinstance(message_or_callback, types.CallbackQuery):
                await message_or_callback.answer("❌ Только для администратора")
            else:
                await message_or_callback.reply("❌ Только для администратора")
            return
        return await func(message_or_callback, *args, **kwargs)
    return wrapper


def check_rate_limit(user_id: int) -> bool:
    now = time.time()
    _user_request_times[user_id] = [t for t in _user_request_times[user_id] if t > now - 60]
    if len(_user_request_times[user_id]) >= MAX_REQUESTS_PER_MINUTE:
        return False
    _user_request_times[user_id].append(now)
    return True


def parse_llm_response(response_text: str) -> SpamResult:
    """Парсит ответ LLM (свободный текст или JSON)."""
    text = response_text.strip()

    # Попробовать JSON (structured output)
    try:
        parsed = json_module.loads(text)
        result_key = parsed.get("result", "")
        if result_key in _STRUCTURED_MAP:
            return _STRUCTURED_MAP[result_key]
    except (json_module.JSONDecodeError, AttributeError):
        pass

    # Fallback: парсинг свободного текста
    cleaned = re.sub(r'[^\w\s_]', '', text.upper())
    if len(cleaned) < 3:
        return SpamResult.MAYBE_SPAM

    exact = {
        'СПАМ': SpamResult.SPAM, 'SPAM': SpamResult.SPAM,
        'НЕ_СПАМ': SpamResult.NOT_SPAM, 'НЕ СПАМ': SpamResult.NOT_SPAM, 'NOT_SPAM': SpamResult.NOT_SPAM,
        'ВОЗМОЖНО_СПАМ': SpamResult.MAYBE_SPAM, 'ВОЗМОЖНО СПАМ': SpamResult.MAYBE_SPAM, 'MAYBE_SPAM': SpamResult.MAYBE_SPAM,
    }
    if cleaned in exact:
        return exact[cleaned]
    if 'ВОЗМОЖНО' in cleaned or 'MAYBE' in cleaned:
        return SpamResult.MAYBE_SPAM
    if 'НЕ_СПАМ' in cleaned or 'НЕ СПАМ' in cleaned or 'NOT_SPAM' in cleaned:
        return SpamResult.NOT_SPAM
    if 'СПАМ' in cleaned or 'SPAM' in cleaned:
        return SpamResult.SPAM
    return SpamResult.MAYBE_SPAM


def build_few_shot_block() -> str:
    examples = db.get_few_shot_examples(FEW_SHOT_EXAMPLES_COUNT)
    if not examples:
        return ""
    lines = ["Примеры из прошлых решений администратора:"]
    for text, is_spam in examples:
        label = "СПАМ" if is_spam else "НЕ_СПАМ"
        lines.append(f"- «{text[:120].replace(chr(10), ' ')}» → {label}")
    lines.append("")
    return "\n".join(lines)


def safe_format_prompt(template: str, message_text: str, few_shot_block: str) -> str:
    safe_text = message_text.replace("{", "{{").replace("}", "}}")
    try:
        return template.format(message_text=safe_text, few_shot_block=few_shot_block)
    except KeyError:
        try:
            return template.format(few_shot_block=few_shot_block)
        except KeyError:
            result = template.replace("{few_shot_block}", few_shot_block)
            result = result.replace("{message_text}", safe_text)
            return result


def validate_prompt(prompt_text: str) -> list[str]:
    problems = []
    # Проверяем наличие категорий (в любом формате — русском или английском)
    has_spam = "SPAM" in prompt_text.upper() or "СПАМ" in prompt_text
    has_not_spam = "NOT_SPAM" in prompt_text or "НЕ_СПАМ" in prompt_text
    has_maybe = "MAYBE_SPAM" in prompt_text or "ВОЗМОЖНО_СПАМ" in prompt_text
    if not has_spam:
        problems.append("Нет SPAM/СПАМ")
    if not has_not_spam:
        problems.append("Нет NOT_SPAM/НЕ_СПАМ")
    if not has_maybe:
        problems.append("Нет MAYBE_SPAM/ВОЗМОЖНО_СПАМ")
    return problems


# ──────────────────────────────────────────────
# CAS (Combot Anti-Spam)
# ──────────────────────────────────────────────

async def check_cas_ban(user_id: int) -> bool:
    try:
        response = await _http_client.get(CAS_API_URL, params={"user_id": user_id}, timeout=5)
        data = response.json()
        return data.get("ok", False)
    except Exception:
        return False


async def check_user_profile(user_id: int) -> str:
    """Проверяет профиль пользователя на спам-сигналы через raw Bot API.

    Использует httpx напрямую (а не aiogram) чтобы получить поля
    personal_chat и другие, которые aiogram 3.4.1 не поддерживает.

    Возвращает описание подозрительного контента или пустую строку.
    """
    try:
        response = await _http_client.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getChat",
            params={"chat_id": user_id},
            timeout=5,
        )
        data = response.json()
        if not data.get("ok"):
            return ""
        chat = data["result"]
    except Exception:
        return ""

    signals = []
    profile_parts = []  # Для LLM-анализа

    # Bio
    bio = chat.get("bio", "")
    if bio:
        profile_parts.append(f"Bio: {bio}")

    # Привязанный личный канал (Bot API 7.2+)
    personal_chat = chat.get("personal_chat")
    if personal_chat:
        channel_title = personal_chat.get("title", "")
        profile_parts.append(f"Личный канал: {channel_title}")

        # Получаем описание канала
        try:
            ch_resp = await _http_client.get(
                f"https://api.telegram.org/bot{BOT_TOKEN}/getChat",
                params={"chat_id": personal_chat["id"]},
                timeout=5,
            )
            ch_data = ch_resp.json()
            if ch_data.get("ok"):
                ch_desc = ch_data["result"].get("description", "")
                if ch_desc:
                    profile_parts.append(f"Описание канала: {ch_desc}")
        except Exception:
            pass

    if not profile_parts:
        return ""

    # Быстрая keyword-проверка (дешёвая, без LLM)
    profile_text = " ".join(profile_parts).lower()
    spam_keywords = [
        'заработ', 'доход', 'прибыл', 'инвестиц', 'крипт', 'казино',
        'ставк', 'букмекер', 'прогноз', 'сигнал', 'трейдинг', 'trading',
        'crypto', 'forex', 'p2p', 'обмен валют', 'пассивный доход',
        'промоутер', 'набор', 'вакансия', 'работа есть',
        'vpn', 'proxy', 'прокси', 'обход блокиров', 'кошельк',
        'betting', 'bet', 'casino', 'earn', 'income', 'profit',
    ]
    for kw in spam_keywords:
        if kw in profile_text:
            return f"Профиль: {'; '.join(profile_parts[:3])}"

    # Если keywords не сработали, но есть личный канал — проверяем через LLM
    if personal_chat and len(profile_parts) >= 2:
        try:
            resp = await openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Ты проверяешь профили пользователей Telegram на спам. Ответь YES если профиль похож на спам/скам (реклама, букмекеры, крипта, мошенничество, продажа), иначе NO. Отвечай одним словом."},
                    {"role": "user", "content": "\n".join(profile_parts)},
                ],
                **_token_limit_param(10),
                temperature=0,
                timeout=10,
            )
            answer = resp.choices[0].message.content.strip().upper()
            if "YES" in answer:
                return f"Профиль (LLM): {'; '.join(profile_parts[:3])}"
        except Exception as e:
            logger.warning(f"Profile LLM check failed: {e}")

    return ""


async def _get_profile_data(user_id: int) -> dict:
    """Получить bio и канал пользователя через raw Bot API."""
    try:
        response = await _http_client.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getChat",
            params={"chat_id": user_id}, timeout=5,
        )
        data = response.json()
        if not data.get("ok"):
            return {}
        chat = data["result"]
        result = {"bio": chat.get("bio", "")}
        personal_chat = chat.get("personal_chat")
        if personal_chat:
            result["channel_title"] = personal_chat.get("title", "")
            try:
                ch_resp = await _http_client.get(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/getChat",
                    params={"chat_id": personal_chat["id"]}, timeout=5,
                )
                ch_data = ch_resp.json()
                if ch_data.get("ok"):
                    result["channel_desc"] = ch_data["result"].get("description", "")
            except Exception:
                pass
        return result
    except Exception:
        return {}


# ──────────────────────────────────────────────
# LLM: классификация (hardened)
# ──────────────────────────────────────────────

# Structured output schema — модель ФИЗИЧЕСКИ не может ответить ничего другого
CLASSIFICATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "spam_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "enum": ["SPAM", "NOT_SPAM", "MAYBE_SPAM"]
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation why this classification was chosen (1-2 sentences)"
                }
            },
            "required": ["result", "reasoning"],
            "additionalProperties": False,
        }
    }
}

# Маппинг structured output → SpamResult
_STRUCTURED_MAP = {
    "SPAM": SpamResult.SPAM,
    "NOT_SPAM": SpamResult.NOT_SPAM,
    "MAYBE_SPAM": SpamResult.MAYBE_SPAM,
}


async def classify_message(
    prompt_template: str,
    message_text: str,
    few_shot: str = "",
    user_msg_count: int = 0,
    is_cas_banned: bool = False,
) -> tuple[SpamResult, str]:
    """Классификация сообщения с защитой от prompt injection.

    Защита:
    1. Текст нормализуется (гомоглифы, zero-width, Zalgo)
    2. System prompt содержит инструкции классификации
    3. User prompt содержит только сообщение в XML-тегах (sandwich defense)
    4. Structured output (JSON enum) — модель не может ответить произвольным текстом
    """
    # Нормализация текста
    normalized = normalize_text(message_text)

    # System prompt: инструкции + few-shot (доверенный контекст)
    system_prompt = safe_format_prompt(prompt_template, "", few_shot)
    # Убираем пустое «Сообщение: «»» из system prompt
    system_prompt = system_prompt.replace("Сообщение: «»", "").strip()

    # Контекст пользователя
    context_parts = []
    if user_msg_count > 0:
        context_parts.append(f"user_messages_in_group: {user_msg_count}")
    if is_cas_banned:
        context_parts.append("cas_banned: true")
    context_xml = ""
    if context_parts:
        context_xml = "<context>\n" + "\n".join(context_parts) + "\n</context>\n"

    # User prompt: sandwich defense с XML-тегами
    user_prompt = (
        f"{context_xml}"
        f"<message>\n{normalized}\n</message>\n\n"
        f"Classify the message above. Respond with JSON."
    )

    response = await openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=CLASSIFICATION_SCHEMA,
        **_token_limit_param(LLM_MAX_TOKENS),
        temperature=LLM_TEMPERATURE,
        timeout=LLM_TIMEOUT,
    )
    raw = response.choices[0].message.content.strip()

    # Parse structured output
    reasoning = ""
    try:
        parsed = json_module.loads(raw)
        result_key = parsed.get("result", "MAYBE_SPAM")
        result = _STRUCTURED_MAP.get(result_key, SpamResult.MAYBE_SPAM)
        reasoning = parsed.get("reasoning", "")
    except (json_module.JSONDecodeError, AttributeError):
        # Fallback: parse as free text (для совместимости со старыми моделями)
        result = parse_llm_response(raw)

    logger.info(f"LLM raw: '{raw}' → {result.value}")
    return result, reasoning


async def classify_image(
    image_url: str,
    caption: str = "",
    user_msg_count: int = 0,
    is_cas_banned: bool = False,
) -> tuple[SpamResult, str]:
    """Классификация изображения через Vision API."""
    system_prompt = (
        "Ты антиспам-классификатор для Telegram-групп.\n"
        "Тебе отправлено изображение из чата. Проанализируй текст на картинке и содержимое.\n"
        "Классифицируй как SPAM, NOT_SPAM или MAYBE_SPAM.\n\n"
        "SPAM: реклама товаров/услуг, продажа наркотиков, казино, криптоспам, "
        "мошенничество, фишинг, ссылки на подозрительные сайты.\n"
        "NOT_SPAM: мемы, фотографии, скриншоты бесед, обычный контент.\n"
        "MAYBE_SPAM: неясно, нужна проверка админом."
    )

    context_parts = []
    if user_msg_count > 0:
        context_parts.append(f"user_messages_in_group: {user_msg_count}")
    if is_cas_banned:
        context_parts.append("cas_banned: true")
    context_info = ", ".join(context_parts) if context_parts else "new user"

    user_content = [
        {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
    ]
    text_part = f"Context: {context_info}."
    if caption:
        text_part += f"\nCaption: {normalize_text(caption)}"
    text_part += "\nClassify this image. Respond with JSON."
    user_content.append({"type": "text", "text": text_part})

    response = await openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format=CLASSIFICATION_SCHEMA,
        **_token_limit_param(LLM_MAX_TOKENS),
        temperature=LLM_TEMPERATURE,
        timeout=LLM_TIMEOUT,
    )
    raw = response.choices[0].message.content.strip()

    reasoning = ""
    try:
        parsed = json_module.loads(raw)
        result_key = parsed.get("result", "MAYBE_SPAM")
        result = _STRUCTURED_MAP.get(result_key, SpamResult.MAYBE_SPAM)
        reasoning = parsed.get("reasoning", "")
    except (json_module.JSONDecodeError, AttributeError):
        result = parse_llm_response(raw)

    logger.info(f"Vision LLM raw: '{raw}' → {result.value}")
    return result, reasoning


async def check_message_with_llm(
    message_text: str,
    user_id: int = None,
    user_msg_count: int = 0,
    is_cas_banned: bool = False,
    photo_url: str = None,
    profile_signal: str = "",
) -> tuple[SpamResult, str]:
    if user_id and not check_rate_limit(user_id):
        return SpamResult.MAYBE_SPAM, "Rate limit exceeded"

    try:
        # Если есть фото — используем Vision API
        if photo_url:
            result, reasoning = await classify_image(photo_url, message_text or "", user_msg_count, is_cas_banned)
            logger.info(f"Vision → {result.value} (caption_len={len(message_text or '')}, msgs={user_msg_count})")
            return result, reasoning

        # Если профиль подозрительный — добавляем в контекст для LLM
        effective_text = message_text or ""
        if profile_signal:
            effective_text += f"\n\n[PROFILE CONTEXT: {profile_signal}]"

        # Текстовая классификация
        prompt_template = db.get_current_prompt()
        few_shot = build_few_shot_block()
        result, reasoning = await classify_message(prompt_template, effective_text, few_shot, user_msg_count, is_cas_banned)
        logger.info(f"LLM → {result.value} (len={len(message_text or '')}, msgs={user_msg_count}, cas={is_cas_banned}, profile={'yes' if profile_signal else 'no'})")

        # Если профиль спамный, но LLM сказал НЕ_СПАМ → повышаем до MAYBE_SPAM
        if profile_signal and result == SpamResult.NOT_SPAM:
            result = SpamResult.MAYBE_SPAM
            reasoning = f"Сообщение безобидное, но профиль подозрительный: {profile_signal}"

        return result, reasoning
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return SpamResult.MAYBE_SPAM, f"Error: {e}"


# ──────────────────────────────────────────────
# Автоматическое улучшение промпта
# ──────────────────────────────────────────────

async def evaluate_prompt(prompt_template: str, examples: list) -> tuple[float, int, int, list]:
    """Оценить промпт на примерах. Возвращает (accuracy, correct, total, errors).

    examples: [(text, is_spam), ...]
    errors: [(text, expected, got), ...] — конкретные ошибочные примеры
    """
    if not examples:
        return 0.0, 0, 0, []

    correct = 0
    total = len(examples)
    errors = []

    for text, is_spam in examples:
        try:
            result, _ = await classify_message(prompt_template, text)
            # СПАМ или ВОЗМОЖНО_СПАМ считаем за "спам" при is_spam=True
            predicted_spam = result in (SpamResult.SPAM, SpamResult.MAYBE_SPAM)
            actual_spam = bool(is_spam)
            if predicted_spam == actual_spam:
                correct += 1
            else:
                expected = "SPAM" if actual_spam else "NOT_SPAM"
                got = "SPAM" if predicted_spam else "NOT_SPAM"
                errors.append((text[:120], expected, got))
        except Exception as e:
            logger.warning(f"Ошибка валидации примера: {e}")
            total -= 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total, errors


async def generate_improved_prompt(
    error_type: str, message_text: str,
    failed_attempts: list[tuple[str, float | None]] | None = None,
    validation_errors: list[tuple[str, str, str]] | None = None,
) -> tuple[str, str] | tuple[None, None]:
    """Генерирует улучшенный промпт. Возвращает (analysis, improved_prompt) или (None, None).

    failed_attempts: список предыдущих неудачных попыток [(analysis, accuracy), ...]
    validation_errors: конкретные ошибки текущего промпта [(text, expected, got), ...]
    """
    current_prompt = db.get_current_prompt()

    descriptions = {
        "missed_spam": "Бот НЕ определил как спам, хотя это спам",
        "uncertain_spam": "Бот определил как ВОЗМОЖНО_СПАМ, но это точно спам",
        "false_positive": "Бот определил как спам/ВОЗМОЖНО_СПАМ, хотя это НЕ спам (ложное срабатывание)",
    }
    description = descriptions.get(error_type, error_type)

    recent_mistakes = db.get_recent_mistakes(5)
    mistakes_block = ""
    if recent_mistakes:
        lines = ["Другие недавние ошибки бота:"]
        for text, bot_dec, admin_dec, _ in recent_mistakes:
            lines.append(f"  - «{text[:80]}» — бот: {bot_dec}, правильно: {admin_dec}")
        mistakes_block = "\n".join(lines)

    # Контекст предыдущих неудачных попыток
    failed_block = ""
    if failed_attempts:
        lines = ["ПРЕДЫДУЩИЕ НЕУДАЧНЫЕ ПОПЫТКИ (не повторяй те же подходы!):"]
        for i, (reason, acc) in enumerate(failed_attempts):
            acc_str = f"{acc:.0%}" if acc is not None else "❌"
            lines.append(f"  Попытка {i+1} (точность: {acc_str}): {reason[:150]}")
        lines.append("")
        lines.append("Каждая новая попытка ДОЛЖНА использовать существенно другой подход.")
        failed_block = "\n".join(lines)

    # Блок конкретных ошибок текущего промпта на валидации
    validation_errors_block = ""
    if validation_errors:
        lines = ["КОНКРЕТНЫЕ ОШИБКИ ТЕКУЩЕГО ПРОМПТА НА ВАЛИДАЦИИ:"]
        for text, expected, got in validation_errors[:10]:
            lines.append(f"  - «{text}» — ожидалось: {expected}, бот сказал: {got}")
        validation_errors_block = "\n".join(lines)

    analysis_prompt = f"""Ты эксперт по созданию промптов для определения спама в Telegram-группах.

ТЕКУЩИЙ ПРОМПТ (используется для классификации сообщений):
---
{current_prompt}
---

ОШИБКА КЛАССИФИКАЦИИ: {description}
Проблемное сообщение: "{message_text}"

{validation_errors_block}

{mistakes_block}

{failed_block}

ЗАДАЧА: Улучши промпт так, чтобы он правильно обрабатывал это и похожие сообщения.

ПРАВИЛА:
1. Сохрани ВСЕ существующие критерии, исключения и структуру. Только дополни/уточни.
2. Промпт ОБЯЗАН содержать {{{{few_shot_block}}}} для вставки примеров.
3. Промпт ОБЯЗАН содержать три варианта: SPAM, NOT_SPAM, MAYBE_SPAM.
4. Промпт используется как system prompt. Сообщение пользователя передаётся отдельно в теге <message>.
5. Если ошибка — ложное срабатывание, добавь исключение чтобы подобные сообщения НЕ считались спамом.

Ответь СТРОГО в формате (два блока, разделённых маркером):

АНАЛИЗ: причина ошибки в 1-2 предложениях

ИТОГОВЫЙ_ПРОМПТ:
полный улучшенный промпт здесь"""

    try:
        response = await openai_client.chat.completions.create(
            model=LLM_IMPROVEMENT_MODEL,
            messages=[
                {"role": "system", "content": "Ты помощник по улучшению промптов. Всегда отвечай строго в указанном формате с маркерами АНАЛИЗ: и ИТОГОВЫЙ_ПРОМПТ:"},
                {"role": "user", "content": analysis_prompt},
            ],
            **_token_limit_param_improvement(16000),
            temperature=0.3,
            timeout=90,
        )
        text = (response.choices[0].message.content or "").strip()
        finish = response.choices[0].finish_reason
        logger.info(f"LLM improvement: len={len(text)}, finish={finish}, has_marker={'ИТОГОВЫЙ_ПРОМПТ:' in text}")
        if not text:
            logger.warning("LLM вернул пустой ответ")
            return None, None

        # Ищем маркер (с возможными вариациями форматирования)
        marker = None
        for m in ["ИТОГОВЫЙ_ПРОМПТ:", "ИТОГОВЫЙ ПРОМПТ:", "**ИТОГОВЫЙ_ПРОМПТ:**", "**ИТОГОВЫЙ_ПРОМПТ**:"]:
            if m in text:
                marker = m
                break

        if not marker:
            logger.warning(f"Маркер ИТОГОВЫЙ_ПРОМПТ не найден в ответе LLM (первые 200 символов): {text[:200]}")
            return text, None

        improved = text.split(marker, 1)[1].strip()

        # Убираем возможные markdown-обёртки
        if improved.startswith("```"):
            improved = improved.split("```", 2)[1]
            if improved.startswith("\n"):
                improved = improved[1:]
            if "```" in improved:
                improved = improved.rsplit("```", 1)[0]
            improved = improved.strip()

        # Патчим если потеряны обязательные элементы
        if "{message_text}" not in improved:
            improved += "\n\nСообщение: «{message_text}»\n\nОтвет:"
        if "{few_shot_block}" not in improved:
            improved = improved.replace("Сообщение: «{message_text}»", "{few_shot_block}\nСообщение: «{message_text}»")

        analysis = text.split(marker)[0].strip()
        # Убираем маркер АНАЛИЗ: из начала
        if analysis.startswith("АНАЛИЗ:"):
            analysis = analysis[7:].strip()
        elif analysis.startswith("**АНАЛИЗ:**"):
            analysis = analysis[11:].strip()

        return analysis, improved

    except Exception as e:
        logger.error(f"Ошибка генерации промпта: {e}", exc_info=True)
        return None, None


async def auto_improve_prompt(trigger_error_type: str, trigger_message: str):
    """Итеративное улучшение промпта с валидацией.

    Цикл до MAX_IMPROVEMENT_ATTEMPTS попыток:
    1. Генерирует улучшенный промпт (с контекстом предыдущих неудач)
    2. Оценивает текущий и новый на validation set
    3. Если лучше — применяет. Если нет — пробует снова с анализом неудачи.
    """
    global _improvement_in_progress
    if _improvement_in_progress:
        return
    _improvement_in_progress = True

    try:
        examples_count = db.count_training_examples()
        has_enough_for_validation = examples_count >= MIN_VALIDATION_EXAMPLES

        # Оцениваем текущий промпт один раз (для всех попыток)
        current_prompt = db.get_current_prompt()
        current_acc, current_ok, current_total, current_errors = 0.0, 0, 0, []
        validation_examples = []

        if has_enough_for_validation:
            validation_examples = db.get_validation_examples(MAX_VALIDATION_EXAMPLES)
            current_acc, current_ok, current_total, current_errors = await evaluate_prompt(current_prompt, validation_examples)
            logger.info(f"Текущая точность: {current_acc:.0%} ({current_ok}/{current_total}), ошибок: {len(current_errors)}")

        # Цикл попыток улучшения
        failed_attempts = []  # [(analysis, accuracy), ...]

        for attempt in range(1, MAX_IMPROVEMENT_ATTEMPTS + 1):
            logger.info(f"Попытка улучшения {attempt}/{MAX_IMPROVEMENT_ATTEMPTS}")

            # Генерируем с контекстом предыдущих неудач и конкретных ошибок
            analysis, improved = await generate_improved_prompt(
                trigger_error_type, trigger_message, failed_attempts, current_errors
            )
            if not improved:
                detail = f"\nАнализ LLM: {analysis[:300]}" if analysis else "\nLLM не вернул ответ"
                failed_attempts.append((detail, None))
                logger.warning(f"Попытка {attempt}: генерация не удалась")
                continue

            problems = validate_prompt(improved)
            if problems:
                failed_attempts.append((f"Невалидный промпт: {', '.join(problems)}", None))
                logger.warning(f"Попытка {attempt}: промпт невалиден: {problems}")
                continue

            if not has_enough_for_validation:
                # Мало примеров — применяем первый валидный промпт
                db.save_prompt_version(improved, f"Авто (без валидации, {examples_count} примеров): {trigger_error_type}")
                report = (
                    f"✅ <b>Промпт обновлён</b> (мало данных для валидации: {examples_count}/{MIN_VALIDATION_EXAMPLES})\n\n"
                    f"Причина: {html.escape(analysis or '')}\n\n"
                    f"Откатить: /rollback (из /history)"
                )
                await bot.send_message(ADMIN_ID, report, parse_mode='HTML')
                return

            # Валидация
            new_acc, new_ok, new_total, _ = await evaluate_prompt(improved, validation_examples)
            logger.info(f"Попытка {attempt}: {new_acc:.0%} ({new_ok}/{new_total}) vs текущий {current_acc:.0%}")

            if new_acc > current_acc:
                # Успех! Применяем
                db.save_prompt_version(improved, f"Авто: {trigger_error_type} ({new_acc:.0%} vs {current_acc:.0%}, попытка {attempt})")
                report = (
                    f"✅ <b>Промпт автоматически обновлён</b>\n\n"
                    f"Было: {current_acc:.0%} ({current_ok}/{current_total})\n"
                    f"Стало: {new_acc:.0%} ({new_ok}/{new_total})\n"
                    f"Попыток: {attempt}/{MAX_IMPROVEMENT_ATTEMPTS}\n\n"
                    f"Причина: {html.escape(analysis or '')}\n\n"
                    f"<code>{html.escape(improved[:500])}{'...' if len(improved) > 500 else ''}</code>\n\n"
                    f"Откатить: /rollback (из /history)"
                )
                await bot.send_message(ADMIN_ID, report, parse_mode='HTML')
                return

            # Не лучше — запоминаем и пробуем снова
            failed_attempts.append((analysis, new_acc))
            logger.info(f"Попытка {attempt}: не лучше ({new_acc:.0%} <= {current_acc:.0%}), продолжаем...")

        # Все попытки исчерпаны — короткий отчёт
        best_attempt_acc = max((acc for _, acc in failed_attempts if acc is not None), default=0)
        errors_summary = ""
        if current_errors:
            error_examples = [f"«{t[:60]}»" for t, _, _ in current_errors[:3]]
            errors_summary = f"\nОшибается на: {', '.join(error_examples)}"
        report = (
            f"🔄 Промпт не улучшен (5 попыток, лучшая: {best_attempt_acc:.0%} vs текущая: {current_acc:.0%}). "
            f"Few-shot примеры учтены.{errors_summary}"
        )
        await bot.send_message(ADMIN_ID, report)

    except Exception as e:
        logger.error(f"Ошибка автоулучшения: {e}")
        await bot.send_message(ADMIN_ID, f"⚠️ Ошибка автоулучшения промпта: {e}")
    finally:
        _improvement_in_progress = False


async def maybe_trigger_improvement(error_type: str, message_text: str):
    """Проверяет, пора ли запускать улучшение промпта."""
    errors_since = db.count_errors_since_last_improvement()
    logger.info(f"Ошибок с последнего улучшения: {errors_since}/{AUTO_IMPROVE_AFTER_ERRORS}")

    if errors_since >= AUTO_IMPROVE_AFTER_ERRORS:
        # Запускаем в фоне чтобы не блокировать ответ
        asyncio.create_task(auto_improve_prompt(error_type, message_text))


# ──────────────────────────────────────────────
# Telegram: проверки и действия
# ──────────────────────────────────────────────
# Полный аудит промпта + детектор спам-волн
# ──────────────────────────────────────────────

async def run_full_audit():
    """Полный аудит: анализ всех данных, поиск паттернов, улучшение промпта."""
    try:
        await bot.send_message(ADMIN_ID, "🔍 Начинаю полный аудит промпта...")

        # 1. Собираем все данные
        all_decisions = db.get_all_admin_decisions(500)
        all_examples = db.get_all_training_examples()
        current_prompt = db.get_current_prompt()
        banned_profiles = db.get_recent_banned_profiles(168)  # 7 дней

        # 2. Формируем статистику
        stats = {
            "total_decisions": len(all_decisions),
            "total_examples": len(all_examples),
            "banned_profiles": len(banned_profiles),
        }

        # Считаем ошибки по типам
        false_positives = []  # бот сказал спам, админ — нет
        missed_spam = []  # бот сказал не спам, админ — спам
        correct = 0
        for text, llm_result, admin_decision, reasoning, _ in all_decisions:
            llm_spam = llm_result in ('СПАМ', 'ВОЗМОЖНО_СПАМ')
            admin_spam = admin_decision == 'СПАМ'
            if llm_spam == admin_spam:
                correct += 1
            elif llm_spam and not admin_spam:
                false_positives.append((text[:100], reasoning or ''))
            elif not llm_spam and admin_spam:
                missed_spam.append((text[:100], reasoning or ''))

        accuracy = correct / len(all_decisions) if all_decisions else 0

        # 3. Анализ спам-волн
        wave_analysis = await detect_spam_waves(banned_profiles)

        # 4. Отправляем всё в LLM для глубокого анализа и генерации нового промпта
        fp_block = "\n".join(f"  - «{t}» (бот думал: {r[:60]})" for t, r in false_positives[:15])
        ms_block = "\n".join(f"  - «{t}» (бот думал: {r[:60]})" for t, r in missed_spam[:15])

        audit_prompt = f"""Ты эксперт по антиспам-системам в Telegram. Проведи полный аудит промпта.

ТЕКУЩИЙ ПРОМПТ:
---
{current_prompt}
---

СТАТИСТИКА:
- Всего решений админа: {stats['total_decisions']}
- Точность бота: {accuracy:.0%} ({correct}/{len(all_decisions)})
- Ложные срабатывания (бот думал спам, а это не спам): {len(false_positives)}
- Пропущенный спам (бот думал не спам, а это спам): {len(missed_spam)}
- Забаненных за 7 дней: {stats['banned_profiles']}

ЛОЖНЫЕ СРАБАТЫВАНИЯ:
{fp_block or '  (нет)'}

ПРОПУЩЕННЫЙ СПАМ:
{ms_block or '  (нет)'}

ОБНАРУЖЕННЫЕ ПАТТЕРНЫ СПАМ-ВОЛН:
{wave_analysis or '  (нет данных)'}

ЗАДАЧА:
1. Проанализируй паттерны ошибок — что общего у ложных срабатываний? Что общего у пропущенного спама?
2. Найди системные проблемы в промпте
3. Напиши ПОЛНОСТЬЮ НОВЫЙ промпт, который устраняет найденные проблемы
4. Промпт ОБЯЗАН содержать {{{{few_shot_block}}}} и три категории: SPAM, NOT_SPAM, MAYBE_SPAM
5. Промпт используется как system prompt, сообщение приходит в теге <message>

Ответь в формате:
АНАЛИЗ: подробный анализ (5-10 предложений)
ИТОГОВЫЙ_ПРОМПТ: полный новый промпт"""

        response = await openai_client.chat.completions.create(
            model=LLM_IMPROVEMENT_MODEL,
            messages=[
                {"role": "system", "content": "Ты эксперт по антиспам-системам. Проводишь полный аудит."},
                {"role": "user", "content": audit_prompt},
            ],
            **_token_limit_param_improvement(16000),
            temperature=0.3,
            timeout=120,
        )
        text = (response.choices[0].message.content or "").strip()

        # ── Часть 1: Отчёт о текущем состоянии ──
        fp_list = "\n".join(f"  • «{html.escape(t)}»" for t, _ in false_positives[:10]) or "  (нет)"
        ms_list = "\n".join(f"  • «{html.escape(t)}»" for t, _ in missed_spam[:10]) or "  (нет)"

        report1 = (
            f"🔍 <b>АУДИТ: Часть 1 — Текущее состояние</b>\n\n"
            f"📊 <b>Статистика:</b>\n"
            f"  Решений админа: {stats['total_decisions']}\n"
            f"  Training examples: {stats['total_examples']}\n"
            f"  Забанено за 7 дней: {stats['banned_profiles']}\n"
            f"  Точность бота: {accuracy:.0%} ({correct}/{len(all_decisions)})\n\n"
            f"❌ <b>Ложные срабатывания ({len(false_positives)}):</b>\n{fp_list}\n\n"
            f"⚠️ <b>Пропущенный спам ({len(missed_spam)}):</b>\n{ms_list}"
        )
        if wave_analysis:
            report1 += f"\n\n🌊 <b>Спам-волны:</b>\n{html.escape(wave_analysis[:400])}"

        await bot.send_message(ADMIN_ID, report1, parse_mode='HTML')

        # ── Генерация нового промпта ──
        # Парсим результат LLM
        if "ИТОГОВЫЙ_ПРОМПТ:" not in text:
            analysis = text[:800] if text else "LLM не вернул ответ"
            await bot.send_message(ADMIN_ID,
                f"⚠️ <b>Новый промпт не сгенерирован</b>\n\nАнализ:\n{html.escape(analysis[:800])}",
                parse_mode='HTML')
            return

        parts_split = text.split("ИТОГОВЫЙ_ПРОМПТ:", 1)
        analysis = parts_split[0].replace("АНАЛИЗ:", "").strip()
        new_prompt = parts_split[1].strip()

        # Валидация
        problems = validate_prompt(new_prompt)
        if problems:
            await bot.send_message(ADMIN_ID, f"⚠️ Аудит: новый промпт невалиден: {', '.join(problems)}")
            return

        # ── Часть 2: Анализ проблем ──
        report2 = (
            f"🔍 <b>АУДИТ: Часть 2 — Анализ</b>\n\n"
            f"{html.escape(analysis[:3800])}"
        )
        await bot.send_message(ADMIN_ID, report2, parse_mode='HTML')

        # ── Часть 3: Валидация нового промпта ──
        all_eval = [(t, s) for t, s, _, _ in all_examples]
        applied = False

        if len(all_eval) >= 10:
            eval_set = all_eval[:50]
            current_acc, current_ok, current_total, current_errors = await evaluate_prompt(current_prompt, eval_set)
            new_acc, new_ok, new_total, new_errors = await evaluate_prompt(new_prompt, eval_set)

            # Детальное сравнение: что исправлено, что сломалось
            # Прогоняем оба промпта на тех же примерах и сравниваем
            fixed = []  # ошибки старого, которые новый исправил
            broken = []  # правильные старого, которые новый сломал

            for err_text, err_expected, err_got in current_errors:
                # Эта ошибка есть у старого — проверяем, исправил ли новый
                if not any(e[0] == err_text for e in new_errors):
                    fixed.append(f"✅ «{err_text[:60]}» ({err_expected})")

            for err_text, err_expected, err_got in new_errors:
                # Эта ошибка есть у нового но не было у старого
                if not any(e[0] == err_text for e in current_errors):
                    broken.append(f"🔴 «{err_text[:60]}» ({err_expected}→{err_got})")

            fixed_block = "\n".join(fixed[:10]) or "  (ничего нового не исправлено)"
            broken_block = "\n".join(broken[:10]) or "  (ничего не сломано)"
            remaining_block = "\n".join(
                f"  • «{t[:60]}» ({e}→{g})" for t, e, g in new_errors[:10]
            ) or "  (нет ошибок)"

            verdict = "✅ ПРИМЕНЁН" if new_acc > current_acc else "❌ НЕ ПРИМЕНЁН (не лучше)"
            if new_acc > current_acc:
                db.save_prompt_version(new_prompt, f"Аудит: {new_acc:.0%} vs {current_acc:.0%}")
                applied = True

            report3 = (
                f"🔍 <b>АУДИТ: Часть 3 — Валидация</b>\n\n"
                f"📈 <b>Точность:</b> {current_acc:.0%} → {new_acc:.0%} ({current_ok}/{current_total} → {new_ok}/{new_total})\n"
                f"<b>Вердикт:</b> {verdict}\n\n"
                f"<b>Исправлено новым промптом:</b>\n{fixed_block}\n\n"
                f"<b>Новые ошибки (регрессии):</b>\n{broken_block}\n\n"
                f"<b>Оставшиеся ошибки:</b>\n{remaining_block}"
            )
            if applied:
                report3 += "\n\nОткатить: /rollback (из /history)"
        else:
            db.save_prompt_version(new_prompt, "Аудит (без валидации)")
            applied = True
            report3 = (
                f"🔍 <b>АУДИТ: Часть 3 — Валидация</b>\n\n"
                f"⚠️ Мало данных для валидации ({len(all_eval)} примеров). Промпт применён.\n"
                f"Откатить: /rollback (из /history)"
            )

        await bot.send_message(ADMIN_ID, report3, parse_mode='HTML')

        # ── Часть 4: Текущий промпт (полный текст) ──
        active_prompt = new_prompt if applied else current_prompt
        prompt_escaped = html.escape(active_prompt)
        # Разбиваем на чанки по 3800 символов (лимит Telegram 4096 - разметка)
        label = "📝 <b>АУДИТ: Часть 4 — Текущий промпт</b>\n\n"
        if len(prompt_escaped) <= 3700:
            await bot.send_message(ADMIN_ID, f"{label}<code>{prompt_escaped}</code>", parse_mode='HTML')
        else:
            chunks = [prompt_escaped[i:i+3700] for i in range(0, len(prompt_escaped), 3700)]
            for i, chunk in enumerate(chunks):
                header = label if i == 0 else f"📝 <b>Промпт (часть {i+1}):</b>\n\n"
                await bot.send_message(ADMIN_ID, f"{header}<code>{chunk}</code>", parse_mode='HTML')

    except Exception as e:
        logger.error(f"Ошибка аудита: {e}", exc_info=True)
        await bot.send_message(ADMIN_ID, f"⚠️ Ошибка аудита: {e}")


async def detect_spam_waves(profiles: list) -> str:
    """Анализирует профили забаненных для поиска общих паттернов (спам-волн)."""
    if len(profiles) < 3:
        return ""

    # Группируем по общим признакам
    bio_keywords = defaultdict(list)
    channel_keywords = defaultdict(list)
    message_patterns = defaultdict(list)

    for row in profiles:
        user_id, username, full_name, bio, ch_title, ch_desc, msg_text, reason, banned_at = row

        # Bio keywords
        if bio:
            for word in bio.lower().split():
                if len(word) > 3:
                    bio_keywords[word].append(username or str(user_id))

        # Channel keywords
        channel_text = f"{ch_title or ''} {ch_desc or ''}".strip()
        if channel_text:
            for word in channel_text.lower().split():
                if len(word) > 3:
                    channel_keywords[word].append(username or str(user_id))

        # Message patterns (first 50 chars as key)
        if msg_text:
            key = msg_text[:50].lower().strip()
            message_patterns[key].append(username or str(user_id))

    # Находим паттерны (слова, встречающиеся у 3+ забаненных)
    waves = []

    bio_waves = {k: v for k, v in bio_keywords.items() if len(v) >= 3}
    if bio_waves:
        top = sorted(bio_waves.items(), key=lambda x: -len(x[1]))[:5]
        waves.append("Bio: " + ", ".join(f"'{k}' ({len(v)} бан.)" for k, v in top))

    ch_waves = {k: v for k, v in channel_keywords.items() if len(v) >= 3}
    if ch_waves:
        top = sorted(ch_waves.items(), key=lambda x: -len(x[1]))[:5]
        waves.append("Каналы: " + ", ".join(f"'{k}' ({len(v)} бан.)" for k, v in top))

    msg_waves = {k: v for k, v in message_patterns.items() if len(v) >= 2}
    if msg_waves:
        top = sorted(msg_waves.items(), key=lambda x: -len(x[1]))[:3]
        waves.append("Сообщения: " + ", ".join(f"«{k[:40]}» ({len(v)} бан.)" for k, v in top))

    return "\n".join(waves) if waves else ""


async def _weekly_audit_loop():
    """Фоновый цикл: еженедельный аудит промпта."""
    while True:
        # Ждём 7 дней (604800 секунд)
        await asyncio.sleep(604800)
        try:
            logger.info("🔍 Запуск еженедельного аудита промпта")
            await run_full_audit()
        except Exception as e:
            logger.error(f"Ошибка еженедельного аудита: {e}")


# ──────────────────────────────────────────────
# Telegram: проверки и действия
# ──────────────────────────────────────────────

def should_skip_message(message: types.Message) -> bool:
    if message.from_user and message.from_user.is_bot:
        return True
    if message.from_user and message.from_user.id == ADMIN_ID:
        return True
    if message.sender_chat:
        return True
    if message.text and message.text.startswith('/'):
        return True
    return False


async def ban_user_in_all_groups(user_id: int, exclude_chat_id: int = None):
    banned, failed = [], []
    for gid in ALLOWED_GROUP_IDS:
        if gid == exclude_chat_id:
            continue
        try:
            await bot.ban_chat_member(chat_id=gid, user_id=user_id)
            banned.append(gid)
        except Exception as e:
            failed.append((gid, str(e)))
    return banned, failed


async def unban_user_in_all_groups(user_id: int):
    for gid in ALLOWED_GROUP_IDS:
        try:
            await bot.unban_chat_member(chat_id=gid, user_id=user_id)
        except Exception:
            pass


async def delete_user_messages(user_id: int) -> int:
    messages = db.get_user_messages(user_id)
    deleted = 0
    for msg_id, chat_id in messages:
        try:
            await bot.delete_message(chat_id, msg_id)
            deleted += 1
        except Exception:
            pass
    return deleted


async def send_to_admin(message: types.Message, result: SpamResult, reasoning: str = ""):
    emoji = "🔴" if result == SpamResult.SPAM else "🟡"
    reasoning_line = f"\n\n💭 <i>{html.escape(reasoning[:200])}</i>" if reasoning else ""
    text = (
        f"{emoji} <b>{result.value}</b>\n\n"
        f"<b>От:</b> {message.from_user.full_name} (@{message.from_user.username or 'n/a'})\n"
        f"<b>Группа:</b> {message.chat.title}\n"
        f"<b>Время:</b> {message.date.strftime('%H:%M:%S')}\n\n"
        f"<b>Сообщение:</b>\n<code>{html.escape(message.text or message.caption or '📷 [Фото без подписи]')}</code>"
        f"{reasoning_line}"
    )
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🔴 СПАМ", callback_data=f"spam_{message.message_id}"),
        InlineKeyboardButton(text="🟢 НЕ СПАМ", callback_data=f"not_spam_{message.message_id}"),
    ]])
    try:
        # Если есть фото — пересылаем его + текст кнопками
        if message.photo:
            await bot.send_photo(ADMIN_ID, message.photo[-1].file_id, caption=text, reply_markup=keyboard, parse_mode='HTML')
        else:
            await bot.send_message(ADMIN_ID, text, reply_markup=keyboard, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Ошибка отправки админу: {e}")


async def ban_and_report(message: types.Message, result: SpamResult, reasoning: str = ""):
    uid, cid = message.from_user.id, message.chat.id

    if message.sender_chat:
        await send_to_admin(message, result, reasoning)
        return
    if db.has_user_old_activity(uid, cid, 10):
        await send_to_admin(message, result, reasoning)
        return

    try:
        await bot.delete_message(cid, message.message_id)
        await bot.ban_chat_member(chat_id=cid, user_id=uid)
        banned, failed = await ban_user_in_all_groups(uid, exclude_chat_id=cid)
    except Exception as e:
        logger.error(f"Ошибка бана: {e}")
        await send_to_admin(message, result)
        return

    # Удаляем ВСЕ сообщения спамера из всех групп
    deleted = await delete_user_messages(uid)
    logger.info(f"Удалено {deleted} сообщений спамера {uid}")

    # Сохраняем профиль спамера для детектора спам-волн
    try:
        profile = await _get_profile_data(uid)
        db.save_banned_profile(
            uid, message.from_user.username or '', message.from_user.full_name,
            profile.get('bio', ''), profile.get('channel_title', ''),
            profile.get('channel_desc', ''),
            message.text or message.caption or '', reasoning[:200]
        )
    except Exception as e:
        logger.warning(f"Не удалось сохранить профиль спамера: {e}")

    text = (
        f"🔴 <b>АВТОБАН ЗА СПАМ</b>\n\n"
        f"<b>Забанен:</b> {message.from_user.full_name} (@{message.from_user.username or 'n/a'})\n"
        f"<b>User ID:</b> <code>{uid}</code>\n"
        f"<b>Группа:</b> {message.chat.title}\n\n"
        f"<b>Сообщение:</b>\n<code>{html.escape(message.text or message.caption or '📷 [Фото без подписи]')}</code>\n\n"
        f"✅ Забанен в {len(banned) + 1} группах\n"
        f"🗑 Удалено сообщений: {deleted}"
    )
    if reasoning:
        text += f"\n\n💭 <i>{html.escape(reasoning[:200])}</i>"
    if failed:
        text += f"\n⚠️ Не удалось в {len(failed)} группах"
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🟢 НЕ СПАМ (разбанить)", callback_data=f"unban_{uid}_{cid}_{message.message_id}")
    ]])
    try:
        if message.photo:
            await bot.send_photo(ADMIN_ID, message.photo[-1].file_id, caption=text, reply_markup=keyboard, parse_mode='HTML')
        else:
            await bot.send_message(ADMIN_ID, text, reply_markup=keyboard, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Ошибка отчёта: {e}")


# ──────────────────────────────────────────────
# Пересланные сообщения от админа = спам
# ──────────────────────────────────────────────

@dp.message(F.chat.type == "private", F.forward_date)
@require_admin
async def handle_forwarded_spam(message: types.Message):
    original_user_id = None
    original_username = None

    if message.forward_from:
        original_user_id = message.forward_from.id
        original_username = message.forward_from.username or message.forward_from.full_name
    elif message.forward_sender_name:
        original_username = message.forward_sender_name
    elif message.forward_from_chat:
        original_username = message.forward_from_chat.title

    spam_text = message.text or message.caption or ""
    if spam_text:
        db.add_training_example(spam_text, True, 'FORWARDED_SPAM')
        # Сохраняем как "ошибку бота" чтобы счётчик ошибок рос
        try:
            db.save_message(
                message.message_id, 0, original_user_id or 0,
                original_username or '', spam_text, 'НЕ_СПАМ', 'Пропущен ботом'
            )
            db.update_admin_decision(message.message_id, 'СПАМ')
        except Exception as e:
            logger.warning(f"Не удалось сохранить forwarded spam в messages: {e}")

    parts = [f"🔄 Обрабатываю спам от <b>{html.escape(original_username or 'неизвестного')}</b>"]

    if not original_user_id and spam_text:
        # Попробуем найти автора по тексту сообщения в БД
        found = db.find_user_by_message_text(spam_text)
        if found:
            original_user_id = found
            parts.append(f"🔍 Найден автор по тексту: <code>{original_user_id}</code>")

    if original_user_id:
        deleted = await delete_user_messages(original_user_id)
        banned, failed = await ban_user_in_all_groups(original_user_id)
        parts.append(f"🗑️ Удалено: {deleted} | 🔨 Забанен в {len(banned)} группах")
    else:
        parts.append("⚠️ User ID недоступен — бан невозможен")

    await message.reply("\n".join(parts), parse_mode='HTML')

    # Запускаем автоулучшение
    if spam_text:
        await maybe_trigger_improvement("missed_spam", spam_text)


# ──────────────────────────────────────────────
# Команды
# ──────────────────────────────────────────────

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.reply(
        "🤖 <b>Kill Yr Spammers</b>\n\n"
        "Анализирую сообщения через ИИ, учусь на ваших решениях.\n"
        "/help — все команды",
        parse_mode='HTML'
    )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.reply(
        "📚 <b>Команды</b>\n\n"
        "/stats — статистика\n"
        "/improve — принудительное улучшение промпта\n"
        "/audit — полный аудит промпта на всех данных\n"
        "/prompt — текущий промпт\n"
        "/history — история версий промпта\n"
        "/rollback N — откатить промпт к версии #N\n"
        "/editprompt — ручное редактирование промпта\n"
        "/resetprompt — сбросить промпт на дефолтный\n"
        "/groups — список групп\n"
        "/cancel — отменить редактирование\n\n"
        "💡 Пересылайте пропущенный спам боту\n"
        "Промпт улучшается автоматически после исправлений",
        parse_mode='HTML'
    )


@dp.message(Command("stats"))
@require_admin
async def cmd_stats(message: types.Message):
    total, spam, maybe, reviewed, training = db.get_stats()
    errors_since = db.count_errors_since_last_improvement()
    await message.reply(
        f"📊 <b>Статистика</b>\n\n"
        f"📝 Всего: {total} | 🔴 Спам: {spam} | 🟡 Возможно: {maybe}\n"
        f"✅ Проверено: {reviewed} | 🧠 Примеров: {training}\n"
        f"🔄 Ошибок до обновления промпта: {errors_since}/{AUTO_IMPROVE_AFTER_ERRORS}",
        parse_mode='HTML'
    )


@dp.message(Command("improve"))
@require_admin
async def cmd_improve(message: types.Message):
    """Принудительный запуск автоулучшения промпта."""
    await message.reply("🔄 Запускаю улучшение промпта...")
    asyncio.create_task(auto_improve_prompt("manual", "ручной запуск"))


@dp.message(Command("audit"))
@require_admin
async def cmd_audit(message: types.Message):
    """Полный аудит промпта на ВСЕХ данных."""
    await message.reply("🔍 Запускаю полный аудит...")
    asyncio.create_task(run_full_audit())


@dp.message(Command("prompt"))
@require_admin
async def cmd_prompt(message: types.Message):
    current = db.get_current_prompt()
    escaped = html.escape(current)
    # Разбиваем на чанки если не влезает
    if len(escaped) <= 3700:
        await message.reply(f"📝 <b>Текущий промпт:</b>\n\n<code>{escaped}</code>", parse_mode='HTML')
    else:
        chunks = [escaped[i:i+3700] for i in range(0, len(escaped), 3700)]
        for i, chunk in enumerate(chunks):
            header = "📝 <b>Текущий промпт:</b>\n\n" if i == 0 else f"📝 <b>Промпт (часть {i+1}/{len(chunks)}):</b>\n\n"
            await message.reply(f"{header}<code>{chunk}</code>", parse_mode='HTML')


@dp.message(Command("history"))
@require_admin
async def cmd_history(message: types.Message):
    rows = db.get_prompt_history(10)
    if not rows:
        await message.reply("📋 История пуста")
        return
    lines = ["📋 <b>История промптов:</b>\n"]
    for vid, reason, created in rows:
        lines.append(f"  #{vid} — {reason} ({created})")
    await message.reply("\n".join(lines), parse_mode='HTML')


@dp.message(Command("rollback"))
@require_admin
async def cmd_rollback(message: types.Message):
    parts = message.text.split()
    if len(parts) < 2 or not parts[1].isdigit():
        await message.reply("Использование: /rollback N (из /history)")
        return
    vid = int(parts[1])
    if db.rollback_prompt(vid):
        await message.reply(f"✅ Откат к версии #{vid}")
    else:
        await message.reply(f"❌ Версия #{vid} не найдена")


@dp.message(Command("editprompt"))
@require_admin
async def cmd_editprompt(message: types.Message):
    db.set_bot_state(ADMIN_ID, awaiting_prompt_edit=True)
    current = db.get_current_prompt()
    await message.reply(
        f"✏️ <b>Текущий промпт:</b>\n<code>{current}</code>\n\n"
        "Отправьте новый. Должен содержать {message_text}, СПАМ, НЕ_СПАМ, ВОЗМОЖНО_СПАМ.\n"
        "/cancel для отмены",
        parse_mode='HTML'
    )


@dp.message(Command("resetprompt"))
@require_admin
async def cmd_resetprompt(message: types.Message):
    db.save_prompt_version(db.DEFAULT_PROMPT, "Сброс на дефолтный промпт")
    await message.reply("✅ Промпт сброшен на дефолтный")


@dp.message(Command("groups"))
@require_admin
async def cmd_groups(message: types.Message):
    lines = [f"• <code>{gid}</code>" for gid in ALLOWED_GROUP_IDS]
    await message.reply(f"🔐 <b>Группы ({len(ALLOWED_GROUP_IDS)}):</b>\n" + "\n".join(lines), parse_mode='HTML')


@dp.message(Command("cancel"))
@require_admin
async def cmd_cancel(message: types.Message):
    db.set_bot_state(ADMIN_ID, awaiting_prompt_edit=False)
    await message.reply("❌ Отменено")


# ──────────────────────────────────────────────
# Приём нового промпта (ручное редактирование)
# ──────────────────────────────────────────────

@dp.message(F.text & (F.chat.type == "private"))
async def handle_admin_text(message: types.Message):
    if message.from_user.id != ADMIN_ID:
        return
    if message.text and message.text.startswith('/'):
        return

    awaiting, _ = db.get_bot_state(ADMIN_ID)
    if not awaiting:
        return

    problems = validate_prompt(message.text)
    if problems:
        await message.reply(f"❌ Невалиден: {', '.join(problems)}")
        return

    db.save_prompt_version(message.text, "Ручное редактирование")
    db.set_bot_state(ADMIN_ID, awaiting_prompt_edit=False)
    await message.reply("✅ Промпт сохранён")


# ──────────────────────────────────────────────
# Основной обработчик сообщений
# ──────────────────────────────────────────────

@dp.message(F.content_type.in_({'text', 'photo'}))
async def handle_message(message: types.Message):
    if message.chat.type == 'private':
        return
    if message.chat.type in ('group', 'supergroup') and message.chat.id not in ALLOWED_GROUP_IDS:
        return
    if should_skip_message(message):
        return

    uid, cid = message.from_user.id, message.chat.id
    username = message.from_user.username or message.from_user.full_name
    msg_text = message.text or message.caption or ""
    text_preview = msg_text[:80].replace('\n', ' ')
    has_photo = bool(message.photo)
    user_msg_count = db.count_user_messages(uid, cid)

    # Пользователь с историей сообщений — доверенный, не проверяем через LLM
    # ИСКЛЮЧЕНИЕ: пересланные сообщения всегда проверяются (VPN-спам паттерн)
    is_forward = bool(message.forward_date)
    if user_msg_count >= TRUSTED_USER_MESSAGES and not is_forward:
        logger.info(f"✅ TRUSTED @{username} (msgs={user_msg_count}) | {message.chat.title} | «{text_preview}»")
        try:
            db.save_message(message.message_id, cid, uid, message.from_user.username or '', msg_text, "НЕ_СПАМ")
        except Exception:
            pass
        return

    # Если нет ни текста, ни фото — пропускаем
    if not msg_text and not has_photo:
        return

    is_cas_banned = await check_cas_ban(uid)

    # Для новых пользователей — проверяем профиль (bio + личный канал)
    profile_spam_signal = ""
    if user_msg_count <= 2:
        profile_spam_signal = await check_user_profile(uid)
        if profile_spam_signal:
            logger.info(f"👤 Profile check @{username}: {profile_spam_signal[:100]}")

    # Пересланное сообщение — повышенная подозрительность
    if is_forward:
        forward_source = ""
        if message.forward_from_chat:
            forward_source = f"Переслано из канала «{message.forward_from_chat.title}»"
        elif message.forward_from:
            forward_source = f"Переслано от {message.forward_from.full_name}"
        elif message.forward_sender_name:
            forward_source = f"Переслано от {message.forward_sender_name}"
        if forward_source:
            profile_spam_signal = f"{profile_spam_signal}; {forward_source}" if profile_spam_signal else forward_source
            logger.info(f"📨 Forward from new user @{username}: {forward_source}")

    # CAS + нет истории → автобан
    if is_cas_banned and user_msg_count == 0:
        logger.info(f"🚫 CAS-BAN @{username} (cas=True, msgs=0) | {message.chat.title} | «{text_preview}»")
        try:
            db.save_message(message.message_id, cid, uid, message.from_user.username or '', msg_text, "СПАМ")
        except Exception:
            pass
        await ban_and_report(message, SpamResult.SPAM)
        return

    # Получаем URL фото если есть
    photo_url = None
    if has_photo:
        try:
            # Берём самое большое фото (последнее в массиве)
            photo = message.photo[-1]
            file_info = await bot.get_file(photo.file_id)
            photo_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}"
            logger.info(f"📷 Photo from @{username} | {message.chat.title} | caption: «{text_preview}»")
        except Exception as e:
            logger.error(f"Ошибка получения фото: {e}")

    result, reasoning = await check_message_with_llm(msg_text, uid, user_msg_count, is_cas_banned, photo_url, profile_spam_signal)
    emoji = {"СПАМ": "🔴", "ВОЗМОЖНО_СПАМ": "🟡", "НЕ_СПАМ": "🟢"}[result.value]
    source = "Vision" if photo_url else "LLM"
    logger.info(f"{emoji} {source}→{result.value} @{username} (msgs={user_msg_count}, cas={is_cas_banned}) | {message.chat.title} | «{text_preview}» | reason: {reasoning[:100]}")

    try:
        db.save_message(message.message_id, cid, uid, message.from_user.username or '', msg_text, result.value, reasoning)
    except Exception as e:
        logger.error(f"Ошибка сохранения: {e}")

    if result == SpamResult.SPAM:
        await ban_and_report(message, result, reasoning)
    elif result == SpamResult.MAYBE_SPAM:
        await send_to_admin(message, result, reasoning)


# ──────────────────────────────────────────────
# Callback: фидбек (СПАМ / НЕ СПАМ) → автообучение
# ──────────────────────────────────────────────

@dp.callback_query(F.data.startswith("spam_") | F.data.startswith("not_spam_"))
@require_admin
async def handle_admin_feedback(callback: types.CallbackQuery):
    try:
        if callback.data.startswith("not_spam_"):
            action, msg_id = "not_spam", int(callback.data[9:])
        else:
            action, msg_id = "spam", int(callback.data[5:])
        if msg_id <= 0:
            raise ValueError
    except (ValueError, TypeError):
        await callback.answer("❌ Некорректные данные")
        return

    row = db.get_message_by_id(msg_id)
    if not row:
        await callback.answer("❌ Не найдено в БД")
        return

    message_text, llm_result, user_id, chat_id, reasoning = row
    decision = "СПАМ" if action == "spam" else "НЕ_СПАМ"
    is_spam = action == "spam"

    db.update_admin_decision(msg_id, decision)
    db.add_training_example(message_text, is_spam, 'ADMIN_FEEDBACK')

    ban_info = ""
    if action == "spam" and user_id:
        try:
            try:
                await bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
            except Exception:
                pass
            banned, _ = await ban_user_in_all_groups(user_id, exclude_chat_id=chat_id)
            deleted = await delete_user_messages(user_id)
            ban_info = f"\n🔨 Забанен в {len(banned) + 1} группах, удалено {deleted} сообщений"
        except Exception as e:
            logger.error(f"Ошибка бана: {e}")
            ban_info = "\n⚠️ Ошибка бана"

    emoji = "❌" if is_spam else "✅"
    reasoning_line = f"\n💭 Бот думал: {html.escape(reasoning[:150])}" if reasoning else ""
    new_text = f"{callback.message.text}\n\n{emoji} <b>Решение: {decision}</b>{ban_info}{reasoning_line}"
    try:
        await callback.message.edit_text(new_text, parse_mode='HTML')
    except Exception:
        pass

    # Определяем тип ошибки и запускаем автоулучшение
    error_type = None
    if action == "not_spam" and llm_result in ('СПАМ', 'ВОЗМОЖНО_СПАМ'):
        error_type = "false_positive"
    elif action == "spam" and llm_result == 'НЕ_СПАМ':
        error_type = "missed_spam"
    elif action == "spam" and llm_result == 'ВОЗМОЖНО_СПАМ':
        error_type = "uncertain_spam"

    if error_type:
        await callback.answer(f"✅ {decision}. Обучаюсь...")
        await maybe_trigger_improvement(error_type, message_text)
    else:
        await callback.answer(f"✅ {decision}")


# ──────────────────────────────────────────────
# Callback: разбан
# ──────────────────────────────────────────────

@dp.callback_query(F.data.startswith("unban_"))
@require_admin
async def handle_unban(callback: types.CallbackQuery):
    try:
        parts = callback.data.split("_")
        if len(parts) != 4:
            raise ValueError
        user_id, chat_id, orig_msg_id = int(parts[1]), int(parts[2]), int(parts[3])
    except (ValueError, IndexError):
        await callback.answer("❌ Некорректные данные")
        return

    try:
        await unban_user_in_all_groups(user_id)
        await callback.answer("✅ Разбанен")

        new_text = f"{callback.message.text}\n\n🟢 <b>РАЗБАНЕН</b>"
        await callback.message.edit_text(new_text, parse_mode='HTML')

        row = db.get_message_by_id(orig_msg_id)
        if row:
            db.add_training_example(row[0], False, 'UNBAN_CORRECTION')
            await maybe_trigger_improvement("false_positive", row[0])

    except Exception as e:
        logger.error(f"Ошибка разбана: {e}")
        await callback.answer(f"❌ Ошибка: {e}")


# ──────────────────────────────────────────────
# Запуск
# ──────────────────────────────────────────────

async def main():
    global openai_client, bot, _http_client

    if not BOT_TOKEN:
        logger.error("BOT_TOKEN не задан")
        return
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY не задан")
        return

    bot = Bot(token=BOT_TOKEN)
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    _http_client = httpx.AsyncClient()

    db.init_database()

    commands = [
        BotCommand(command="start", description="Информация о боте"),
        BotCommand(command="help", description="Справка"),
        BotCommand(command="stats", description="Статистика (админ)"),
        BotCommand(command="improve", description="Улучшить промпт (админ)"),
        BotCommand(command="audit", description="Полный аудит промпта (админ)"),
        BotCommand(command="prompt", description="Текущий промпт (админ)"),
        BotCommand(command="history", description="История промптов (админ)"),
        BotCommand(command="rollback", description="Откат промпта (админ)"),
        BotCommand(command="editprompt", description="Редактировать промпт (админ)"),
        BotCommand(command="resetprompt", description="Сбросить промпт (админ)"),
        BotCommand(command="groups", description="Список групп (админ)"),
        BotCommand(command="cancel", description="Отменить"),
    ]
    try:
        await bot.set_my_commands(commands)
    except Exception:
        pass

    logger.info(
        f"🤖 Kill Yr Spammers | admin={ADMIN_ID} | groups={len(ALLOWED_GROUP_IDS)} "
        f"| model={LLM_MODEL} | improve={LLM_IMPROVEMENT_MODEL} "
        f"| auto_improve_after={AUTO_IMPROVE_AFTER_ERRORS} errors"
    )
    # Запускаем еженедельный аудит в фоне
    asyncio.create_task(_weekly_audit_loop())
    logger.info("📅 Еженедельный аудит запланирован")

    try:
        await dp.start_polling(bot)
    finally:
        await _http_client.aclose()


if __name__ == "__main__":
    if not os.getenv("RAILWAY_ENVIRONMENT"):
        print("⚠️  Локальный запуск. Ctrl+C для остановки.")
    asyncio.run(main())
