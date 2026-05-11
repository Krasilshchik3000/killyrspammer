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
    LLM_MODEL_CANDIDATES, LLM_IMPROVEMENT_MODEL_CANDIDATES,
    LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TIMEOUT, MAX_REQUESTS_PER_MINUTE,
    FEW_SHOT_EXAMPLES_COUNT, CAS_API_URL, TRUSTED_USER_MESSAGES,
    AUTO_IMPROVE_AFTER_ERRORS, MIN_VALIDATION_EXAMPLES, MAX_VALIDATION_EXAMPLES,
    MAX_IMPROVEMENT_ATTEMPTS, REGRESSION_CHECK_SAMPLES, ORDINARY_MESSAGES_SAMPLES,
    MIN_ACCURACY_GAIN, MAX_REGRESSIONS,
)
from config import LLM_MODEL as _ENV_LLM_MODEL
from config import LLM_IMPROVEMENT_MODEL as _ENV_LLM_IMPROVEMENT_MODEL

# Реально используемые модели (определяются на старте через autodetect)
LLM_MODEL = _ENV_LLM_MODEL or LLM_MODEL_CANDIDATES[0]
LLM_IMPROVEMENT_MODEL = _ENV_LLM_IMPROVEMENT_MODEL or LLM_IMPROVEMENT_MODEL_CANDIDATES[0]
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


def _is_reasoning_model(model: str) -> bool:
    """Reasoning models (gpt-5+, o-series) имеют особые требования к параметрам."""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


async def _probe_model(model: str) -> tuple[bool, str]:
    """Проверяет доступность модели одним минимальным запросом.
    Возвращает (доступна, описание_ошибки_если_нет)."""
    try:
        params = {
            "model": model,
            "messages": [{"role": "user", "content": "ok"}],
            "timeout": 10,
        }
        if _is_reasoning_model(model):
            params["max_completion_tokens"] = 5
        else:
            params["max_tokens"] = 5
            params["temperature"] = 0
        await openai_client.chat.completions.create(**params)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


async def _autodetect_models() -> dict:
    """Подбирает доступные модели из списков-кандидатов.
    Возвращает {classification, improvement, errors: [...]} для отчёта."""
    global LLM_MODEL, LLM_IMPROVEMENT_MODEL

    result = {"classification": None, "improvement": None, "errors": []}

    # Если env переменная задана — используем её без проверки
    if _ENV_LLM_MODEL:
        result["classification"] = _ENV_LLM_MODEL
        logger.info(f"LLM_MODEL задан через env: {_ENV_LLM_MODEL}")
    else:
        for candidate in LLM_MODEL_CANDIDATES:
            ok, err = await _probe_model(candidate)
            if ok:
                LLM_MODEL = candidate
                result["classification"] = candidate
                logger.info(f"✅ Автодетект LLM_MODEL: {candidate}")
                break
            else:
                result["errors"].append(f"{candidate}: {err[:80]}")
                logger.warning(f"❌ {candidate} недоступна: {err[:80]}")

    if _ENV_LLM_IMPROVEMENT_MODEL:
        result["improvement"] = _ENV_LLM_IMPROVEMENT_MODEL
        logger.info(f"LLM_IMPROVEMENT_MODEL задан через env: {_ENV_LLM_IMPROVEMENT_MODEL}")
    else:
        for candidate in LLM_IMPROVEMENT_MODEL_CANDIDATES:
            ok, err = await _probe_model(candidate)
            if ok:
                LLM_IMPROVEMENT_MODEL = candidate
                result["improvement"] = candidate
                logger.info(f"✅ Автодетект LLM_IMPROVEMENT_MODEL: {candidate}")
                break
            else:
                result["errors"].append(f"{candidate}: {err[:80]}")
                logger.warning(f"❌ {candidate} недоступна: {err[:80]}")

    return result


def _token_limit_param(max_tokens: int) -> dict:
    """gpt-5+ требуют max_completion_tokens вместо max_tokens."""
    if _is_reasoning_model(LLM_MODEL):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def _token_limit_param_improvement(max_tokens: int) -> dict:
    if _is_reasoning_model(LLM_IMPROVEMENT_MODEL):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def _temperature_param(model: str, value: float) -> dict:
    """Reasoning models (gpt-5+) НЕ поддерживают temperature — пропускаем."""
    if _is_reasoning_model(model):
        return {}
    return {"temperature": value}


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
                **_temperature_param(LLM_MODEL, 0),
                timeout=10,
            )
            answer = resp.choices[0].message.content.strip().upper()
            if "YES" in answer:
                return f"Профиль (LLM): {'; '.join(profile_parts[:3])}"
        except Exception as e:
            logger.warning(f"Profile LLM check failed: {e}")

    return ""


def _classify_spam_type(text: str) -> str:
    """Определяет, можно ли распознать спам по тексту или только по контексту.

    'text' — текст содержит явные спам-признаки (ссылки, @username, предложения)
    'context' — текст выглядит невинно, спам определяется по профилю/контексту
    """
    if not text:
        return 'context'
    t = text.lower()
    # Явные текстовые спам-признаки
    text_spam_signals = [
        '@', 't.me/', 'http', 'подпис', 'канал', 'бот ', 'перейд',
        'заработ', 'доход', 'инвестиц', 'крипт', 'p2p', 'казино',
        'ставк', 'прогноз', 'сигнал', 'букмекер', 'промоутер',
        'набор', 'ищем', 'нужны люди', 'оплата', 'выплат',
        'пишите', 'в личку', 'лс', 'обращайтесь', 'пассивный доход',
        'vpn', 'proxy', 'оформля', 'бесплатно', 'скидк',
        'водитель', 'автомойк', 'разгрузк', 'промокод',
    ]
    for signal in text_spam_signals:
        if signal in t:
            return 'text'
    # Короткий невинный текст без спам-признаков = profile spam
    if len(text) < 80:
        return 'context'
    return 'text'


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
        **_temperature_param(LLM_MODEL, LLM_TEMPERATURE),
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
        **_temperature_param(LLM_MODEL, LLM_TEMPERATURE),
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
    """Оценить промпт на примерах параллельно (батчами по 10).

    Использует тот же few_shot_block, что и в проде — чтобы валидация
    отражала реальное поведение бота.

    examples: [(text, is_spam), ...]
    Возвращает (accuracy, correct, total, errors).
    """
    if not examples:
        return 0.0, 0, 0, []

    # ВАЖНО: используем те же few-shot примеры, что и в production
    few_shot = build_few_shot_block()

    async def classify_one(text: str, is_spam: bool):
        try:
            result, _ = await classify_message(prompt_template, text, few_shot=few_shot)
            predicted_spam = result in (SpamResult.SPAM, SpamResult.MAYBE_SPAM)
            actual_spam = bool(is_spam)
            return text, is_spam, predicted_spam, actual_spam, None
        except Exception as e:
            return text, is_spam, None, None, str(e)

    # Параллельная классификация батчами по 10 (rate limit safety)
    BATCH = 10
    results = []
    for i in range(0, len(examples), BATCH):
        batch = examples[i:i+BATCH]
        batch_results = await asyncio.gather(
            *[classify_one(text, is_spam) for text, is_spam in batch]
        )
        results.extend(batch_results)

    correct = 0
    total = 0
    errors = []
    eval_errors_sample = []  # для логгирования
    for text, is_spam, predicted_spam, actual_spam, err in results:
        if err is not None:
            if len(eval_errors_sample) < 3:
                eval_errors_sample.append(err)
            continue
        total += 1
        if predicted_spam == actual_spam:
            correct += 1
        else:
            expected = "SPAM" if actual_spam else "NOT_SPAM"
            got = "SPAM" if predicted_spam else "NOT_SPAM"
            errors.append((text[:120], expected, got))

    if eval_errors_sample:
        logger.error(f"evaluate_prompt: {len(examples) - total - len(errors)} примеров упали с ошибкой. Примеры: {eval_errors_sample}")

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total, errors


# 5 стратегий генерации — каждая попытка использует свою
IMPROVEMENT_STRATEGIES = [
    {
        "name": "обобщение паттернов",
        "instruction": (
            "Найди ОБЩИЕ паттерны в ошибках бота. Не вставляй конкретные тексты сообщений. "
            "Сформулируй универсальные правила-обобщения, описывающие НАМЕРЕНИЕ и СТРУКТУРУ "
            "спам-сообщений. Например, вместо «сообщения вроде „Купи курс“» пиши «короткие "
            "рекламные призывы с глаголами действия». Сохрани все существующие правила, добавь новые обобщения."
        ),
    },
    {
        "name": "упрощение и фокус на intent",
        "instruction": (
            "Текущий промпт перегружен. Сократи его, убери дубликаты и избыточные примеры. "
            "Сфокусируй модель на НАМЕРЕНИИ автора (продажа, вербовка, реклама) "
            "вместо ключевых слов. Сохрани ключевые категории спама и исключения для NOT_SPAM."
        ),
    },
    {
        "name": "реструктуризация",
        "instruction": (
            "Переструктурируй промпт: чёткие секции SPAM/NOT_SPAM/MAYBE_SPAM с критериями. "
            "Каждая категория — это описание ТИПА сообщения, а не список конкретных фраз. "
            "Добавь чёткую логику принятия решения (decision tree)."
        ),
    },
    {
        "name": "точечное расширение",
        "instruction": (
            "Сделай МИНИМАЛЬНОЕ изменение: добавь 1-2 правила, покрывающих новые типы ошибок. "
            "Не трогай остальное. Новые правила должны быть универсальными (тип/паттерн), "
            "а не дословными цитатами сообщений."
        ),
    },
    {
        "name": "полная перезапись с нуля",
        "instruction": (
            "Напиши промпт с нуля, учитывая все наблюдения. Структура: цель → категории с "
            "критериями → исключения → правило при сомнении → защита от prompt injection. "
            "Категории описывай через ТИПЫ спама (финансовый оффер, реклама канала, "
            "вербовка в личку, флирт-бот, рекламная картинка), а не примеры."
        ),
    },
]


def _contains_literal_messages(prompt_text: str, messages: list[str]) -> list[str]:
    """Возвращает список сообщений из messages, которые буквально (5+ символов подряд)
    содержатся в prompt_text. Используется для запрета дословного цитирования."""
    found = []
    p = prompt_text.lower()
    for msg in messages:
        if not msg:
            continue
        # Берём фрагменты 25-символьные из сообщения (без пробелов в начале)
        m = msg.strip().lower()
        if len(m) < 25:
            # Короткие — проверяем целиком
            if m in p and len(m) >= 8:
                found.append(msg[:60])
        else:
            # Длинные — проверяем фрагменты по 25 символов
            fragments = [m[i:i+25] for i in range(0, len(m) - 25, 15)]
            for frag in fragments:
                if frag in p:
                    found.append(msg[:60])
                    break
    return found


async def generate_improved_prompt_with_strategy(
    strategy: dict,
    current_prompt: str,
    trigger_message: str,
    error_type: str,
    validation_errors: list,
    failed_attempts: list,
    wave_analysis: str = "",
) -> tuple[str | None, str | None]:
    """Генерирует промпт с конкретной стратегией. Возвращает (analysis, prompt) или (None, None)."""

    descriptions = {
        "missed_spam": "Бот НЕ определил как спам, хотя это спам",
        "uncertain_spam": "Бот определил как ВОЗМОЖНО_СПАМ, но это точно спам",
        "false_positive": "Бот определил как спам, хотя это НЕ спам",
        "manual": "Ручной запуск улучшения",
        "weekly": "Еженедельное обучение на полной базе",
    }
    description = descriptions.get(error_type, error_type)

    # Конкретные ошибки на валидации (только text-spam)
    errors_block = ""
    if validation_errors:
        lines = ["ОШИБКИ ТЕКУЩЕГО ПРОМПТА (для понимания, не цитировать в промпте):"]
        for text, expected, got in validation_errors[:15]:
            lines.append(f"  - «{text[:120]}» — ожидалось: {expected}, бот: {got}")
        errors_block = "\n".join(lines)

    # Что не сработало в предыдущих попытках
    failed_block = ""
    if failed_attempts:
        lines = ["ПРЕДЫДУЩИЕ ПОПЫТКИ (не повторяй):"]
        for i, (strat_name, acc, reason) in enumerate(failed_attempts):
            acc_str = f"{acc:.0%}" if acc is not None else "❌"
            lines.append(f"  #{i+1} «{strat_name}» — {acc_str}: {reason[:120]}")
        failed_block = "\n".join(lines)

    waves_block = ""
    if wave_analysis:
        waves_block = f"АКТУАЛЬНЫЕ СПАМ-ВОЛНЫ (из недавних банов):\n{wave_analysis}"

    analysis_prompt = f"""Ты эксперт по антиспам-системам Telegram.

ТЕКУЩИЙ ПРОМПТ:
---
{current_prompt}
---

КОНТЕКСТ: {description}. Сообщение-триггер: «{trigger_message[:200]}»

{errors_block}

{waves_block}

{failed_block}

СТРАТЕГИЯ ЭТОЙ ПОПЫТКИ: {strategy['name']}
{strategy['instruction']}

ОБЯЗАТЕЛЬНЫЕ ТРЕБОВАНИЯ К НОВОМУ ПРОМПТУ:
- Содержит три категории: SPAM, NOT_SPAM, MAYBE_SPAM
- Содержит плейсхолдер {{{{few_shot_block}}}} для подстановки примеров
- Используется как system prompt; сообщение пользователя приходит отдельно в теге <message>
- Описывает ТИПЫ спама и КРИТЕРИИ их определения (это нужно и важно!)
- ЗАПРЕЩЕНО: вставлять дословные цитаты конкретных сообщений из ошибок выше
- ЗАПРЕЩЕНО: упоминать конкретные @username из реальных сообщений
- РАЗРЕШЕНО и нужно: описывать паттерны, структуру, намерение, эмодзи-стиль

Ответь СТРОГО в формате:

АНАЛИЗ: 2-3 предложения о том, что меняется в этой попытке

ИТОГОВЫЙ_ПРОМПТ:
<полный текст нового промпта>"""

    try:
        response = await openai_client.chat.completions.create(
            model=LLM_IMPROVEMENT_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Ты эксперт по промпт-инжинирингу для антиспам-систем. "
                    "Формат ответа: АНАЛИЗ: ... ИТОГОВЫЙ_ПРОМПТ: <текст>"
                )},
                {"role": "user", "content": analysis_prompt},
            ],
            **_token_limit_param_improvement(16000),
            **_temperature_param(LLM_IMPROVEMENT_MODEL, 0.5),
            timeout=180,
        )
        text = (response.choices[0].message.content or "").strip()
        finish = response.choices[0].finish_reason
        logger.info(f"LLM gen ({strategy['name']}): len={len(text)}, finish={finish}")
        if not text:
            return None, None

        # Парсим маркер
        marker = None
        for m in ["ИТОГОВЫЙ_ПРОМПТ:", "ИТОГОВЫЙ ПРОМПТ:", "**ИТОГОВЫЙ_ПРОМПТ:**", "**ИТОГОВЫЙ_ПРОМПТ**:"]:
            if m in text:
                marker = m
                break
        if not marker:
            return text, None

        improved = text.split(marker, 1)[1].strip()
        if improved.startswith("```"):
            improved = improved.split("```", 2)[1]
            if improved.startswith("\n"):
                improved = improved[1:]
            if "```" in improved:
                improved = improved.rsplit("```", 1)[0]
            improved = improved.strip()

        # Патчим обязательные плейсхолдеры
        if "{message_text}" not in improved and "<message>" not in improved:
            improved += "\n\nСообщение: «{message_text}»"
        if "{few_shot_block}" not in improved:
            improved += "\n\n{few_shot_block}"

        analysis = text.split(marker)[0].strip()
        if analysis.startswith("АНАЛИЗ:"):
            analysis = analysis[7:].strip()
        elif analysis.startswith("**АНАЛИЗ:**"):
            analysis = analysis[11:].strip()

        return analysis, improved

    except Exception as e:
        logger.error(f"Ошибка генерации ({strategy['name']}): {e}", exc_info=True)
        return f"Ошибка LLM: {str(e)[:200]}", None


async def _send_progress(text: str):
    """Отправляет прогресс-сообщение админу. Не падает при ошибке."""
    try:
        await bot.send_message(ADMIN_ID, text, parse_mode='HTML')
    except Exception:
        try:
            await bot.send_message(ADMIN_ID, text)  # без HTML
        except Exception as e:
            logger.warning(f"Не удалось отправить прогресс: {e}")


async def _send_full_prompt(prompt_text: str, label: str = "📝 <b>ПОЛНЫЙ ПРОМПТ</b>"):
    """Отправляет промпт целиком, разбивая на чанки по 3700 символов."""
    escaped = html.escape(prompt_text)
    if len(escaped) <= 3600:
        await _send_progress(f"{label}\n\n<code>{escaped}</code>")
        return
    chunks = [escaped[i:i+3600] for i in range(0, len(escaped), 3600)]
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        h = f"{label} (часть {i+1}/{total})" if total > 1 else label
        await _send_progress(f"{h}\n\n<code>{chunk}</code>")


async def auto_improve_prompt(trigger_error_type: str, trigger_message: str):
    """Многостратегическое автоулучшение промпта с прогресс-репортингом и валидацией на полной базе.

    Алгоритм:
    1. Собираем полный датасет: text-spam примеры + correctly-classified сообщения (детектор регрессий)
    2. Оцениваем текущий промпт
    3. Для каждой из 5 стратегий: генерируем → валидируем → сравниваем
    4. Выбираем лучшую попытку с учётом точности, регрессий, FP/FN
    5. Применяем, если выполнены критерии; иначе отчитываемся
    """
    global _improvement_in_progress
    if _improvement_in_progress:
        return
    _improvement_in_progress = True

    try:
        # ── Фаза 1: Сбор контекста ──
        await _send_progress(f"🔄 <b>Запуск автообучения</b>\nТриггер: {html.escape(trigger_error_type)}")

        current_prompt = db.get_current_prompt()
        spam_examples = db.get_validation_examples(MAX_VALIDATION_EXAMPLES)  # уже фильтр по text-spam
        correct_messages = db.get_correctly_classified_messages(REGRESSION_CHECK_SAMPLES)
        ordinary_messages = db.get_ordinary_messages(ORDINARY_MESSAGES_SAMPLES)

        # Формируем датасет: спам + correctly-classified + обычные сообщения
        # is_spam=False для обычных (они НЕ должны флагаться как спам)
        regression_set = []
        for text, llm_result in correct_messages:
            is_spam = llm_result in ('СПАМ', 'ВОЗМОЖНО_СПАМ')
            regression_set.append((text, is_spam))

        ordinary_set = [(text, False) for text, _ in ordinary_messages]

        total_eval = len(spam_examples) + len(regression_set) + len(ordinary_set)
        if total_eval < MIN_VALIDATION_EXAMPLES:
            await _send_progress(
                f"⚠️ Мало данных для валидации: {total_eval} (минимум {MIN_VALIDATION_EXAMPLES}). "
                f"Автообучение отложено."
            )
            return

        # Считаем разбивку и общее количество доступных данных
        spam_count = sum(1 for _, is_s in spam_examples if is_s)
        notspam_count = len(spam_examples) - spam_count
        reg_spam = sum(1 for _, is_s in regression_set if is_s)
        reg_notspam = len(regression_set) - reg_spam
        total_training = db.count_training_examples()
        total_reviewed = db.count_correctly_classified()
        total_ordinary = db.count_ordinary_messages()

        await _send_progress(
            f"📊 <b>Датасет валидации</b>\n"
            f"  • Training examples: <b>{len(spam_examples)}</b> из {total_training} "
            f"(спам: {spam_count}, не-спам: {notspam_count})\n"
            f"  • Прошедшие ревью админа: <b>{len(regression_set)}</b> из {total_reviewed} "
            f"(спам: {reg_spam}, не-спам: {reg_notspam})\n"
            f"  • Обычные сообщения без ревью: <b>{len(ordinary_set)}</b> из {total_ordinary} "
            f"(cap для скорости)\n"
            f"  • <b>Всего: {total_eval}</b>"
        )

        # Анализ спам-волн (паттерны у недавних забаненных)
        wave_analysis = ""
        try:
            banned_profiles = db.get_recent_banned_profiles(168)  # 7 дней
            if banned_profiles:
                wave_analysis = await detect_spam_waves(banned_profiles)
                if wave_analysis:
                    await _send_progress(
                        f"🌊 <b>Обнаружены спам-волны</b> (за 7 дней, {len(banned_profiles)} забаненных)\n"
                        f"{html.escape(wave_analysis[:600])}"
                    )
        except Exception as e:
            logger.warning(f"Ошибка анализа спам-волн: {e}")

        # ── Фаза 2: Оценка текущего промпта ──
        await _send_progress(f"🔍 Оцениваю текущий промпт на {total_eval} примерах...")
        full_eval_set = spam_examples + regression_set + ordinary_set
        current_acc, current_ok, current_total, current_errors = await evaluate_prompt(current_prompt, full_eval_set)

        if current_total == 0:
            # Сделаем один прямой тест чтобы увидеть точную ошибку
            test_err = "неизвестная ошибка"
            try:
                test_text = full_eval_set[0][0] if full_eval_set else "тест"
                await classify_message(current_prompt, test_text)
            except Exception as e:
                test_err = f"{type(e).__name__}: {str(e)[:300]}"
            await _send_progress(
                f"❌ <b>Валидация невозможна</b>: все {total_eval} примеров упали.\n"
                f"Модель: <code>{html.escape(LLM_MODEL)}</code>\n"
                f"Ошибка: <code>{html.escape(test_err)}</code>"
            )
            return

        # Запоминаем что бот сейчас правильно классифицирует — для детектора регрессий
        current_correct_set = set()
        for text, is_spam in full_eval_set:
            is_in_errors = any(e[0] == text[:120] for e in current_errors)
            if not is_in_errors:
                current_correct_set.add(text)

        await _send_progress(
            f"📈 <b>Текущая точность:</b> {current_acc:.0%} ({current_ok}/{current_total})\n"
            f"Ошибок: {len(current_errors)}"
        )

        # ── Фаза 3: Многостратегическая генерация ──
        candidates = []  # [(strategy_name, prompt, analysis, accuracy, regressions, fp, fn)]
        failed_attempts = []  # для контекста следующих попыток

        # Названия сообщений-триггеров для проверки literal containment
        trigger_texts = [trigger_message] + [e[0] for e in current_errors[:10]]

        for i, strategy in enumerate(IMPROVEMENT_STRATEGIES, 1):
            await _send_progress(f"🧠 <b>Попытка {i}/{len(IMPROVEMENT_STRATEGIES)}</b>: стратегия «{strategy['name']}»")

            analysis, improved = await generate_improved_prompt_with_strategy(
                strategy, current_prompt, trigger_message, trigger_error_type,
                current_errors, failed_attempts, wave_analysis
            )

            if not improved:
                msg = f"LLM не вернул промпт ({analysis[:120] if analysis else 'нет ответа'})"
                failed_attempts.append((strategy['name'], None, msg))
                await _send_progress(f"❌ Попытка {i}: {html.escape(msg)}")
                continue

            # Валидация структуры
            problems = validate_prompt(improved)
            if problems:
                msg = f"невалидный: {', '.join(problems)}"
                failed_attempts.append((strategy['name'], None, msg))
                await _send_progress(f"❌ Попытка {i}: {html.escape(msg)}")
                continue

            # Проверка: не содержит ли дословных цитат сообщений
            literal_quotes = _contains_literal_messages(improved, trigger_texts)
            if literal_quotes:
                msg = f"содержит дословные цитаты ({len(literal_quotes)}): {literal_quotes[0]}"
                failed_attempts.append((strategy['name'], None, msg))
                await _send_progress(f"❌ Попытка {i}: {html.escape(msg)}")
                continue

            await _send_progress(f"✓ Попытка {i}: промпт сгенерирован ({len(improved)} симв). Валидирую...")

            # Полная валидация
            new_acc, new_ok, new_total, new_errors = await evaluate_prompt(improved, full_eval_set)

            # Считаем регрессии (было правильно → стало неправильно) и fixes (наоборот)
            current_error_set = set(e[0] for e in current_errors)
            new_error_set = set(e[0] for e in new_errors)
            regressions = len(new_error_set - current_error_set)  # новые ошибки
            fixes = len(current_error_set - new_error_set)  # исправленные ошибки

            # FP/FN считаем
            fp, fn = 0, 0
            for err_text, expected, got in new_errors:
                if expected == "SPAM" and got == "NOT_SPAM":
                    fn += 1
                elif expected == "NOT_SPAM" and got == "SPAM":
                    fp += 1

            verdict_emoji = "✅" if new_acc > current_acc else "🔄"
            await _send_progress(
                f"{verdict_emoji} <b>Попытка {i} результат:</b>\n"
                f"  Точность: {new_acc:.0%} (было {current_acc:.0%})\n"
                f"  Исправлено: {fixes} | Регрессий: {regressions}\n"
                f"  FP: {fp} | FN: {fn}"
            )

            candidates.append({
                "strategy": strategy['name'],
                "prompt": improved,
                "analysis": analysis or "",
                "accuracy": new_acc,
                "ok": new_ok,
                "total": new_total,
                "regressions": regressions,
                "fixes": fixes,
                "fp": fp,
                "fn": fn,
                "attempt": i,
            })

            # Запомним для следующих попыток
            failed_attempts.append((strategy['name'], new_acc, analysis[:120] if analysis else ""))

        # ── Фаза 4: Выбор лучшего ──
        if not candidates:
            await _send_progress(
                "❌ <b>Ни одна попытка не дала валидный промпт.</b>\n"
                "Промпт не изменён."
            )
            return

        # Сортируем по net gain (fixes - regressions), затем по точности
        candidates.sort(key=lambda c: (c["fixes"] - c["regressions"], c["accuracy"]), reverse=True)
        best = candidates[0]

        # Критерии: net-positive (исправил > сломал) И точность не упала
        accuracy_gain = best["accuracy"] - current_acc
        net_gain = best["fixes"] - best["regressions"]

        should_apply = False
        reason = ""
        if net_gain > 0 and accuracy_gain > 0:
            should_apply = True
            reason = (
                f"Net-positive: исправил {best['fixes']}, сломал {best['regressions']} "
                f"(чистый выигрыш +{net_gain}). Точность {accuracy_gain:+.0%}"
            )
        elif current_acc < 0.5 and best["accuracy"] > current_acc:
            should_apply = True
            reason = f"Текущая точность критически низкая ({current_acc:.0%}), применяю лучший"
        else:
            if net_gain <= 0:
                reason = f"Net-negative: регрессий {best['regressions']} ≥ исправлений {best['fixes']}"
            elif accuracy_gain <= 0:
                reason = f"Точность не выросла ({accuracy_gain:+.0%})"

        # Сводка по всем кандидатам — с net gain
        summary_lines = ["📋 <b>Все попытки:</b>"]
        for c in candidates:
            mark = "🏆" if c is best else "•"
            net = c["fixes"] - c["regressions"]
            summary_lines.append(
                f"  {mark} #{c['attempt']} «{c['strategy']}»: {c['accuracy']:.0%}, "
                f"fix={c['fixes']}, рег={c['regressions']}, net={net:+d}"
            )
        await _send_progress("\n".join(summary_lines))

        if should_apply:
            db.save_prompt_version(
                best["prompt"],
                f"Авто ({best['strategy']}): {best['accuracy']:.0%} vs {current_acc:.0%}, net={net_gain:+d}"
            )
            await _send_progress(
                f"✅ <b>Промпт обновлён</b>\n"
                f"Стратегия: «{best['strategy']}»\n"
                f"Точность: {current_acc:.0%} → {best['accuracy']:.0%}\n"
                f"Исправлено: {best['fixes']} | Регрессий: {best['regressions']} | Net: {net_gain:+d}\n"
                f"Причина: {reason}\n\n"
                f"<b>Анализ:</b> {html.escape(best['analysis'][:400])}\n\n"
                f"Откатить: /rollback (см. /history)"
            )
            await _send_full_prompt(best["prompt"], "📝 <b>ФИНАЛЬНЫЙ ПРОМПТ</b>")
        else:
            await _send_progress(
                f"🔄 <b>Промпт НЕ обновлён</b>\n"
                f"Лучший кандидат: «{best['strategy']}» — {best['accuracy']:.0%} (текущий {current_acc:.0%})\n"
                f"Причина: {reason}\n\n"
                f"Текущий промпт остаётся в силе. Few-shot примеры продолжают учитывать ошибки."
            )

    except Exception as e:
        logger.error(f"Ошибка автообучения: {e}", exc_info=True)
        await _send_progress(f"⚠️ Ошибка автообучения: {html.escape(str(e))}")
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


async def _weekly_improve_loop():
    """Фоновый цикл: еженедельное обучение промпта."""
    while True:
        await asyncio.sleep(604800)  # 7 дней
        try:
            logger.info("🔍 Запуск еженедельного обучения промпта")
            await auto_improve_prompt("weekly", "еженедельное обучение")
        except Exception as e:
            logger.error(f"Ошибка еженедельного обучения: {e}")


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
        # Определяем тип: короткий невинный текст = profile spam, иначе text spam
        spam_type = _classify_spam_type(spam_text)
        db.add_training_example(spam_text, True, 'FORWARDED_SPAM', spam_type)
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
        "/models — какие LLM-модели сейчас используются\n"
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


@dp.message(Command("models"))
@require_admin
async def cmd_models(message: types.Message):
    """Показывает текущие модели и проверяет доступность всех кандидатов."""
    await message.reply("🔍 Проверяю доступность моделей...")
    lines = [
        f"🤖 <b>Используются сейчас:</b>",
        f"  • Классификация: <code>{html.escape(LLM_MODEL)}</code>",
        f"  • Улучшение промпта: <code>{html.escape(LLM_IMPROVEMENT_MODEL)}</code>",
        "",
        "<b>Проверка кандидатов:</b>",
    ]
    all_candidates = list(dict.fromkeys(LLM_MODEL_CANDIDATES + LLM_IMPROVEMENT_MODEL_CANDIDATES))
    for c in all_candidates:
        ok, err = await _probe_model(c)
        if ok:
            lines.append(f"  ✅ <code>{html.escape(c)}</code>")
        else:
            short_err = err.split(":")[0] if err else "error"
            lines.append(f"  ❌ <code>{html.escape(c)}</code> — {html.escape(short_err)}")
    await message.reply("\n".join(lines), parse_mode='HTML')


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
    # Определяем тип спама: если reasoning упоминает профиль/канал — это context spam
    spam_type = 'text'
    if is_spam and reasoning:
        r_lower = (reasoning or '').lower()
        if any(kw in r_lower for kw in ['профил', 'profile', 'канал', 'channel', 'bio', 'переслано']):
            spam_type = 'context'
    db.add_training_example(message_text, is_spam, 'ADMIN_FEEDBACK', spam_type)

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
        BotCommand(command="models", description="Проверить доступные LLM модели (админ)"),
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

    # Автодетект моделей (пробуем каждую из списка до первой рабочей)
    detection = await _autodetect_models()

    logger.info(
        f"🤖 Kill Yr Spammers | admin={ADMIN_ID} | groups={len(ALLOWED_GROUP_IDS)} "
        f"| model={LLM_MODEL} | improve={LLM_IMPROVEMENT_MODEL} "
        f"| auto_improve_after={AUTO_IMPROVE_AFTER_ERRORS} errors"
    )

    # Отчёт админу о выбранных моделях
    try:
        report_lines = ["🚀 <b>Бот запущен</b>"]
        if detection["classification"]:
            report_lines.append(f"  • Классификация: <code>{detection['classification']}</code>")
        else:
            report_lines.append("  • ❌ Не найдена рабочая модель классификации!")
        if detection["improvement"]:
            report_lines.append(f"  • Улучшение промпта: <code>{detection['improvement']}</code>")
        else:
            report_lines.append("  • ❌ Не найдена рабочая модель улучшения!")
        if detection["errors"]:
            report_lines.append(f"\n<i>Проверено моделей: {len(detection['errors'])} не работают</i>")
        await bot.send_message(ADMIN_ID, "\n".join(report_lines), parse_mode='HTML')
    except Exception as e:
        logger.warning(f"Не удалось отправить startup-отчёт: {e}")

    # Запускаем еженедельный аудит в фоне
    asyncio.create_task(_weekly_improve_loop())
    logger.info("📅 Еженедельный аудит запланирован")

    try:
        await dp.start_polling(bot)
    finally:
        await _http_client.aclose()


if __name__ == "__main__":
    if not os.getenv("RAILWAY_ENVIRONMENT"):
        print("⚠️  Локальный запуск. Ctrl+C для остановки.")
    asyncio.run(main())
