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

import httpx
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, BotCommand
from openai import AsyncOpenAI

from config import (
    BOT_TOKEN, OPENAI_API_KEY, ADMIN_ID, ALLOWED_GROUP_IDS,
    LLM_MODEL, LLM_IMPROVEMENT_MODEL, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, LLM_TIMEOUT, MAX_REQUESTS_PER_MINUTE,
    FEW_SHOT_EXAMPLES_COUNT, CAS_API_URL,
    AUTO_IMPROVE_AFTER_ERRORS, MIN_VALIDATION_EXAMPLES, MAX_VALIDATION_EXAMPLES,
)
import database as db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot: Bot = None
dp = Dispatcher()
openai_client: AsyncOpenAI = None

_user_request_times: dict[int, list[float]] = defaultdict(list)
_http_client: httpx.AsyncClient = None
# Блокировка чтобы не запускать два улучшения одновременно
_improvement_in_progress = False


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
    cleaned = re.sub(r'[^\w\s_]', '', response_text.strip().upper())
    if len(cleaned) < 3:
        return SpamResult.MAYBE_SPAM

    exact = {
        'СПАМ': SpamResult.SPAM, 'SPAM': SpamResult.SPAM,
        'НЕ_СПАМ': SpamResult.NOT_SPAM, 'НЕ СПАМ': SpamResult.NOT_SPAM, 'NOT_SPAM': SpamResult.NOT_SPAM,
        'ВОЗМОЖНО_СПАМ': SpamResult.MAYBE_SPAM, 'ВОЗМОЖНО СПАМ': SpamResult.MAYBE_SPAM, 'MAYBE_SPAM': SpamResult.MAYBE_SPAM,
    }
    if cleaned in exact:
        return exact[cleaned]
    if 'ВОЗМОЖНО' in cleaned:
        return SpamResult.MAYBE_SPAM
    if 'НЕ_СПАМ' in cleaned or 'НЕ СПАМ' in cleaned or 'NOT' in cleaned:
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
            return template.format(message_text=safe_text)
        except KeyError:
            return template.replace("{message_text}", safe_text)


def validate_prompt(prompt_text: str) -> list[str]:
    problems = []
    if "{message_text}" not in prompt_text:
        problems.append("Нет {message_text}")
    for kw in ("СПАМ", "НЕ_СПАМ", "ВОЗМОЖНО_СПАМ"):
        if kw not in prompt_text:
            problems.append(f"Нет {kw}")
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


# ──────────────────────────────────────────────
# LLM: классификация
# ──────────────────────────────────────────────

async def classify_message(
    prompt_template: str,
    message_text: str,
    few_shot: str = "",
    user_msg_count: int = 0,
    is_cas_banned: bool = False,
) -> SpamResult:
    """Классификация одного сообщения заданным промптом. Вынесена для переиспользования в валидации."""
    prompt = safe_format_prompt(prompt_template, message_text, few_shot)

    # Добавляем контекст (только если передан)
    context_lines = []
    if user_msg_count > 0:
        context_lines.append(f"Контекст: пользователь ранее написал {user_msg_count} сообщений (снижает вероятность спама).")
    if is_cas_banned:
        context_lines.append("Контекст: пользователь найден в антиспам-базе CAS (СИЛЬНО повышает вероятность спама).")
    if context_lines:
        context = "\n".join(context_lines) + "\n"
        safe_text = message_text.replace("{", "{{").replace("}", "}}")
        prompt = prompt.replace(f"Сообщение: «{safe_text}»", f"{context}Сообщение: «{safe_text}»")

    response = await openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        timeout=LLM_TIMEOUT,
    )
    return parse_llm_response(response.choices[0].message.content.strip())


async def check_message_with_llm(
    message_text: str,
    user_id: int = None,
    user_msg_count: int = 0,
    is_cas_banned: bool = False,
) -> SpamResult:
    if user_id and not check_rate_limit(user_id):
        return SpamResult.MAYBE_SPAM

    prompt_template = db.get_current_prompt()
    few_shot = build_few_shot_block()

    try:
        result = await classify_message(prompt_template, message_text, few_shot, user_msg_count, is_cas_banned)
        logger.info(f"LLM → {result.value} (len={len(message_text)}, msgs={user_msg_count}, cas={is_cas_banned})")
        return result
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return SpamResult.MAYBE_SPAM


# ──────────────────────────────────────────────
# Автоматическое улучшение промпта
# ──────────────────────────────────────────────

async def evaluate_prompt(prompt_template: str, examples: list) -> tuple[float, int, int]:
    """Оценить промпт на примерах. Возвращает (accuracy, correct, total).

    examples: [(text, is_spam), ...]
    """
    if not examples:
        return 0.0, 0, 0

    correct = 0
    total = len(examples)

    for text, is_spam in examples:
        try:
            result = await classify_message(prompt_template, text)
            # СПАМ или ВОЗМОЖНО_СПАМ считаем за "спам" при is_spam=True
            predicted_spam = result in (SpamResult.SPAM, SpamResult.MAYBE_SPAM)
            actual_spam = bool(is_spam)
            if predicted_spam == actual_spam:
                correct += 1
        except Exception as e:
            logger.warning(f"Ошибка валидации примера: {e}")
            total -= 1  # Не считаем ошибочные

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


async def generate_improved_prompt(error_type: str, message_text: str) -> tuple[str, str] | tuple[None, None]:
    """Генерирует улучшенный промпт. Возвращает (analysis, improved_prompt) или (None, None)."""
    current_prompt = db.get_current_prompt()

    descriptions = {
        "missed_spam": "Бот НЕ определил как спам, хотя это спам",
        "uncertain_spam": "Бот определил как ВОЗМОЖНО_СПАМ, но это точно спам",
        "false_positive": "Бот определил как спам, хотя это НЕ спам",
    }
    description = descriptions.get(error_type, error_type)

    recent_mistakes = db.get_recent_mistakes(5)
    mistakes_block = ""
    if recent_mistakes:
        lines = ["Другие недавние ошибки бота:"]
        for text, bot_dec, admin_dec, _ in recent_mistakes:
            lines.append(f"  - «{text[:80]}» — бот: {bot_dec}, правильно: {admin_dec}")
        mistakes_block = "\n".join(lines)

    analysis_prompt = f"""Ты эксперт по созданию промптов для определения спама в Telegram.

ТЕКУЩИЙ ПРОМПТ:
{current_prompt}

ОШИБКА: {description}
Сообщение: "{message_text}"

{mistakes_block}

ЗАДАЧА: Улучши промпт так, чтобы он правильно обрабатывал это и похожие сообщения.
Сохрани ВСЕ существующие критерии, исключения и структуру. Только дополни/уточни.
Промпт ОБЯЗАН содержать {{message_text}} и {{few_shot_block}} — это шаблонные переменные.
Промпт ОБЯЗАН содержать три варианта ответа: СПАМ, НЕ_СПАМ, ВОЗМОЖНО_СПАМ.

Ответь в формате:
АНАЛИЗ: [причина ошибки в 1-2 предложениях]
ИТОГОВЫЙ_ПРОМПТ: [полный улучшенный промпт]"""

    try:
        response = await openai_client.chat.completions.create(
            model=LLM_IMPROVEMENT_MODEL,
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=2000,
            temperature=0.3,
            timeout=30,
        )
        text = response.choices[0].message.content.strip()

        if "ИТОГОВЫЙ_ПРОМПТ:" not in text:
            return text, None

        improved = text.split("ИТОГОВЫЙ_ПРОМПТ:", 1)[1].strip()

        # Патчим если потеряны обязательные элементы
        if "{message_text}" not in improved:
            improved += "\n\nСообщение: «{message_text}»\n\nОтвет:"
        if "{few_shot_block}" not in improved:
            improved = improved.replace("Сообщение: «{message_text}»", "{few_shot_block}\nСообщение: «{message_text}»")

        analysis = text.split("ИТОГОВЫЙ_ПРОМПТ:")[0].strip()
        return analysis, improved

    except Exception as e:
        logger.error(f"Ошибка генерации промпта: {e}")
        return None, None


async def auto_improve_prompt(trigger_error_type: str, trigger_message: str):
    """Автоматическое улучшение промпта с валидацией.

    Логика:
    1. Генерирует улучшенный промпт
    2. Оценивает текущий и новый на validation set
    3. Применяет новый только если он не хуже
    4. Отправляет отчёт админу
    """
    global _improvement_in_progress
    if _improvement_in_progress:
        return
    _improvement_in_progress = True

    try:
        examples_count = db.count_training_examples()
        has_enough_for_validation = examples_count >= MIN_VALIDATION_EXAMPLES

        # 1. Генерируем улучшенный промпт
        analysis, improved = await generate_improved_prompt(trigger_error_type, trigger_message)
        if not improved:
            await bot.send_message(ADMIN_ID, f"⚠️ Не удалось сгенерировать улучшенный промпт")
            return

        problems = validate_prompt(improved)
        if problems:
            await bot.send_message(ADMIN_ID, f"⚠️ Сгенерированный промпт невалиден: {', '.join(problems)}")
            return

        # 2. Валидация на примерах (если достаточно данных)
        if has_enough_for_validation:
            validation_examples = db.get_validation_examples(MAX_VALIDATION_EXAMPLES)

            current_prompt = db.get_current_prompt()
            current_acc, current_ok, current_total = await evaluate_prompt(current_prompt, validation_examples)
            new_acc, new_ok, new_total = await evaluate_prompt(improved, validation_examples)

            logger.info(f"Валидация: текущий={current_acc:.0%} ({current_ok}/{current_total}), "
                        f"новый={new_acc:.0%} ({new_ok}/{new_total})")

            if new_acc < current_acc:
                # Новый промпт хуже → НЕ применяем
                report = (
                    f"🔄 <b>Промпт НЕ обновлён (не прошёл валидацию)</b>\n\n"
                    f"Текущий: {current_acc:.0%} ({current_ok}/{current_total})\n"
                    f"Новый: {new_acc:.0%} ({new_ok}/{new_total})\n\n"
                    f"Анализ ошибки: {analysis}\n"
                    f"Few-shot примеры продолжают учитывать это исправление."
                )
                await bot.send_message(ADMIN_ID, report, parse_mode='HTML')
                return

            # Новый промпт не хуже → применяем
            db.save_prompt_version(improved, f"Авто: {trigger_error_type} ({new_acc:.0%} vs {current_acc:.0%})")

            report = (
                f"✅ <b>Промпт автоматически обновлён</b>\n\n"
                f"Было: {current_acc:.0%} ({current_ok}/{current_total})\n"
                f"Стало: {new_acc:.0%} ({new_ok}/{new_total})\n\n"
                f"Причина: {analysis}\n\n"
                f"<code>{improved[:500]}{'...' if len(improved) > 500 else ''}</code>\n\n"
                f"Откатить: /rollback (из /history)"
            )
            await bot.send_message(ADMIN_ID, report, parse_mode='HTML')

        else:
            # Мало примеров для валидации — применяем без проверки, но предупреждаем
            db.save_prompt_version(improved, f"Авто (без валидации, {examples_count} примеров): {trigger_error_type}")

            report = (
                f"✅ <b>Промпт обновлён</b> (мало данных для валидации: {examples_count}/{MIN_VALIDATION_EXAMPLES})\n\n"
                f"Причина: {analysis}\n\n"
                f"Откатить: /rollback (из /history)"
            )
            await bot.send_message(ADMIN_ID, report, parse_mode='HTML')

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


async def send_to_admin(message: types.Message, result: SpamResult):
    emoji = "🔴" if result == SpamResult.SPAM else "🟡"
    text = (
        f"{emoji} <b>{result.value}</b>\n\n"
        f"<b>От:</b> {message.from_user.full_name} (@{message.from_user.username or 'n/a'})\n"
        f"<b>Группа:</b> {message.chat.title}\n"
        f"<b>Время:</b> {message.date.strftime('%H:%M:%S')}\n\n"
        f"<b>Сообщение:</b>\n<code>{message.text}</code>"
    )
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🔴 СПАМ", callback_data=f"spam_{message.message_id}"),
        InlineKeyboardButton(text="🟢 НЕ СПАМ", callback_data=f"not_spam_{message.message_id}"),
    ]])
    try:
        await bot.send_message(ADMIN_ID, text, reply_markup=keyboard, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Ошибка отправки админу: {e}")


async def ban_and_report(message: types.Message, result: SpamResult):
    uid, cid = message.from_user.id, message.chat.id

    if message.sender_chat:
        await send_to_admin(message, result)
        return
    if db.has_user_old_activity(uid, cid, 10):
        await send_to_admin(message, result)
        return

    try:
        await bot.delete_message(cid, message.message_id)
        await bot.ban_chat_member(chat_id=cid, user_id=uid)
        banned, failed = await ban_user_in_all_groups(uid, exclude_chat_id=cid)
    except Exception as e:
        logger.error(f"Ошибка бана: {e}")
        await send_to_admin(message, result)
        return

    text = (
        f"🔴 <b>АВТОБАН ЗА СПАМ</b>\n\n"
        f"<b>Забанен:</b> {message.from_user.full_name} (@{message.from_user.username or 'n/a'})\n"
        f"<b>User ID:</b> <code>{uid}</code>\n"
        f"<b>Группа:</b> {message.chat.title}\n\n"
        f"<b>Сообщение:</b>\n<code>{message.text}</code>\n\n"
        f"✅ Забанен в {len(banned) + 1} группах"
    )
    if failed:
        text += f"\n⚠️ Не удалось в {len(failed)} группах"
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🟢 НЕ СПАМ (разбанить)", callback_data=f"unban_{uid}_{cid}_{message.message_id}")
    ]])
    try:
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

    db.add_training_example(message.text, True, 'FORWARDED_SPAM')

    parts = [f"🔄 Обрабатываю спам от <b>{original_username or 'неизвестного'}</b>"]

    if original_user_id:
        deleted = await delete_user_messages(original_user_id)
        banned, failed = await ban_user_in_all_groups(original_user_id)
        parts.append(f"🗑️ Удалено: {deleted} | 🔨 Забанен в {len(banned)} группах")
    else:
        parts.append("⚠️ User ID недоступен — бан невозможен")

    await message.reply("\n".join(parts), parse_mode='HTML')

    # Запускаем автоулучшение
    await maybe_trigger_improvement("missed_spam", message.text)


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
        "/prompt — текущий промпт\n"
        "/history — история версий промпта\n"
        "/rollback N — откатить промпт к версии #N\n"
        "/editprompt — ручное редактирование промпта\n"
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


@dp.message(Command("prompt"))
@require_admin
async def cmd_prompt(message: types.Message):
    current = db.get_current_prompt()
    await message.reply(f"📝 <b>Текущий промпт:</b>\n\n<code>{current}</code>", parse_mode='HTML')


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

@dp.message(F.content_type == 'text')
async def handle_message(message: types.Message):
    if message.chat.type == 'private':
        return
    if message.chat.type in ('group', 'supergroup') and message.chat.id not in ALLOWED_GROUP_IDS:
        return
    if should_skip_message(message):
        return

    uid, cid = message.from_user.id, message.chat.id
    user_msg_count = db.count_user_messages(uid, cid)
    is_cas_banned = await check_cas_ban(uid)

    # CAS + нет истории → автобан
    if is_cas_banned and user_msg_count == 0:
        try:
            db.save_message(message.message_id, cid, uid, message.from_user.username or '', message.text, "СПАМ")
        except Exception:
            pass
        await ban_and_report(message, SpamResult.SPAM)
        return

    result = await check_message_with_llm(message.text, uid, user_msg_count, is_cas_banned)

    try:
        db.save_message(message.message_id, cid, uid, message.from_user.username or '', message.text, result.value)
    except Exception as e:
        logger.error(f"Ошибка сохранения: {e}")

    if result == SpamResult.SPAM:
        await ban_and_report(message, result)
    elif result == SpamResult.MAYBE_SPAM:
        await send_to_admin(message, result)


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

    message_text, llm_result, user_id, chat_id = row
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
    new_text = f"{callback.message.text}\n\n{emoji} <b>Решение: {decision}</b>{ban_info}"
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
        BotCommand(command="prompt", description="Текущий промпт (админ)"),
        BotCommand(command="history", description="История промптов (админ)"),
        BotCommand(command="rollback", description="Откат промпта (админ)"),
        BotCommand(command="editprompt", description="Редактировать промпт (админ)"),
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
    try:
        await dp.start_polling(bot)
    finally:
        await _http_client.aclose()


if __name__ == "__main__":
    if not os.getenv("RAILWAY_ENVIRONMENT"):
        print("⚠️  Локальный запуск. Ctrl+C для остановки.")
    asyncio.run(main())
