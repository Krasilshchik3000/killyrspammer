# 🤖 Kill Yr Spammers — Telegram антиспам-бот

LLM-классификация спама в Telegram-группах с самообучающимся промптом.

## Архитектура детекции (по слоям, от дешёвых к дорогим)

1. **Fingerprint** — точное совпадение с подтверждённым спамом → мгновенный бан (без LLM)
2. **Базы спамеров** — CAS + lols.bot параллельно; в базе + нет истории → автобан
3. **Trust** — 3+ осмысленных сообщений (≥10 симв) → пропуск без проверки (кроме forwards)
4. **Сигналы риска** — профиль (bio/канал), опасные документы, forwards от новичков
5. **LLM-классификация** — gpt-5.4-mini, structured output, few-shot из решений админа
6. **Эскалация** — ВОЗМОЖНО_СПАМ + сильный сигнал → бан; слабые сигналы → ревью админу

Также: перепроверка отредактированных сообщений (edit-to-spam), Vision для
картиночного спама, массовый бан по похожему тексту при пересылке.

## Самообучение

- Каждое решение админа (кнопки СПАМ/НЕ СПАМ, пересылка пропущенного спама)
  → training example → few-shot в промпте
- Раз в неделю (или /improve): генерация улучшенного промпта (gpt-5.5),
  3 стратегии с early-stop, валидация на всей размеченной базе,
  применение только при net-positive результате

## Деплой (Railway)

Переменные окружения: `BOT_TOKEN`, `OPENAI_API_KEY`, `ADMIN_ID`, `DATABASE_URL`
(PostgreSQL — обязательно, иначе данные стираются при деплое).
Опционально: `LLM_MODEL`, `LLM_BASE_URL`/`LLM_API_KEY` (любой OpenAI-совместимый
провайдер, например Gemini), `TRUSTED_USER_MESSAGES`, `AUTO_IMPROVE_COOLDOWN_MINUTES`.

## Команды админа

`/stats` `/improve` `/models` `/prompt` `/history` `/rollback N`
`/editprompt` `/resetprompt` `/groups`

## Тесты

```bash
pytest tests/ --ignore=tests/test_golden.py   # юнит + интеграционные (без API)
pytest tests/test_golden.py                   # golden dataset (реальные API-вызовы)
```

## Откат

Стабильные срезы: ветка `backup/v1-stable`, тег `backup-v1-before-overhaul`.
