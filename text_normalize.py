"""
Нормализация текста перед отправкой в LLM.

Спамеры обходят классификаторы через:
1. Гомоглифы: замена кириллических букв латинскими (визуально идентичны)
2. Zero-width символы: невидимые символы между буквами
3. Combining diacritics: мусорные диакритики поверх букв
4. RTL override: смена направления текста
5. Whitespace tricks: нестандартные пробелы, табуляция вместо пробелов

Этот модуль приводит текст к каноническому виду.
"""

import re
import unicodedata

# ──────────────────────────────────────────────
# 1. Zero-width и невидимые символы
# ──────────────────────────────────────────────

# Полный список невидимых Unicode символов, используемых для обхода
INVISIBLE_CHARS = set([
    '\u200b',  # Zero Width Space
    '\u200c',  # Zero Width Non-Joiner
    '\u200d',  # Zero Width Joiner
    '\u200e',  # Left-to-Right Mark
    '\u200f',  # Right-to-Left Mark
    '\u2060',  # Word Joiner
    '\u2061',  # Function Application
    '\u2062',  # Invisible Times
    '\u2063',  # Invisible Separator
    '\u2064',  # Invisible Plus
    '\ufeff',  # Zero Width No-Break Space (BOM)
    '\u00ad',  # Soft Hyphen
    '\u034f',  # Combining Grapheme Joiner
    '\u061c',  # Arabic Letter Mark
    '\u115f',  # Hangul Choseong Filler
    '\u1160',  # Hangul Jungseong Filler
    '\u17b4',  # Khmer Vowel Inherent Aq
    '\u17b5',  # Khmer Vowel Inherent Aa
    '\u180e',  # Mongolian Vowel Separator
    '\uffa0',  # Halfwidth Hangul Filler
])

# Bidi control characters — используются для RTL override атак
BIDI_CONTROLS = set([
    '\u202a',  # Left-to-Right Embedding
    '\u202b',  # Right-to-Left Embedding
    '\u202c',  # Pop Directional Formatting
    '\u202d',  # Left-to-Right Override
    '\u202e',  # Right-to-Left Override
    '\u2066',  # Left-to-Right Isolate
    '\u2067',  # Right-to-Left Isolate
    '\u2068',  # First Strong Isolate
    '\u2069',  # Pop Directional Isolate
])

ALL_INVISIBLE = INVISIBLE_CHARS | BIDI_CONTROLS


def strip_invisible(text: str) -> str:
    """Удалить все невидимые и bidi-control символы."""
    return ''.join(c for c in text if c not in ALL_INVISIBLE)


# ──────────────────────────────────────────────
# 2. Гомоглифы: Latin ↔ Cyrillic
# ──────────────────────────────────────────────

# Наиболее частые подмены: латинские буквы, визуально идентичные кириллическим
# Направление: Latin → Cyrillic (приводим к кириллице, т.к. чаты русскоязычные)
LATIN_TO_CYRILLIC = {
    'A': 'А', 'a': 'а',
    'B': 'В',
    'C': 'С', 'c': 'с',
    'E': 'Е', 'e': 'е',
    'H': 'Н',
    'K': 'К',
    'M': 'М',
    'O': 'О', 'o': 'о',
    'P': 'Р', 'p': 'р',
    'T': 'Т',
    'X': 'Х', 'x': 'х',
    'y': 'у',
}


def normalize_homoglyphs(text: str) -> str:
    """Заменить латинские гомоглифы на кириллические в тексте с преобладанием кириллицы.

    Работает только если текст преимущественно кириллический.
    Для латинского текста замена не нужна (спам на английском и так поймается).
    """
    # Считаем пропорцию кириллических символов
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return text

    cyrillic_count = sum(1 for c in alpha_chars if '\u0400' <= c <= '\u04ff')
    cyrillic_ratio = cyrillic_count / len(alpha_chars)

    # Если текст менее 30% кириллица — не трогаем (английский текст, смешанный)
    if cyrillic_ratio < 0.3:
        return text

    # Заменяем только отдельные латинские символы внутри кириллических слов
    result = []
    for c in text:
        if c in LATIN_TO_CYRILLIC:
            result.append(LATIN_TO_CYRILLIC[c])
        else:
            result.append(c)
    return ''.join(result)


# ──────────────────────────────────────────────
# 3. Combining diacritics (Zalgo text)
# ──────────────────────────────────────────────

def strip_excessive_diacritics(text: str) -> str:
    """Удалить combining diacritics если их больше 1 на символ (Zalgo text)."""
    result = []
    combining_count = 0
    for c in text:
        cat = unicodedata.category(c)
        if cat.startswith('M'):  # Mark (combining)
            combining_count += 1
            if combining_count <= 1:
                result.append(c)
            # Больше 1 combining mark подряд — пропускаем (Zalgo)
        else:
            combining_count = 0
            result.append(c)
    return ''.join(result)


# ──────────────────────────────────────────────
# 4. Whitespace нормализация
# ──────────────────────────────────────────────

# Нестандартные пробелы, используемые для обхода
FANCY_SPACES = re.compile(r'[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]')


def normalize_whitespace(text: str) -> str:
    """Заменить нестандартные пробелы на обычные, схлопнуть множественные."""
    text = FANCY_SPACES.sub(' ', text)
    # Схлопнуть множественные пробелы (но сохранить переносы строк)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


# ──────────────────────────────────────────────
# 5. Общая нормализация
# ──────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Полная нормализация текста перед классификацией.

    Порядок важен:
    1. Unicode NFC нормализация (каноническая форма)
    2. Удаление невидимых символов
    3. Удаление Zalgo diacritics
    4. Нормализация пробелов
    5. Замена гомоглифов (после NFC, чтобы composed формы были корректны)
    """
    if not text:
        return text

    # 1. Unicode NFC
    text = unicodedata.normalize('NFC', text)

    # 2. Невидимые символы
    text = strip_invisible(text)

    # 3. Zalgo
    text = strip_excessive_diacritics(text)

    # 4. Пробелы
    text = normalize_whitespace(text)

    # 5. Гомоглифы
    text = normalize_homoglyphs(text)

    return text
