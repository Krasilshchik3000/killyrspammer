#!/usr/bin/env python3
"""
Менеджер процессов для антиспам-бота
Обеспечивает безопасный запуск, остановку и перезапуск без конфликтов
"""
import os
import sys
import signal
import time
import psutil
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PID_FILE = SCRIPT_DIR / "bot.pid"
LOG_FILE = SCRIPT_DIR / "bot.log"

def get_bot_pid():
    """Получить PID бота из файла"""
    if PID_FILE.exists():
        try:
            with open(PID_FILE, 'r') as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            return None
    return None

def is_bot_running():
    """Проверить, запущен ли бот"""
    pid = get_bot_pid()
    if pid is None:
        return False
    
    try:
        process = psutil.Process(pid)
        # Проверяем, что это действительно наш скрипт
        if 'main.py' in ' '.join(process.cmdline()):
            return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    
    # Если процесс не найден, удаляем PID файл
    if PID_FILE.exists():
        PID_FILE.unlink()
    return False

def stop_bot():
    """Остановить бота"""
    pid = get_bot_pid()
    if pid is None:
        print("❌ Бот не запущен")
        return False
    
    try:
        process = psutil.Process(pid)
        print(f"🛑 Останавливаю бота (PID: {pid})...")
        
        # Сначала пробуем мягкую остановку
        process.terminate()
        
        # Ждем до 10 секунд
        try:
            process.wait(timeout=10)
        except psutil.TimeoutExpired:
            print("⚠️ Принудительная остановка...")
            process.kill()
            process.wait(timeout=5)
        
        print("✅ Бот остановлен")
        
        # Удаляем PID файл
        if PID_FILE.exists():
            PID_FILE.unlink()
        
        return True
        
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        print(f"❌ Ошибка остановки: {e}")
        # Удаляем PID файл в любом случае
        if PID_FILE.exists():
            PID_FILE.unlink()
        return False

def start_bot():
    """Запустить бота"""
    if is_bot_running():
        print("⚠️ Бот уже запущен!")
        return False
    
    print("🚀 Запускаю бота...")
    
    # Запускаем бота в фоновом режиме
    cmd = [sys.executable, str(SCRIPT_DIR / "main.py")]
    
    # Перенаправляем вывод в лог файл
    with open(LOG_FILE, 'a') as log:
        log.write(f"\n--- Запуск бота: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        process = psutil.Popen(
            cmd,
            stdout=log,
            stderr=log,
            cwd=SCRIPT_DIR
        )
    
    # Сохраняем PID
    with open(PID_FILE, 'w') as f:
        f.write(str(process.pid))
    
    # Проверяем, что процесс запустился
    time.sleep(2)
    if process.is_running():
        print(f"✅ Бот запущен (PID: {process.pid})")
        print(f"📝 Логи: {LOG_FILE}")
        return True
    else:
        print("❌ Ошибка запуска бота")
        if PID_FILE.exists():
            PID_FILE.unlink()
        return False

def restart_bot():
    """Перезапустить бота"""
    print("🔄 Перезапуск бота...")
    stop_bot()
    time.sleep(1)  # Небольшая пауза
    return start_bot()

def status_bot():
    """Показать статус бота"""
    if is_bot_running():
        pid = get_bot_pid()
        try:
            process = psutil.Process(pid)
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            create_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(process.create_time()))
            
            print(f"✅ Бот работает")
            print(f"   PID: {pid}")
            print(f"   Запущен: {create_time}")
            print(f"   Память: {memory_mb:.1f} MB")
            print(f"   CPU: {cpu_percent:.1f}%")
        except psutil.NoSuchProcess:
            print("❌ Бот не запущен")
    else:
        print("❌ Бот не запущен")

def show_logs():
    """Показать последние логи"""
    if LOG_FILE.exists():
        print("📝 Последние 50 строк логов:")
        print("-" * 50)
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            for line in lines[-50:]:
                print(line.rstrip())
    else:
        print("📝 Лог файл не найден")

def main():
    parser = argparse.ArgumentParser(description='Менеджер антиспам-бота')
    parser.add_argument('command', choices=['start', 'stop', 'restart', 'status', 'logs'], 
                       help='Команда для выполнения')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        start_bot()
    elif args.command == 'stop':
        stop_bot()
    elif args.command == 'restart':
        restart_bot()
    elif args.command == 'status':
        status_bot()
    elif args.command == 'logs':
        show_logs()

if __name__ == "__main__":
    main()
