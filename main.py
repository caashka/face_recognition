# main.py

import os
import sys
import json
import subprocess
import time

USERS_FILE = "users.json"


def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def get_next_user_id(users):
    if not users:
        return 1
    existed_ids = list(map(int, users.keys()))
    return max(existed_ids) + 1


def add_new_person():
    users = load_users()
    print("\n=== Добавление нового человека ===")
    new_name = input("Enter name: ").strip()
    if not new_name:
        print("Имя не может быть пустым. Отмена.\n")
        return

    new_id = get_next_user_id(users)
    users[str(new_id)] = new_name
    save_users(users)

    print(f"\nНовый пользователь: [ID={new_id}, Имя={new_name}] добавлен")
    print("Сейчас будет запущен сбор датасета...\n")

    cmd_dataset = [sys.executable, "dataset_creator.py", str(new_id), new_name]
    subprocess.run(cmd_dataset)

    print("\nСбор датасета завершён. Запускаем тренировку модели...\n")
    time.sleep(1)
    cmd_trainer = [sys.executable, "trainer.py", "LBPH"]
    subprocess.run(cmd_trainer)
    print("\nОбучение завершено. Файл trained_faces.yml обновлён.")
    print("Новый пользователь успешно добавлен и обучен!\n")


def train_model(algorithm='LBPH'):
    cmd_trainer = [sys.executable, "trainer.py", algorithm]
    subprocess.run(cmd_trainer)


def start_recognition(algorithm='LBPH'):
    cmd_recognition = [sys.executable, "face_recognition.py", "--algorithm", algorithm]
    subprocess.run(cmd_recognition)


def start_stream_recognition(algorithm='LBPH'):
    stream_url = input("Введите URL видеопотока: ").strip()
    if not stream_url:
        print("URL не может быть пустым. Отмена.\n")
        return

    delay = input("Введите задержку между кадрами (в миллисекундах, по умолчанию 30): ").strip()
    delay = int(delay) if delay and delay.isdigit() else 30

    cmd_stream_recognition = [sys.executable, "face_recognition.py", "--stream", stream_url, "--delay", str(delay), "--algorithm", algorithm]
    subprocess.run(cmd_stream_recognition)


def main_menu():
    algorithm = "LBPH"
    while True:
        print("=========================================")
        print("    ПОТОКОВАЯ ДЕТЕКЦИЯ ЛИЦ   ")
        print("=========================================")
        print("1. Добавить нового человека")
        print("2. Тренировать распознаватель (manual)")
        print("3. Определение лиц c вебкамеры")
        print("4. Определение лиц на потоковом видео")
        print("5. Изменить алгоритм (по умолчанию - LBPH)")
        print("6. Выход")
        choice = input("\nВыберите действие (1-6): ").strip()

        if choice == '1':
            add_new_person()
        elif choice == '2':
            train_model(algorithm)
        elif choice == '3':
            start_recognition(algorithm)
        elif choice == '4':
            start_stream_recognition(algorithm)
        elif choice == '5':
            algorithm = input("Выбирите алгоритм LBPH или Fisher: ").strip() or "LBPH"
        elif choice == '6':
            print("\nВыход из программы...")
            break
        else:
            print("Некорректный выбор!\n")


if __name__ == "__main__":
    main_menu()
