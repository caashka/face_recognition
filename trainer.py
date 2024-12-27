# trainer.py

import os
import cv2
from PIL import Image
import numpy as np
import json

USERS_FILE = "users.json"
TRAINED_FILE_LBPH = "trained_faces_lbph.yml"
TRAINED_FILE_FISHER = "trained_faces_fisher.yml"
FACES_DIR = os.path.abspath(r"faces")  # Use absolute path


def load_users():
    # Загружаем пользователей из файла users.json
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users_dict):
    # Сохраняем пользователей в файл users.json
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users_dict, f, ensure_ascii=False, indent=2)


def get_next_id(users_dict):
    # Получаем следующий доступный ID для нового пользователя
    if not users_dict:
        return 1
    existing_ids = list(map(int, users_dict.keys()))
    return max(existing_ids) + 1


def open_image(file_path):
    # Пытаемся открыть изображение с помощью OpenCV
    img_cv = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # Если OpenCV не смог открыть изображение, пробуем использовать Pillow
    if img_cv is None:
        try:
            img_pil = Image.open(file_path).convert('L')
            # Преобразуем изображение PIL в numpy массив
            img_np = np.array(img_pil)
            # Убеждаемся, что изображение в правильном формате для OpenCV
            img_cv = img_np.astype(np.uint8)
        except Exception as e:
            print(f"Ошибка при открытии изображения с помощью Pillow: {e}")
            return None
    return img_cv


def collect_faces_and_ids():
    # Собираем лица и ID для обучения модели
    users_dict = load_users()
    face_samples = []
    face_ids = []
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    for person_name in os.listdir(FACES_DIR):
        person_folder = os.path.join(FACES_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        found_id = None
        for uid, uname in users_dict.items():
            if uname == person_name:
                found_id = int(uid)
                break
        if found_id is None:
            found_id = get_next_id(users_dict)
            users_dict[str(found_id)] = person_name

        for filename in os.listdir(person_folder):
            base_name, extension = os.path.splitext(filename)
            extension = extension.lower()
            if extension not in [".jpg", ".jpeg", ".png"]:
                continue

            image_path = os.path.join(person_folder, filename)
            try:
                img_gray = open_image(image_path)
                if img_gray is None:
                    print(f"Ошибка: не удалось прочитать изображение {image_path}")
                    continue

                # Предобработка изображения
                img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
                img_gray = cv2.equalizeHist(img_gray)

                faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                if len(faces) == 0:
                    print(f"Предупреждение: на изображении {image_path} не найдено лиц")
                    continue

                for (x, y, w, h) in faces:
                    face_img = img_gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (100, 100))
                    face_samples.append(face_img)
                    face_ids.append(found_id)
            except Exception as e:
                print(f"Ошибка при обработке изображения {image_path}: {str(e)}")

    save_users(users_dict)
    return face_samples, face_ids


def train_model(algorithm='LBPH'):
    # Создаем и обучаем выбарнную модель распознавания лиц
    if algorithm == 'LBPH':
        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=80)
        trained_file = TRAINED_FILE_LBPH
    elif algorithm == 'Fisher':
        recognizer = cv2.face.FisherFaceRecognizer_create(num_components=0)
        trained_file = TRAINED_FILE_FISHER
    else:
        print("Этот алгоритм не поддерживается! Ввидите 'LBPH' или 'Fisher'")
        return

    face_samples, face_ids = collect_faces_and_ids()
    if len(face_samples) == 0:
        print("Нет изображений для обучения! Добавьте хотя бы одну папку с фото и повторите.")
        return

    print(f"Найдено {len(face_samples)} изображений, начинаем обучение...")
    if algorithm == 'LBPH':
        recognizer.train(face_samples, np.array(face_ids))
    else:
        unique_ids = np.unique(face_ids)
        if len(unique_ids) < 2:
            print("Не удалось обучить модель FisherFace. Требуется хотя бы два класса для распознования")
            return
        recognizer.train(face_samples, np.array(face_ids))
    recognizer.save(trained_file)
    print(f"Готово! Файл '{trained_file}' сохранён.")


def main():
    # Основная функция для запуска обучения модели
    algorithm = 'LBPH'  # Default algorithm
    if len(sys.argv) > 1:
        algorithm = sys.argv[1]
        if algorithm not in ['LBPH', 'Fisher']:
            print("Этот алгоритм не поддерживается! Ввидите 'LBPH' или 'Fisher'")
            return

    trained_file = TRAINED_FILE_LBPH if algorithm == 'LBPH' else TRAINED_FILE_FISHER
    if not os.path.exists(trained_file):
        print(f"Файл '{trained_file}' не найден. Создаём его впервые...")
        train_model(algorithm)
    else:
        print(f"Файл '{trained_file}' уже существует, но мы можем пересобрать модель.")
        ans = input("Пересоздать модель (да/нет)?").strip().lower()
        if ans in ["да", "yes", "y"]:
            train_model(algorithm)
        else:
            print("Операция отменена.")


if __name__ == "__main__":
    import sys
    main()
