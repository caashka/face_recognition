# face_recognition.py

import cv2
import numpy as np
import os
import datetime
import json
import argparse

USERS_FILE = "users.json"
TRAINED_FILE_LBPH = "trained_faces_lbph.yml"
TRAINED_FILE_FISHER = "trained_faces_fisher.yml"


def load_users():
    """
    Считываем users.json и конвертируем ключи (ID) из str в int,
    чтобы они совпадали с типом predicted_id, который возвращает recognizer.predict().
    """
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    user_dict = {}
    # Преобразуем ключи словаря к int, а значения (имена) оставляем нетронутыми
    for k, v in data.items():
        try:
            user_dict[int(k)] = v
        except ValueError:
            # если почему-то ключ не число, можно пропустить или обработать иначе
            pass
    return user_dict


# Загружаем словарь имен (ID -> Имя)
USER_NAMES = load_users()


def process_video_stream(video_source, delay, algorithm):
    # Загружаем каскад Хаара
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Инициализируем распознаватель и подгружаем обученную выбранную модель
    if algorithm == 'LBPH':
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        trained_file = TRAINED_FILE_LBPH
    elif algorithm == 'Fisher':
        recognizer = cv2.face.FisherFaceRecognizer_create()
        trained_file = TRAINED_FILE_FISHER
    else:
        print("Этот алгоритм не поддерживается! Ввидите 'LBPH' или 'Fisher'")
        return

    if os.path.exists(trained_file):
        recognizer.read(trained_file)
    else:
        print(f"Ошибка: не найден файл '{trained_file}'. Сначала запустите тренеровку (trainer.py)")
        return

    # Открываем видеопоток
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Не удалось открыть видеопоток {video_source}. Проверьте его доступность и права.")
        return

    # Файл для логов (открываем в режиме добавления)
    log_file = open("face_log.txt", "a", encoding="utf-8")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с видеопотока.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Вырезаем лицо из кадра
            roi_gray = gray[y:y + h, x:x + w]

            # Пытаемся распознать
            try:
                if algorithm == 'LBPH':
                    predicted_id, confidence = recognizer.predict(roi_gray)
                else:
                    # уменьшаем размер изображения дла FisherFaces
                    roi_gray = cv2.resize(roi_gray, (100, 100))
                    print(roi_gray.shape)
                    predicted_id = recognizer.predict(roi_gray)
                    confidence = recognizer.predict_proba(roi_gray)
            except cv2.error as e:
                print(f"Ошибка при распознавании лица: {e}")
                continue

            # Преобразуем confidence в некий "процент уверенности"
            # (чем меньше confidence у LBPH, тем выше реальная уверенность)
            confidence_text = max(0, min(100, int(100 - confidence))) if algorithm == 'LBPH' else int(confidence * 100)

            # Проверяем, есть ли такой ID в словаре и достаточно ли высокий "процент"
            if predicted_id in USER_NAMES and confidence_text > 15:
                name = USER_NAMES[predicted_id]
                label = f"{name} ({confidence_text}%)"
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{now} - Распознан: {name}, Уверенность: {confidence_text}%\n")
                color = (0, 255, 0)
            else:
                label = f"Неопознанный  ({confidence_text}%)"
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{now} - Неопознанный, Уверенность: {confidence_text}%\n")
                color = (0, 0, 255)

            # Рисуем рамку и подпись на кадре
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)

        # Нажмите 'q' для выхода
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Закрываем ресурсы
    cap.release()
    cv2.destroyAllWindows()
    log_file.close()


def main():
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument("--stream", type=str, help="URL of the video stream")
    parser.add_argument("--delay", type=int, default=30, help="Delay between frames in milliseconds")
    parser.add_argument("--algorithm", type=str, help="Algorithm to use: LBPH or Fisher")
    args = parser.parse_args()

    # Загружаем каскад Хаара
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Инициализируем распознаватель и подгружаем выбранную обученную модель
    if args.algorithm not in ['LBPH', 'Fisher']:
        print("Этот алгоритм не поддерживается! Ввидите 'LBPH' или 'Fisher'")
        return
    if args.algorithm == 'LBPH':
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        trained_file = TRAINED_FILE_LBPH
    elif args.algorithm == 'Fisher':
        recognizer = cv2.face.FisherFaceRecognizer_create()
        trained_file = TRAINED_FILE_FISHER

    if os.path.exists(trained_file):
        recognizer.read(trained_file)
    else:
        print(f"Ошибка: не найден файл '{trained_file}'. Сначала запустите тренеровку (trainer.py)")
        return

    if args.stream:
        process_video_stream(args.stream, args.delay, args.algorithm)
    else:
        # Открываем веб-камеру
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Не удалось получить кадр с видеопотока.")
            return

        # Файл для логов (открываем в режиме добавления)
        log_file = open("face_log.txt", "a", encoding="utf-8")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Не удалось получить кадр с видеопотока.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Вырезаем лицо из кадра
                roi_gray = gray[y:y + h, x:x + w]

                # Пытаемся распознать
                try:
                    if args.algorithm == 'LBPH':
                        predicted_id, confidence = recognizer.predict(roi_gray)
                    else:
                        roi_gray = cv2.resize(roi_gray, (100, 100))
                        predicted_id, confidence = recognizer.predict(roi_gray)
                except cv2.error as e:
                    print(f"Ошибка при распознавании лица: {e}")
                    continue

                # Преобразуем confidence в некий "процент уверенности"
                # (чем меньше confidence у LBPH, тем выше реальная уверенность)
                confidence_text = max(0, min(100, int(100 - confidence)))

                # Проверяем, есть ли такой ID в словаре и достаточно ли высокий "процент"
                if predicted_id in USER_NAMES and confidence_text > 20:
                    name = USER_NAMES[predicted_id]
                    label = f"{name} ({confidence_text}%)"
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"{now} - Recognized: {name}, Уверенность: {confidence_text}%\n")
                    color = (0, 255, 0)
                else:
                    label = f"Неопознанный ({confidence_text}%)"
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"{now} - Неопознанный, Уверенность: {confidence_text}%\n")
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Рисуем рамку и подпись на кадре
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()


if __name__ == "__main__":
    main()
