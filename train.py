from ultralytics import YOLO

def main():
    print("Загрузка базовой модели YOLOv8n...")
    model = YOLO('yolov8n.pt')

    print("Начинаем обучение на нашем датасете...")
    try:
        results = model.train(
            data='data.yaml',  # Путь к файлу конфигурации датасета
            epochs=3,         # Количество эпох (проходов по данным). Для теста можно 10.
            imgsz=640,         # Размер изображения (640 - стандарт)
            plots=True,        # Строить графики обучения
            project='cow_project', # Папка, куда сохранятся результаты
            name='train_run'   # Имя текущего запуска
        )
        print("Обучение успешно завершено!")
        print("Ваши веса модели находятся здесь: cow_project/train_run/weights/best.pt")
    except Exception as e:
        print(f"Ошибка при обучении (возможно, нет датасета или data.yaml): {e}")

if __name__ == "__main__":
    main()