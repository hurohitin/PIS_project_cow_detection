import cv2
import os
import sys
from ultralytics import YOLO


def run_detection(image_path):
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл '{image_path}' не найден!")
        return

    print(f"Обработка изображения: {image_path}")
    custom_model_path = r'C:\Users\admin\PycharmProjects\PIS_project\runs\detect\cow_project\train_run-3\weights\best.pt'

    if os.path.exists(custom_model_path):
        print("Используется ОБУЧЕННАЯ модель (best.pt)")
        model = YOLO(custom_model_path)
    else:
        print(
            "Внимание: Обученная модель не найдена. Используется стандартная YOLOv8n (она тоже знает, как выглядит корова).")
        model = YOLO('yolov8n.pt')

    results = model(image_path, conf=0.3)
    for result in results:

        annotated_img = result.plot()

        boxes = result.boxes
        cow_count = 0

        for box in boxes:
            cls_id = int(box.cls[0])

            cow_count += 1

        print(f"--> Обнаружено объектов на фото: {cow_count}")


        cv2.imshow('Cow Detection (Press any key to close)', annotated_img)
        cv2.waitKey(0)  # Ждем нажатия любой клавиши
        cv2.destroyAllWindows()


        output_filename = "result_" + os.path.basename(image_path)
        cv2.imwrite(output_filename, annotated_img)
        print(f"Результат сохранен в файл: {output_filename}")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:

        img_path = "test1.jpg"
        print("Путь к картинке не указан. Пытаюсь открыть 'test.jpg'...")

    run_detection(img_path)