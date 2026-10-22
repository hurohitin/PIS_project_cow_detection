import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
from ultralytics import YOLO

# --- НАСТРОЙКИ ---
# Путь к вашей обученной модели (проверьте название папки train_run-3)
MODEL_PATH = r'C:\Users\admin\PycharmProjects\PIS_project\runs\detect\cow_project\train_run-3\weights\best.pt'


class CowDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Детекция коров в коровнике")
        self.root.geometry("900x750")
        self.root.configure(bg="#f0f0f0")

        # Ссылка на текущую картинку (защита от удаления сборщиком мусора)
        self.current_image_ref = None

        self.load_model()
        self.setup_ui()

    def load_model(self):
        """Загрузка нейросети при старте программы"""
        if os.path.exists(MODEL_PATH):
            print(f"[INFO] Загрузка обученной модели: {MODEL_PATH}")
            try:
                self.model = YOLO(MODEL_PATH)
            except Exception as e:
                print(f"[ERROR] Ошибка загрузки модели: {e}")
                self.model = YOLO('yolov8n.pt')
        else:
            print("[WARN] Обученная модель не найдена! Использую стандартную YOLOv8n.")
            self.model = YOLO('yolov8n.pt')

    def setup_ui(self):
        """Создание элементов интерфейса"""
        tk.Label(self.root, text="Система компьютерного зрения", font=("Arial", 20, "bold"), bg="#f0f0f0").pack(pady=10)
        tk.Label(self.root, text="Детекция коров в коровнике", font=("Arial", 14), bg="#f0f0f0", fg="#555").pack(pady=5)

        self.btn_load = tk.Button(self.root, text="Загрузить фото", font=("Arial", 14), bg="#4CAF50", fg="white",
                                  activebackground="#45a049", padx=20, pady=10, command=self.load_image)
        self.btn_load.pack(pady=20)

        # Контейнер для картинки (фиксированный размер)
        self.image_frame = tk.Frame(self.root, bg="#e0e0e0", width=860, height=480)
        self.image_frame.pack(pady=10, padx=20)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, bg="#e0e0e0", text="Здесь появится изображение",
                                    font=("Arial", 12), fg="#888")
        self.image_label.pack(fill="both", expand=True)

        self.result_label = tk.Label(self.root, text="", font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#d32f2f")
        self.result_label.pack(pady=10)

        self.info_label = tk.Label(self.root, text="", font=("Arial", 10), bg="#f0f0f0", fg="#666")
        self.info_label.pack(pady=5)

    def read_image_safe(self, file_path):
        """Универсальное чтение картинки (3 способа подряд)"""

        # Способ 1: np.fromfile + imdecode (для кириллицы в пути и webp)
        try:
            data = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                print("[INFO] Файл прочитан способом 1 (imdecode)")
                return img
        except Exception as e:
            print(f"[WARN] Способ 1 не сработал: {e}")

        # Способ 2: классический cv2.imread
        try:
            img = cv2.imread(file_path)
            if img is not None:
                print("[INFO] Файл прочитан способом 2 (imread)")
                return img
        except Exception as e:
            print(f"[WARN] Способ 2 не сработал: {e}")

        # Способ 3: Pillow (всеядный, для webp, tiff, cmyk)
        try:
            pil_img = Image.open(file_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            print("[INFO] Файл прочитан способом 3 (Pillow)")
            return img
        except Exception as e:
            print(f"[ERROR] Способ 3 (Pillow) не сработал: {e}")

        return None

    def load_image(self):
        """Обработка нажатия на кнопку"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Все изображения", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                       ("Все файлы", "*.*")]
        )

        if not file_path:
            return

        print(f"\n[INFO] Выбран файл: {file_path}")
        self.result_label.config(text="Обработка...", fg="black")
        self.info_label.config(text="")
        self.root.update()

        # 1. Чтение картинки
        img = self.read_image_safe(file_path)
        if img is None:
            messagebox.showerror("Ошибка чтения",
                                 f"Не удалось прочитать файл:\n{os.path.basename(file_path)}\n\n"
                                 "Файл поврежден или не является изображением.")
            self.result_label.config(text="Ошибка чтения файла", fg="#d32f2f")
            return

        h, w = img.shape[:2]
        self.info_label.config(text=f"Исходное изображение: {w}x{h} px")
        print(f"[INFO] Картинка прочитана: {w}x{h}")

        # 2. Детекция
        try:
            results = self.model(img, conf=0.3)
        except Exception as e:
            messagebox.showerror("Ошибка модели", f"Ошибка при работе нейросети:\n{e}")
            self.result_label.config(text="Ошибка модели", fg="#d32f2f")
            return

        # 3. Рисуем рамки и считаем
        result_img_bgr = results[0].plot()
        count = len(results[0].boxes)
        print(f"[INFO] Обнаружено объектов: {count}")

        # 4. Конвертация BGR -> RGB -> PIL
        result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(result_img_rgb)

        # 5. Масштабирование под размер окна
        frame_w = self.image_frame.winfo_width() or 860
        frame_h = self.image_frame.winfo_height() or 480
        img_pil.thumbnail((frame_w - 10, frame_h - 10), Image.LANCZOS)

        # 6. Отображение в Tkinter с сохранением ссылки
        try:
            img_tk = ImageTk.PhotoImage(img_pil)
        except Exception as e:
            print(f"[ERROR] Ошибка конвертации в PhotoImage: {e}")
            messagebox.showerror("Ошибка отображения", f"Не удалось отобразить картинку:\n{e}")
            return

        self.current_image_ref = img_tk
        self.image_label.config(image=self.current_image_ref, text="")
        self.image_label.image = self.current_image_ref

        # 7. Вывод результата
        if count > 0:
            self.result_label.config(text=f"Обнаружено коров: {count}", fg="#2e7d32")
        else:
            self.result_label.config(text="Коровы не обнаружены", fg="#d32f2f")


if __name__ == "__main__":
    root = tk.Tk()
    app = CowDetectorApp(root)
    root.mainloop()