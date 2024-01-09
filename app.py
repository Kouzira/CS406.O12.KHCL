import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import playsound
import cv2

class DrowsyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("600x650")
        self.title("Drowsy Boi 4.0")

        # Tạo các widget
        self.create_widgets()

    def create_widgets(self):
        # Khung video
        vid_frame = tk.Frame(self, height=480, width=600)
        vid_frame.pack()
        self.vid_label = tk.Label(vid_frame)
        self.vid_label.pack()

        # Thêm Label để hiển thị thông tin từ terminal
        self.info_label_yolo = tk.Label(self, text="", font=("Helvetica", 12))
        self.info_label_yolo.pack()

        # Tải mô hình YOLO
        self.model = YOLO("D:/Drowsiness_Detection/runs/detect/yolov8n_v8_50e2/weights/best.pt")

        # Tải tệp âm thanh
        self.sound_file = "D:/Drowsiness_Detection/nhac-chuong-tieng-coi-canh-bao.wav"

        # Bắt đầu luồng video
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()

    def update_video(self):
        # Logic cập nhật video của bạn ở đây
        cap = cv2.VideoCapture(0)

        while True:
            ret, original_frame = cap.read()

            # Tạo một bản sao của frame để hiển thị mà không ảnh hưởng đến dự đoán
            display_frame = original_frame.copy()

            # Gọi phương thức dự đoán của mô hình YOLO
            results = self.model.predict(original_frame, show=False)
            


            # Kiểm tra nhãn buồn ngủ và phát âm thanh
            threshold = 0.3
            drowsy_detected = False

            # Check if results is not None
            if results is not None:
                # Check for drowsy label in results
                for result in results:
                    if result is not None and result.names is not None and result.probs is not None:
                        for name, prob in zip(result.names, result.probs):
                            if prob > threshold and name == "drowsy":
                                playsound.playsound(self.sound_file)
                                drowsy_detected = True
                                print("Đã có âm thanh")

            # Hiển thị frame trên giao diện
            if ret:
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(Image.fromarray(display_frame))
                self.vid_label.config(image=photo)
                self.vid_label.image = photo

            # Cập nhật thông số từ terminal lên self.info_label_yolo
            if results is not None and len(results) > 0:
                speed_info = results[0].speed
                info_text = "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms postprocess per image at shape {}".format(
                    speed_info["preprocess"],
                    speed_info["inference"],
                    speed_info["postprocess"],
                    results[0].orig_shape
                )
                self.info_label_yolo.config(text=info_text)

            # Cập nhật video trong một vòng lặp
            self.update()

        # Ngừng quay video khi đóng ứng dụng
        cap.release()

# Chạy ứng dụng
if __name__ == "__main__":
    app = DrowsyApp()
    app.mainloop()