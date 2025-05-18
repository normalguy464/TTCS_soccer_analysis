import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import os
import threading
import numpy as np
from PIL import Image, ImageTk

# Import các module từ project hiện tại
from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator


class FootballAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phân tích video bóng đá")
        self.root.geometry("1000x700")

        # Các biến
        self.video_path = ""
        self.output_path = ""
        self.is_processing = False
        self.video_playing = False
        self.cap = None

        # Tạo giao diện
        self.create_widgets()

    def create_widgets(self):
        # Frame chính
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame chọn và xử lý video
        control_frame = ttk.LabelFrame(main_frame, text="Tùy chọn", padding=10)
        control_frame.pack(fill=tk.X, pady=5)

        # Nút chọn video
        ttk.Button(control_frame, text="Chọn video", command=self.select_video).grid(row=0, column=0, padx=5, pady=5)
        self.path_label = ttk.Label(control_frame, text="Chưa chọn video")
        self.path_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # Nút xử lý video
        self.process_btn = ttk.Button(control_frame, text="Xử lý video", command=self.process_video)
        self.process_btn.grid(row=1, column=0, padx=5, pady=5)

        # Thanh tiến trình
        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # Trạng thái
        self.status_label = ttk.Label(control_frame, text="Trạng thái: Sẵn sàng")
        self.status_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

        # Frame hiển thị video
        video_frame = ttk.LabelFrame(main_frame, text="Video", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Canvas để hiển thị video
        self.canvas = tk.Canvas(video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Frame điều khiển video
        player_control = ttk.Frame(video_frame)
        player_control.pack(fill=tk.X, pady=5)

        self.play_btn = ttk.Button(player_control, text="Phát", command=self.play_video, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.pause_btn = ttk.Button(player_control, text="Tạm dừng", command=self.pause_video, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.restart_btn = ttk.Button(player_control, text="Xem lại", command=self.restart_video, state=tk.DISABLED)
        self.restart_btn.pack(side=tk.LEFT, padx=5)

    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video files", "*.mp4 *.avi")]
        )

        if file_path:
            self.video_path = file_path
            self.path_label.config(text=os.path.basename(file_path))
            self.status_label.config(text="Trạng thái: Đã chọn video")

    def process_video(self):
        if not self.video_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn video trước khi xử lý")
            return

        if self.is_processing:
            return

        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED)

        # Bắt đầu xử lý trong thread riêng biệt
        processing_thread = threading.Thread(target=self.run_processing)
        processing_thread.daemon = True
        processing_thread.start()

    def run_processing(self):
        try:
            # Chuẩn bị đường dẫn đầu ra
            video_name = os.path.basename(self.video_path)
            os.makedirs('input_video', exist_ok=True)
            os.makedirs('output_video', exist_ok=True)

            # Cập nhật trạng thái
            self.update_status("Đang đọc video...", 10)

            # Đọc video
            video_frames = read_video(self.video_path)

            # Khởi tạo tracker
            self.update_status("Đang khởi tạo bộ theo dõi...", 20)
            tracker = Tracker('final_model/best.pt')

            # Theo dõi đối tượng
            self.update_status("Đang theo dõi đối tượng...", 30)
            tracks = tracker.get_object_track(video_frames, read_from_stub=False,
                                                  stub_path='stubs/track_stubs.pkl')

            # Xác định vị trí đối tượng
            self.update_status("Đang tính toán vị trí...", 40)
            tracker.add_position_to_tracks(tracks)

            # Ước tính chuyển động camera
            self.update_status("Đang ước tính chuyển động camera...", 50)
            camera_movement_estimator = CameraMovementEstimator(video_frames[0])

            camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
                    video_frames, read_from_stub=False, stub_path='stubs/camera_movement.pkl')

            camera_movement_estimator.add_adjust_position_to_tracks(tracks, camera_movement_per_frame)

            # Chuyển đổi góc nhìn
            self.update_status("Đang chuyển đổi góc nhìn...", 60)
            view_transformer = ViewTransformer()
            view_transformer.add_transformed_position_to_tracks(tracks)

            # Nội suy vị trí bóng
            tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

            # Tính toán tốc độ và khoảng cách
            self.update_status("Đang tính toán tốc độ và khoảng cách...", 70)
            speed_and_distance_estimator = SpeedAndDistanceEstimator()
            speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

            # Phân công đội
            self.update_status("Đang phân chia đội...", 80)
            team_assigner = TeamAssigner()
            team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

            for frame_num, player_track in enumerate(tracks['players']):
                for player_id, track in player_track.items():
                    team = team_assigner.get_player_team(video_frames[frame_num],
                                                         track['bbox'],
                                                         player_id,
                                                         is_goalkeeper=track.get('is_goalkeeper',
                                                                                 False))  # Truyền thông tin thủ môn
                    tracks['players'][frame_num][player_id]['team'] = team
                    tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

            # Xác định kiểm soát bóng
            self.update_status("Đang xác định kiểm soát bóng...", 85)
            player_assigner = PlayerBallAssigner()
            team_ball_control = []
            for frame_num, player_track in enumerate(tracks['players']):
                ball_bbox = tracks['ball'][frame_num][1]['bbox']
                if any(np.isnan(ball_bbox)):
                    if team_ball_control:
                        team_ball_control.append(team_ball_control[-1])
                    else:
                        team_ball_control.append(-1)
                    continue

                assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

                if assigned_player != -1:
                    tracks['players'][frame_num][assigned_player]['has_ball'] = True
                    team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
                else:
                    if team_ball_control:
                        team_ball_control.append(team_ball_control[-1])
                    else:
                        team_ball_control.append(-1)

            team_ball_control = np.array(team_ball_control)

            # Vẽ chú thích
            self.update_status("Đang vẽ chú thích...", 90)
            output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

            # Vẽ chuyển động camera
            output_video_frames = camera_movement_estimator.draw_camera_movement(
                output_video_frames, camera_movement_per_frame)

            # Vẽ tốc độ và khoảng cách
            speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

            # Lưu video
            self.update_status("Đang lưu video...", 95)
            self.output_path = os.path.join('output_video', f'output_{video_name}')
            save_video(output_video_frames, self.output_path)

            # Hoàn thành
            self.update_status("Xử lý hoàn tất!", 100)

            # Cập nhật giao diện
            self.root.after(0, self.on_processing_complete)


        except Exception as e:
            self.show_error_later(e)

    def show_error_later(self, error):
        self.root.after(0, lambda: self.show_error(str(error)))

    def update_status(self, message, progress_value):
        self.root.after(0, lambda: self.status_label.config(text=f"Trạng thái: {message}"))
        self.root.after(0, lambda: self.progress.config(value=progress_value))

    def on_processing_complete(self):
        self.is_processing = False
        self.process_btn.config(state=tk.NORMAL)

        # Kích hoạt các nút điều khiển video
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.NORMAL)
        self.restart_btn.config(state=tk.NORMAL)

        messagebox.showinfo("Thành công", "Video đã được xử lý thành công!")

        # Tự động tải video đầu ra
        self.load_output_video()

    def show_error(self, error_message):
        self.is_processing = False
        self.process_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Trạng thái: Xảy ra lỗi")
        messagebox.showerror("Lỗi", f"Xảy ra lỗi trong quá trình xử lý: {error_message}")

    def load_output_video(self):
        if os.path.exists(self.output_path):
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.output_path)
            self.play_video()

    def play_video(self):
        if not self.cap:
            return

        self.video_playing = True
        self.update_frame()

    def update_frame(self):
        if not self.video_playing:
            return

        ret, frame = self.cap.read()
        if ret:
            # Chuyển đổi khung hình để hiển thị
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Điều chỉnh kích thước để vừa với canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width > 0 and canvas_height > 0:
                frame = cv2.resize(frame, (canvas_width, canvas_height))

            # Hiển thị khung hình
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            # Lập lịch khung hình tiếp theo
            self.root.after(30, self.update_frame)
        else:
            # Hết video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_playing = False

    def pause_video(self):
        self.video_playing = False

    def restart_video(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_playing = True
            self.update_frame()


if __name__ == "__main__":
    root = tk.Tk()
    app = FootballAnalysisApp(root)
    root.mainloop()