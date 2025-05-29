from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.append('../')
from utils.bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill().ffill()
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    
    def interpolate_missing_positions(self, tracks):
        for object_name, object_tracks in tracks.items():
            if object_name == 'ball' or object_name == 'referees':
                continue
                
            # Thu thập tất cả track_id
            all_track_ids = set()
            for frame_tracks in object_tracks:
                all_track_ids.update(frame_tracks.keys())
            
            # Nội suy cho từng track_id
            for track_id in all_track_ids:
                # Tìm tất cả frame có track_id này
                positions = {}
                for frame_num, frame_tracks in enumerate(object_tracks):
                    if track_id in frame_tracks:
                        pos = (frame_tracks[track_id].get('position_transformed') or
                            frame_tracks[track_id].get('position_adjusted') or
                            frame_tracks[track_id].get('position'))
                        if pos is not None:
                            positions[frame_num] = pos
                
                # Nội suy cho các frame bị thiếu
                frame_numbers = sorted(positions.keys())
                if len(frame_numbers) < 2:
                    continue
                    
                for i in range(len(frame_numbers) - 1):
                    start_frame = frame_numbers[i]
                    end_frame = frame_numbers[i + 1]
                    
                    # Nếu có khoảng trống nhỏ (≤ 10 frames)
                    if end_frame - start_frame <= 10 and end_frame - start_frame > 1:
                        start_pos = positions[start_frame]
                        end_pos = positions[end_frame]
                        
                        # Nội suy tuyến tính
                        for frame_num in range(start_frame + 1, end_frame):
                            if frame_num < len(object_tracks):
                                alpha = (frame_num - start_frame) / (end_frame - start_frame)
                                interpolated_pos = [
                                    start_pos[0] + alpha * (end_pos[0] - start_pos[0]),
                                    start_pos[1] + alpha * (end_pos[1] - start_pos[1])
                                ]
                                
                                # Tạo track entry nếu chưa có
                                if track_id not in object_tracks[frame_num]:
                                    # Copy từ frame gần nhất
                                    if track_id in object_tracks[start_frame]:
                                        object_tracks[frame_num][track_id] = object_tracks[start_frame][track_id].copy()
                                    else:
                                        object_tracks[frame_num][track_id] = {}
                                
                                # Gán vị trí nội suy
                                if 'position_transformed' in object_tracks[start_frame][track_id]:
                                    object_tracks[frame_num][track_id]['position_transformed'] = interpolated_pos
                                elif 'position_adjusted' in object_tracks[start_frame][track_id]:
                                    object_tracks[frame_num][track_id]['position_adjusted'] = interpolated_pos
                                else:
                                    object_tracks[frame_num][track_id]['position'] = interpolated_pos

    def detect_frames(self, frames):
        batch_size = 25
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_track(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Lưu thông tin goalkeeper
            goalkeepers = {}
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    goalkeepers[object_ind] = True
                else:
                    goalkeepers[object_ind] = False
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for i, frame_detection in enumerate(detection_with_tracks):
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox,
                        "is_goalkeeper": goalkeepers.get(i, False)
                    }
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(frame,
                    (x_center, y2),
                    axes=(int(width), int(0.35 * width)),
                    angle=0,
                    startAngle=-45,
                    endAngle=235,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4)
        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2)
        return frame

    def draw_triangle(self, frame, bbox, color):
        if any(np.isnan(bbox)):
            return frame
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control, tracks):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_num_frames = np.sum(team_ball_control_till_frame == 2)
        total = team_1_num_frames + team_2_num_frames

        team1_color = None
        team2_color = None
        for player_id, player in tracks["players"][frame_num].items():
            if player.get("team") == 1 and team1_color is None:
                team1_color = tuple(map(int, player.get("team_color", (0, 0, 255))))
            elif player.get("team") == 2 and team2_color is None:
                team2_color = tuple(map(int, player.get("team_color", (255, 0, 0))))
            if team1_color and team2_color:
                break

        team1_color = team1_color or (0, 0, 255)
        team2_color = team2_color or (255, 0, 0)

        if total > 0:
            team_1 = team_1_num_frames / total
            team_2 = team_2_num_frames / total
            cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, team1_color, 3)
            cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, team2_color, 3)
        else:
            cv2.putText(frame, "No ball possession data", (1400, 925),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), track_id)

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, tracks)
            output_video_frames.append(frame)
        return output_video_frames
