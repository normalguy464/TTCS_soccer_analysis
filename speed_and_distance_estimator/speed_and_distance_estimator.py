import sys

import cv2

sys.path.append('../')
from utils import measure_distance, get_foot_position

class SpeedAndDistanceEstimator:
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == 'ball' or object == 'referees':
                continue
                
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)
                
                # Thu thập tất cả ID
                all_ids_in_window = set()
                for f in range(frame_num, last_frame + 1):
                    if f < len(object_tracks):
                        all_ids_in_window.update(object_tracks[f].keys())
                
                for track_id in all_ids_in_window:
                    # Tìm vị trí đầu và cuối, ưu tiên position_transformed
                    start_position = None
                    end_position = None
                    start_frame = None
                    end_frame = None
                    
                    # Tìm điểm đầu tiên hợp lệ
                    for f in range(frame_num, last_frame + 1):
                        if f >= len(object_tracks) or track_id not in object_tracks[f]:
                            continue
                        
                        # Ưu tiên position_transformed, fallback về position_adjusted, cuối cùng là position
                        pos = (object_tracks[f][track_id].get('position_transformed') or 
                            object_tracks[f][track_id].get('position_adjusted') or 
                            object_tracks[f][track_id].get('position'))
                        
                        if pos is not None:
                            start_position = pos
                            start_frame = f
                            break
                            
                    # Tìm điểm cuối cùng hợp lệ
                    for f in range(last_frame, frame_num - 1, -1):
                        if f >= len(object_tracks) or track_id not in object_tracks[f]:
                            continue
                        
                        pos = (object_tracks[f][track_id].get('position_transformed') or 
                            object_tracks[f][track_id].get('position_adjusted') or 
                            object_tracks[f][track_id].get('position'))
                        
                        if pos is not None:
                            end_position = pos
                            end_frame = f
                            break

                    # Tính toán nếu có đủ dữ liệu
                    if start_position and end_position and start_frame != end_frame:
                        distance_covered = measure_distance(start_position, end_position)
                        
                        # Sử dụng hệ số chuyển đổi khác nhau tùy theo loại position
                        if object_tracks[start_frame][track_id].get('position_transformed'):
                            # Đã chuyển đổi sang mét thực tế
                            scale_factor = 1.0
                        else:
                            # Vẫn là pixel, cần chuyển đổi (ước lượng)
                            scale_factor = 0.1  # 1 pixel ≈ 0.1 mét
                        
                        distance_covered *= scale_factor
                        time_elapsed = (end_frame - start_frame) / self.frame_rate
                        
                        if time_elapsed > 0:
                            speed_meter_per_second = distance_covered / time_elapsed
                            speed_km_per_hour = speed_meter_per_second * 3.6
                            
                            if object not in total_distance:
                                total_distance[object] = {}
                            if track_id not in total_distance[object]:
                                total_distance[object][track_id] = 0
                                
                            total_distance[object][track_id] += distance_covered
                            
                            # Gán cho tất cả frame trong cửa sổ
                            for f in range(frame_num, last_frame + 1):
                                if f < len(object_tracks) and track_id in object_tracks[f]:
                                    tracks[object][f][track_id]['speed'] = speed_km_per_hour
                                    tracks[object][f][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        if speed is None:
                            continue

                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40
                        position[0] -= 30

                        position = tuple(map(int, position))
                        # Chỉ hiển thị vận tốc, bỏ dòng hiển thị distance
                        cv2.putText(frame, f"{speed:.1f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        # Đã xóa dòng hiển thị distance
            output_frames.append(frame)

        return output_frames
    