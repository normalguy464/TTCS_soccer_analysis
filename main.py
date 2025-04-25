from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
import cv2
import numpy as np
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator

def main():

    # Key:
    video_name = 'test (1).mp4' #new video mp4   

    #Read video
    video_frames = read_video(f'input_video/{video_name}')

    #Initialize tracker
    tracker = Tracker('final_model/best.pt')

    tracks = tracker.get_object_track(video_frames, 
                                      read_from_stub=True, 
                                      stub_path= 'stubs/track_stubs.pkl')
    
    #get object possition
    tracker.add_possition_to_tracks(tracks)



    # Camera Movement Estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement.pkl')

    camera_movement_estimator.add_adjust_position_to_tracks(tracks, camera_movement_per_frame)

    #Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])



    #team assigner
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])


    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]



    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        if any(np.isnan(ball_bbox)):  # Kiểm tra nếu ball_bbox chứa NaN
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])  # Sử dụng giá trị cuối cùng nếu danh sách không rỗng
            else:
                team_ball_control.append(-1)  # Giá trị mặc định nếu danh sách rỗng (ví dụ: -1 để biểu thị không có đội nào kiểm soát bóng)
            continue

        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])  # Sử dụng giá trị cuối cùng nếu danh sách không rỗng
            else:
                team_ball_control.append(-1)  # Giá trị mặc định nếu danh sách rỗng

    team_ball_control = np.array(team_ball_control)    
        

    
    #Draw tracks on video frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    #draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    #Save video
    save_video(output_video_frames, 'output_video/output_video.avi')

if __name__ == "__main__":
    main()