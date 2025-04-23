from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
import cv2
def main():
    #Read video
    video_frames = read_video('input_video/08fd33_4.mp4')

    #Initialize tracker
    tracker = Tracker('final_model/best.pt')

    tracks = tracker.get_object_track(video_frames, 
                                      read_from_stub=True, 
                                      stub_path= 'stubs/track_stubs.pkl')
    
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

    # for frame_num, goalkeeper_track in enumerate(tracks['goalkeepers']):
    #     for player_id, track in goalkeeper_track.items():
    #         team = team_assigner.get_player_team(video_frames[frame_num],   
    #                                              track['bbox'],
    #                                              player_id)
    #         tracks['goalkeepers'][frame_num][player_id]['team'] = team 
    #         tracks['goalkeepers'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    #Draw tracks on video frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    #Save video
    save_video(output_video_frames, 'output_video/output_video.avi')

if __name__ == "__main__":
    main()