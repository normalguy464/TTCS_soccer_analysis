import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance
import numpy as np

class PlayerBallAssigner():
   def __init__(self):
      self.max_player_ball_distance = 70

   # def assign_ball_to_player(self,players,ball_bbox):
   #    ball_position = get_center_of_bbox(ball_bbox)

   #    miniumum_distance = 99999
   #    assigned_player = -1

   #    for player_id, player in players.items():
   #       player_bbox = player['bbox']

   #       distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
   #       distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
   #       distance = min(distance_left, distance_right)

   #       if distance < self.max_player_ball_distance:
   #          if distance < miniumum_distance:
   #             minimum_distance = distance
   #             assigned_player = player_id

   #    return assigned_player
   def assign_ball_to_player(self, players, ball_bbox):
    if any(np.isnan(ball_bbox)):  # Kiểm tra nếu bất kỳ giá trị nào trong ball_bbox là NaN
        print("Warning: ball_bbox contains NaN values:", ball_bbox)
        return -1  # Trả về -1 nếu ball_bbox không hợp lệ

    ball_position = get_center_of_bbox(ball_bbox)

    minimum_distance = 99999
    assigned_player = -1

    for player_id, player in players.items():
        player_bbox = player['bbox']

        distance_left = measure_distance(player_bbox[0:2], ball_position)
        distance_right = measure_distance(player_bbox[2:4], ball_position)
        distance = min(distance_left, distance_right)

        if distance < self.max_player_ball_distance and distance < minimum_distance:
            minimum_distance = distance
            assigned_player = player_id

    return assigned_player