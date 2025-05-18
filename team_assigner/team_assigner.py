from sklearn.cluster import KMeans
import numpy as np

from utils import get_center_of_bbox


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2), :]
        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            if player_detection.get("is_goalkeeper", False):
                continue
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        if len(player_colors) == 0:
            print("Warning: No non-goalkeeper players detected for team color assignment.")
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        print("Team 1 color:", self.team_colors[1])
        print("Team 2 color:", self.team_colors[2])

    def get_player_team(self, frame, player_bbox, player_id, is_goalkeeper=False):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if is_goalkeeper:
            # Gán đội cho thủ môn dựa trên vị trí
            team_id = self.determine_team_by_position(player_bbox, frame.shape)
        else:
            # Gán đội cho cầu thủ bình thường dựa trên màu áo
            player_color = self.get_player_color(frame, player_bbox)
            if self.kmeans is None:
                team_id = 1  # Fallback if kmeans is not initialized
            else:
                team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        self.player_team_dict[player_id] = team_id
        return team_id

    def determine_team_by_position(self, bbox, frame_shape):
        # Giả định: Thủ môn đội 1 ở bên trái (x < frame_width / 2), đội 2 ở bên phải
        x_center, _ = get_center_of_bbox(bbox)
        frame_width = frame_shape[1]
        if x_center < frame_width / 2:
            return 1  # Đội 1
        else:
            return 2  # Đội 2