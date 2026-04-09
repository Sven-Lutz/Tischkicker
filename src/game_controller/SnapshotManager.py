import cv2
import os
import logging
from datetime import datetime
import numpy as np


class SnapshotManager:
    """Manages saving goal snapshots."""
    
    def __init__(self, snapshot_dir: str = "../snapshots"):
        """
        :param snapshot_dir: Directory for goal snapshots
        """
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)
        logging.info(f"[SnapshotManager] Snapshot-Verzeichnis: {self.snapshot_dir}")
    
    def save_snapshot(self, frame: np.ndarray, team: str, scoreboard) -> None:
        """
        Saves a snapshot of the current frame.
        
        :param frame: The frame to save (already drawn with trajectory)
        :param team: Name of the team that scored
        :param scoreboard: ScoreBoard instance for the current score
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        score = scoreboard.get_score_string().replace(" : ", "-")
        filename = f"goal_{team}_{score}_{timestamp}.png"
        filepath = os.path.join(self.snapshot_dir, filename)
        
        cv2.imwrite(filepath, frame)
        logging.info(f"[SnapshotManager] Snapshot gespeichert: {filepath}")
