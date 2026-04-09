from __future__ import annotations

import cv2
import numpy as np


def nothing(x):
    pass


class BallTracker:
    """
    Detects and tracks ball based on color in HSV space. Calculates position and speed.
    """

    def __init__(self, fps: float = 30.0, cm_per_pixel: float = 0.1):
        """
        :param fps:           Frames per second for velocity
        :param cm_per_pixel:  conversion factor pixels to centimeters
        """
        self.fps = fps
        self.cm_per_pixel = cm_per_pixel

        self.hsv_lower = np.array([5, 150, 150])
        self.hsv_upper = np.array([25, 255, 255])

        self.position: tuple[int, int] | None = None
        self.radius: int = 0
        self.speed_cm_s: float = 0.0

        self._prev_position: tuple[int, int] | None = None
        self._speed_history: list[float] = []

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> tuple[int, int] | None:
        """
        Process frame, detects ball and updates position and velocity accordingly.
        """
        self._prev_position = self.position

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.position = None
            self.radius = 0
            self.speed_cm_s = 0.0
            return None

        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            self.position = None
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        area = cv2.contourArea(largest)
        radius = (area / np.pi) ** 0.5

        if radius < 2:
            self.position = None
            return None

        self.position = (int(cx), int(cy))
        self.radius = int(radius)
        self._update_speed()
        return self.position

    def _update_speed(self) -> None:
        """Berechnet Geschwindigkeit aus Positionsdifferenz zwischen zwei Frames."""
        if self._prev_position is None or self.position is None:
            self.speed_cm_s = 0.0
            return

        dx = self.position[0] - self._prev_position[0]
        dy = self.position[1] - self._prev_position[1]
        pixel_dist = (dx ** 2 + dy ** 2) ** 0.5
        self.speed_cm_s = pixel_dist * self.cm_per_pixel * self.fps

        self._speed_history.append(self.speed_cm_s)
        if len(self._speed_history) > 100:
            self._speed_history.pop(0)

    def average_speed_cm_s(self) -> float:
        """Returns the average speed across all stored frames so far."""
        if not self._speed_history:
            return 0.0
        return sum(self._speed_history) / len(self._speed_history)

    def reset_speed_history(self) -> None:
        """Resets the speed history."""
        self._speed_history.clear()

    def draw(self, frame: np.ndarray) -> None:
        """Draws ball position and velocity in frame."""
        if self.position is None:
            return
        # Ball-Kreis
        cv2.circle(frame, self.position, self.radius, (0, 255, 0), 2)
        # Mittelpunkt
        cv2.circle(frame, self.position, 3, (0, 255, 0), -1)
        # Geschwindigkeits-Label
        cv2.putText(frame, f"{self.speed_cm_s:.1f} cm/s",
                    (self.position[0] + self.radius + 5, self.position[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ------------------------------------------------------------------
    # Kalibrierung
    # ------------------------------------------------------------------

    def _auto_calibrate_from_roi(
        self,
        camera,
        roi_center: tuple[int, int],
        roi_radius: int,
        frames: int = 30,
        padding: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sammelt HSV-Werte aus einer kreisförmigen ROI über mehrere Frames
        und berechnet daraus automatisch lower/upper Grenzen.

        :param camera:     Camera-Objekt
        :param roi_center: Mittelpunkt des Kreises (x, y)
        :param roi_radius: Radius des Kreises in Pixeln
        :param frames:     Anzahl Frames die gesammelt werden
        :param padding:    Toleranz-Puffer der auf die Perzentile addiert wird
        :return:           (hsv_lower, hsv_upper)
        """
        H, S, V = [], [], []
        collected = 0

        print(f"[BallTracker] Sammle {frames} Frames für Auto-Kalibrierung …")

        while collected < frames:
            ok, frame = camera.read_frame()
            if not ok:
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Kreisförmige Maske über ROI
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(mask, roi_center, roi_radius, 255, -1)

            roi_pixels = hsv[mask == 255]
            if len(roi_pixels) == 0:
                continue

            H.extend(roi_pixels[:, 0].tolist())
            S.extend(roi_pixels[:, 1].tolist())
            V.extend(roi_pixels[:, 2].tolist())

            collected += 1
            cv2.waitKey(1)

        lower = np.array([
            max(0,   int(np.percentile(H, 5))  - padding),
            max(0,   int(np.percentile(S, 5))  - padding),
            max(0,   int(np.percentile(V, 5))  - padding),
        ])
        upper = np.array([
            min(179, int(np.percentile(H, 95)) + padding),
            min(255, int(np.percentile(S, 95)) + padding),
            min(255, int(np.percentile(V, 95)) + padding),
        ])

        print(f"  hsv_lower = {lower}")
        print(f"  hsv_upper = {upper}")
        return lower, upper

    def hsv_trackbar(self) -> None:
        """Trackbar to choose HSV values."""
        cv2.namedWindow("HSV Settings")
        cv2.resizeWindow("HSV Settings", 400, 300)
        # Initialisiere Trackbars mit aktuellen Werten
        cv2.createTrackbar("H_Min", "HSV Settings", int(self.hsv_lower[0]), 179, nothing)
        cv2.createTrackbar("H_Max", "HSV Settings", int(self.hsv_upper[0]), 179, nothing)
        cv2.createTrackbar("S_Min", "HSV Settings", int(self.hsv_lower[1]), 255, nothing)
        cv2.createTrackbar("S_Max", "HSV Settings", int(self.hsv_upper[1]), 255, nothing)
        cv2.createTrackbar("V_Min", "HSV Settings", int(self.hsv_lower[2]), 255, nothing)
        cv2.createTrackbar("V_Max", "HSV Settings", int(self.hsv_upper[2]), 255, nothing)

    def _sync_trackbars_to_hsv(self) -> None:
        """Schreibt die aktuellen hsv_lower/upper Werte in die Trackbars."""
        cv2.setTrackbarPos("H_Min", "HSV Settings", int(self.hsv_lower[0]))
        cv2.setTrackbarPos("S_Min", "HSV Settings", int(self.hsv_lower[1]))
        cv2.setTrackbarPos("V_Min", "HSV Settings", int(self.hsv_lower[2]))
        cv2.setTrackbarPos("H_Max", "HSV Settings", int(self.hsv_upper[0]))
        cv2.setTrackbarPos("S_Max", "HSV Settings", int(self.hsv_upper[1]))
        cv2.setTrackbarPos("V_Max", "HSV Settings", int(self.hsv_upper[2]))

    def update_hsv_from_trackbar(self) -> None:
        """Reads current HSV settings from trackbar."""
        try:
            self.hsv_lower = np.array([
                cv2.getTrackbarPos("H_Min", "HSV Settings"),
                cv2.getTrackbarPos("S_Min", "HSV Settings"),
                cv2.getTrackbarPos("V_Min", "HSV Settings"),
            ])
            self.hsv_upper = np.array([
                cv2.getTrackbarPos("H_Max", "HSV Settings"),
                cv2.getTrackbarPos("S_Max", "HSV Settings"),
                cv2.getTrackbarPos("V_Max", "HSV Settings"),
            ])
        except cv2.error:
            pass

    def calibrate_hsv_interactive(self, camera, roi_radius: int = 30) -> None:
        """
        Interactive calibration procedure.
        
        :param camera: camera object.
        """
        print("[BallTracker] HSV-Kalibrierung gestartet.Drücke 'c' zum Kalibrieren . Drücke 'q' zum Beenden.")
        
        # Trackbars
        self.hsv_trackbar()
        cv2.namedWindow("Original")
        cv2.waitKey(100)

        # ROI-Kreis in der Bildmitte platzieren
        ok, first_frame = camera.read_frame()
        h, w = first_frame.shape[:2] if ok else (480, 640)
        roi_center = (w // 2, h // 2)

        while True:
            # Frame lesen
            ok, frame = camera.read_frame()
            if not ok:
                print("[BallTracker] Kein Frame verfügbar.")
                break
            
            # Aktuelle Trackbar-Werte übernehmen
            self.update_hsv_from_trackbar()
            
            # HSV konvertieren
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Maske mit aktuellen Werten erstellen
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

            mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
            # Ergebnis auf Original anwenden
            result = cv2.bitwise_and(frame, frame, mask=mask_cleaned)

            # ROI-Kreis einzeichnen
            cv2.circle(frame, roi_center, roi_radius, (0, 255, 255), 2)

            cv2.imshow("Original", frame)
            cv2.imshow("Mask", mask_cleaned)
            cv2.imshow("Result", result)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'): # Kalibrierung
                self.hsv_lower, self.hsv_upper = self._auto_calibrate_from_roi(
                    camera, roi_center, roi_radius
                )
                self._sync_trackbars_to_hsv()

            elif key == ord('q'):
                print(f"[BallTracker] Finale HSV-Werte:")
                print(f"  hsv_lower = {self.hsv_lower}")
                print(f"  hsv_upper = {self.hsv_upper}")
                break

        cv2.destroyWindow("Original")
        cv2.destroyWindow("Mask")
        cv2.destroyWindow("Result")
        cv2.destroyWindow("HSV Settings")