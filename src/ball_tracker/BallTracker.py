from __future__ import annotations

import cv2
import numpy as np


def nothing(x):
    pass


class BallTracker:
    """
    Erkennt und trackt den Ball per Farb-Segmentierung in jedem Frame.
    Berechnet Position, Geschwindigkeit
    """

    def __init__(self, fps: float = 30.0, cm_per_pixel: float = 0.1):
        """
        :param fps:           Frames pro Sekunde der Kamera (für Geschwindigkeitsberechnung)
        :param cm_per_pixel:  Umrechnungsfaktor Pixel → Zentimeter (nach Kalibrierung setzen)
        """
        self.fps = fps
        self.cm_per_pixel = cm_per_pixel

        # HSV-Farbraum-Filter für den Ball
        self.hsv_lower = np.array([5, 150, 150])
        self.hsv_upper = np.array([25, 255, 255])

        # Aktueller Zustand
        self.position: tuple[int, int] | None = None   # (x, y) Mittelpunkt in Pixeln
        self.radius: int = 0                            # Erkannter Radius in Pixeln
        self.speed_cm_s: float = 0.0                   # Geschwindigkeit in cm/s

        # Interne Historie
        self._prev_position: tuple[int, int] | None = None
        self._speed_history: list[float] = []           # Für gleitenden Durchschnitt

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------
    '''
    def set_hsv_range(self, lower: np.ndarray, upper: np.ndarray) -> None:
        """Setzt den HSV-Farbbereich für die Ball-Erkennung."""
        self.hsv_lower = lower
        self.hsv_upper = upper
    '''


    def update(self, frame: np.ndarray) -> tuple[int, int] | None:
        """
        Verarbeitet einen Frame, erkennt den Ball und aktualisiert Position + Geschwindigkeit.

        :param frame: BGR-Frame von der Kamera
        :return:      Erkannte Ball-Position (x, y) oder None
        """
        self._prev_position = self.position

        # 1. In HSV konvertieren und Maske erstellen
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # 2. Rauschen reduzieren
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # 3. Konturen finden
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.position = None
            self.radius = 0
            self.speed_cm_s = 0.0
            return None

        # 4. Größte Kontur = Ball
        largest = max(contours, key=cv2.contourArea)
        ((cx, cy), radius) = cv2.minEnclosingCircle(largest)

        # Mindestgröße um Rauschen auszuschließen
        if radius < 5:
            self.position = None
            return None

        self.position = (int(cx), int(cy))
        self.radius = int(radius)

        # 5. Geschwindigkeit berechnen
        self._update_speed()

        return self.position

    def _update_speed(self) -> None:
        """Berechnet Geschwindigkeit aus Positionsdifferenz zwischen zwei Frames."""
        if self._prev_position is None or self.position is None:
            self.speed_cm_s = 0.0
            return

        dx = self.position[0] - self._prev_position[0]
        dy = self.position[1] - self._prev_position[1]
        pixel_dist = (dx**2 + dy**2) ** 0.5
        cm_dist = pixel_dist * self.cm_per_pixel

        # Geschwindigkeit = Strecke / Zeit; Zeit = 1 Frame / FPS
        self.speed_cm_s = cm_dist * self.fps

        # In gleitenden Durchschnitt aufnehmen
        self._speed_history.append(self.speed_cm_s)
        if len(self._speed_history) > 100:
            self._speed_history.pop(0)


    def average_speed_cm_s(self) -> float:
        """Gibt die Durchschnittsgeschwindigkeit über alle bisher gespeicherten Frames zurück."""
        if not self._speed_history:
            return 0.0
        return sum(self._speed_history) / len(self._speed_history)

    def reset_speed_history(self) -> None:
        """Setzt die Geschwindigkeits-Historie zurück (z.B. nach einem Tor)."""
        self._speed_history.clear()


    def draw(self, frame: np.ndarray) -> None:
        """Zeichnet Ball-Position und Geschwindigkeit in den Frame."""
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


    def hsv_trackbar(self) -> None:
        """Erstellt Trackbars zum interaktiven Einstellen der HSV-Werte."""
        cv2.namedWindow("HSV Settings")
        cv2.resizeWindow("HSV Settings", 400, 300)

        # Initialisiere Trackbars mit aktuellen Werten
        cv2.createTrackbar("H_Min", "HSV Settings", int(self.hsv_lower[0]), 179, nothing)
        cv2.createTrackbar("H_Max", "HSV Settings", int(self.hsv_upper[0]), 179, nothing)
        cv2.createTrackbar("S_Min", "HSV Settings", int(self.hsv_lower[1]), 255, nothing)
        cv2.createTrackbar("S_Max", "HSV Settings", int(self.hsv_upper[1]), 255, nothing)
        cv2.createTrackbar("V_Min", "HSV Settings", int(self.hsv_lower[2]), 255, nothing)
        cv2.createTrackbar("V_Max", "HSV Settings", int(self.hsv_upper[2]), 255, nothing)

    def update_hsv_from_trackbar(self) -> None:
        """Liest die aktuellen Trackbar-Werte aus und aktualisiert hsv_lower/hsv_upper."""
        try:
            h_min = cv2.getTrackbarPos("H_Min", "HSV Settings")
            h_max = cv2.getTrackbarPos("H_Max", "HSV Settings")
            s_min = cv2.getTrackbarPos("S_Min", "HSV Settings")
            s_max = cv2.getTrackbarPos("S_Max", "HSV Settings")
            v_min = cv2.getTrackbarPos("V_Min", "HSV Settings")
            v_max = cv2.getTrackbarPos("V_Max", "HSV Settings")

            self.hsv_lower = np.array([h_min, s_min, v_min])
            self.hsv_upper = np.array([h_max, s_max, v_max])
        except cv2.error:
            pass

    def calibrate_hsv_interactive(self, camera) -> None:
        """
        Interaktive HSV-Kalibrierung mit Live-Vorschau der Maske.
        
        :param camera: Camera-Objekt zum Abrufen von Frames
        """
        print("[BallTracker] HSV-Kalibrierung gestartet. Drücke 'q' zum Beenden.")
        
        # Trackbars erstellen
        self.hsv_trackbar()
        cv2.namedWindow("Original")
        cv2.namedWindow("Mask")

        cv2.waitKey(100) # damit Trackbars initialisiert werden können
        
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
            
            # Optional: Morphologische Operationen für bessere Darstellung
            mask_cleaned = cv2.erode(mask, None, iterations=2)
            mask_cleaned = cv2.dilate(mask_cleaned, None, iterations=2)
            
            # Ergebnis auf Original anwenden
            result = cv2.bitwise_and(frame, frame, mask=mask_cleaned)
            
            # HSV-Werte ins Bild schreiben
            info_text = f"HSV Lower: {self.hsv_lower}  Upper: {self.hsv_upper}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Anzeigen
            cv2.imshow("Original", frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", result)
            
            # Auf Tasteneingabe warten
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"[BallTracker] Kalibrierung beendet. Finale Werte:")
                print(f"  hsv_lower = {self.hsv_lower}")
                print(f"  hsv_upper = {self.hsv_upper}")
                break
        
        # Fenster schließen
        cv2.destroyWindow("Original")
        cv2.destroyWindow("Mask")
        cv2.destroyWindow("Result")
        cv2.destroyWindow("HSV Settings")

