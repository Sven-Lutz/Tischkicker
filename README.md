# EDV4 - Tischkicker
---

## Projektidee

Wie weit kann ein interdisziplinäres Team innerhalb kurzer Zeit kommen, um einen analogen
Tischfußball mithilfe von **Python und Computer Vision** so zu analysieren, dass Tore automatisch
erkannt, Zählungen durchgeführt und weiterführende Statistiken wie Schussgeschwindigkeit oder
Balltrajektorien abgeleitet werden können?

Eine kabelgebundene Webcam überträgt das Spielgeschehen direkt an einen Laptop, der die Daten
per Computer Vision (OpenCV) auswertet und live auf einem Dashboard darstellt.

---

## Systemarchitektur

Anstelle eines Edge-Devices nutzen wir eine **Direct-Compute-Architektur**:

| Komponente                  | Lösung                                                                     |
| --------------------------- | -------------------------------------------------------------------------- |
| **Sensorik**                | Kabelgebundene Webcam, zentral über dem Spielfeld montiert                 |
| **Datenübertragung**        | Hindernisfreies USB-Verlängerungskabel zum Laptop                          |
| **Recheneinheit & Display** | Laptop der Spielenden – verbindet Rechenkraft und Dashboard in einem Gerät |

Wir treffen hier die bewusste Entscheidung für die Option Laptop - USB Kabel - direkte Visualisierung, da wir so Thematiken umgehen, die wir in der Kürze der Zeit nicht zuverlässig umsetzen können. Mehr dazu bei den später aufgeführten Grenzen.

---

## Anwendung

Das System ist auf eine nahtlose, selbsterklärende User Experience ausgelegt:

```
1. CONNECT   →  Laptop per USB-Kabel mit der Kamera verbinden
2. IGNITION  →  Programm starten (Konsolenbefehl oder Desktop-Icon)
3. DASHBOARD →  GUI öffnet sich – „Spiel starten" klicken
4. MATCH     →  Live-Dashboard: Spielstand, Trajektorien & Statistiken in Echtzeit
5. REVIEW    →  Nach dem Abpfiff: Zusammenfassung & Auswertung der Session
```

> **Session-Based:** Alle Daten werden live verarbeitet und direkt während des Spiels
> im Dashboard angezeigt. Nach dem Spiel folgt eine vollständige Zusammenfassung der Session.
> Ein persistentes Speichern in eine Datenbank ist in Phase 1 bewusst nicht vorgesehen –
> das hält das System schlank und wartungsarm.

---

## Live-Statistiken

**Basis**
- Spielstand – automatische Torerkennung
- Spielzeit – läuft ab Spielstart
- Zonenverteilung – Aufenthaltszeit des Balls in Angriff / Mitte / Abwehr (% je Seite)

**Erweitert**
- Schussgeschwindigkeit – Pixelverschiebung pro Frame → m/s (kalibriert)
- Ball-Trajektorie – Pfad der letzten N Frames als Overlay im Livebild
- Top-Scorer-Stange – x-Zone des Balls in den Frames vor dem Tor
- Abprallpunkte – Richtungswechsel an den Banden
- Heatmap – Aufenthaltswahrscheinlichkeit des Balls über das Spielfeld

### Zusammenfassung
Alle Live-Werte bleiben sichtbar, ergänzt um:
- Höchstgeschwindigkeit des Spiels
- Torübersicht mit Zeitstempel
- Heatmap über die gesamte Spielzeit

---

## Struktur

```
EDV4_Tischkicker/
src/
 ├── ball_tracker/
 │    ├── __init__.py
 │    └── BallTracker.py
 ├── camera/
 │    ├── __init__.py
 │    └── Camera.py
 ├── game_controller/
 │    ├── __init__.py
 │    ├── EventHandler.py
 │    ├── GameController.py
 │    ├── HUDRenderer.py
 │    ├── SnapshotManager.py
 │    └── ScoreBoard.py
 ├── statistics/
 │    ├── __init__.py
 │    └── Statistics.py
 ├── table/
 │    ├── __init__.py
 │    └── field.py
 ├── GameEvents.py
 └── main.py
```

---

## Setup & Schnellstart

### 1. SSH-Key einrichten (einmalig pro Laptop)

SSH ist die Verbindung zwischen deinem Laptop und GitHub. Ohne SSH kannst du keine
Änderungen hochladen. Prüfe zuerst ob du bereits einen Key hast:
```bash
ssh -T git@github.com
```

- `Hi <deinName>! You've successfully authenticated` → SSH funktioniert, weiter zu Schritt 2
- Alles andere → SSH Key erstellen:
```bash
# Key erstellen
ssh-keygen -t ed25519 -C "deine@email.com"
# Dreimal Enter drücken

# Key zum SSH-Agent hinzufügen
eval "$(ssh-agent -s)"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519   # Mac
ssh-add ~/.ssh/id_ed25519                         # Linux

# Public Key kopieren
pbcopy < ~/.ssh/id_ed25519.pub   # Mac
cat ~/.ssh/id_ed25519.pub        # Linux – Ausgabe manuell kopieren
```

Dann auf GitHub: **Settings → SSH and GPG keys → New SSH key** → einfügen & speichern.

Verbindung testen:
```bash
ssh -T git@github.com
# Erwartet: "Hi <deinName>! You've successfully authenticated..."
```

---

### 2. Repository klonen
```bash
git clone git@github.com:Sven-Lutz/EDV4_Tischkicker.git
cd EDV4_Tischkicker
```

---

### 3. Einmalig: Setup ausführen
```bash
./scripts/setup.sh      # Mac / Linux
scripts\setup.bat       # Windows
```

---

### 4. Starten
```bash
./scripts/start.sh      # Mac / Linux
scripts\start.bat       # Windows
```

## Team & Branches

Wir arbeiten kollaborativ – alle sind an allen Bereichen beteiligt.
Die Branches spiegeln inhaltliche Verantwortlichkeiten wider, keine Personenzuständigkeiten.

| Branch                   | Verantwortung                                             |
| ------------------------ | --------------------------------------------------------- |
| `feature/video-source`   | Kamerazugriff, Frame-Lieferung, Auflösung                 |
| `feature/ball-detection` | HSV-Filter, Konturerkennung, x/y-Position pro Frame       |
| `feature/game-events`    | Datenklassen: GameEvent, BallPosition, Team, EventType    |
| `feature/goal-detection` | Torzonendefinition, Torerkennung, Cooldown-Logik          |
| `feature/statistics`     | Geschwindigkeit, Zonenverteilung, Heatmap, Trajektorie    |
| `feature/controller`     | Hauptloop, Orchestrierung aller Komponenten               |
| `feature/gui`            | Dashboard, Start-Screen, Live-Overlay, Spielstand-Anzeige |

## 🔀 Git-Workflow

Alle Änderungen laufen über Feature-Branches. Niemand committed direkt auf `develop` oder `main`.

**Branch-Strategie:**
- `main` → stabiler Stand, wird nur kurz vor der Präsentation gemerged
- `develop` → Integrationsstand, alle Änderungen laufen hier zusammen
- `feature/*` → ein Branch pro Komponente, siehe Tabelle oben

**Täglicher Ablauf:**
```bash
# 1. In den Branch wechseln
git checkout feature/ball-detection

# 2. Vor dem Arbeiten: aktuellen Stand holen
git pull origin develop

# 3. Arbeiten, dann committen
git add .
git commit -m "feat: Ballerkennung via HSV-Farbraum"

# 4. Hochladen
git push origin feature/ball-detection

# 5. Auf GitHub: Pull Request → develop öffnen
```

**Commit-Konventionen** – damit die History lesbar bleibt:
- `feat:` – neue Funktion
- `fix:` – Bugfix
- `docs:` – nur Dokumentation
- `test:` – Tests hinzugefügt
- `chore:` – Setup, Config, Dependencies
---

## Tests

Die Statistik-Funktionen sind durch Unit Tests abgedeckt. Da sie reinen Berechnungen
sind, können sie ohne Kamera oder Video getestet werden – die Eingabewerte werden
im Test einfach selbst definiert.
```bash
pytest tests/
```

**Was getestet wird:**
- Geschwindigkeitsberechnung zwischen zwei Ballpositionen
- Torstand-Logik (Zählung je Team)
- Zonenverteilung (Ball in welchem Drittel?)
- Abprall-Erkennung (Richtungswechsel)

**Was nicht getestet wird** *(abhängig von Hardware)*:
- Ballerkennung im echten Kamerabild
- Kamerazugriff & Frame-Lieferung

---

## Technologien

| Technologie  | Zweck                                   |
| ------------ | --------------------------------------- |
| Python 3.11+ | Hauptsprache                            |
| OpenCV       | Bildverarbeitung & Ball-Tracking        |
| NumPy        | Numerische Berechnungen                 |
| dataclasses  | Strukturierte Ereignisdaten             |
| logging      | Strukturiertes Logging (File + Console) |
| pytest       | Unit Tests                              |

---

## Outlook

### Bewusst ausgeklammert
Folgende Ansätze wurden evaluiert, aber für Phase 1 gezielt nicht umgesetzt:

- **Edge Computing** – Ein fest verbauter Raspberry Pi 5 oder Mini-PC würde den
  Laptop als Recheneinheit ersetzen und das Dashboard kabellos auf Smartphones streamen.
  Für Phase 1 bewusst nicht umgesetzt: zu aufwändig, zu fehleranfällig, kein Mehrwert
  für den Proof of Concept.

### Mit mehr Zeit
Funktionen die sinnvoll wären, den zeitlichen Rahmen aber sprengen:

- **Persistente Daten** – Datenbank-Integration für Langzeit-Spielstatistiken
- **Spielerprofile** – User Management, historische Performance-Daten, ELO-Ranking
- **Turniermodus** – Automatisierte Bracket-Erstellung, abteilungsweite Bestenlisten

## Präsentation

**Montag, 13.04 – 16:00 Uhr**

Inhalt: Idee & Forschungsfrage · Architekturüberblick · Live-Demo · Lessons Learned · Next Steps
