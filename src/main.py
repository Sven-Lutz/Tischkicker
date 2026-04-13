from game_controller.GameController import GameController


def main():
    controller = GameController(
        camera_source=0,
        team_names=("Links", "Rechts"),
        cm_per_pixel=0.1,
        goals_to_win=10
    )
    controller.start()


if __name__ == "__main__":
    main()