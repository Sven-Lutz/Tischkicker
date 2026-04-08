"""Kicker GT3 — entry point."""

import logging
import sys

from PyQt6.QtWidgets import QApplication

from src.controller import Controller
from src.gui import KickerGUI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    app = QApplication(sys.argv)

    # Placeholders — wired up after both objects exist
    gui: KickerGUI
    controller: Controller

    def on_start(config):
        controller.start_game(config)

    def on_end_game():
        controller.end_game()

    def on_new_game():
        controller.new_game()

    def on_quit():
        controller.quit()
        app.quit()

    gui = KickerGUI(
        on_start=on_start,
        on_end_game=on_end_game,
        on_new_game=on_new_game,
        on_quit=on_quit,
    )
    controller = Controller(gui)

    gui.show_start_screen()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
