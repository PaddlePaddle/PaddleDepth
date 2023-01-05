# -*coding: utf-8 -*-
from Core.application import Application
from UserModelImplementation.user_interface import UserInterface


def main() -> None:
    app = Application(UserInterface(), "Masked Representation Learning for stereo matching")
    app.start()


# execute the main function
if __name__ == "__main__":
    main()
