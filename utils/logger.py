import constants


class Logger:
    def info(self, message: str):
        print(f"{constants.LOG_COLOR_OKGREEN}[INFO]{constants.LOG_COLOR_ENDC}",
        f"{message}")

    def error(self, message: str):
        print(f"{constants.LOG_COLOR_WARNING}[ERROR]{constants.LOG_COLOR_ENDC}",
              f"{message}")
