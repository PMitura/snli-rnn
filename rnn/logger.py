from progress.bar import Bar


class LoggerBar(Bar):
    suffix = '%(percent).1f%% - %(eta)ds'


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def header(text):
    print('\n' + Bcolors.BOLD + Bcolors.HEADER + text + Bcolors.ENDC)


def success(text, level=1):
    print((' ' * level * 4) + Bcolors.OKGREEN + "[OK]   " + Bcolors.ENDC + text)


def info(text, level=1):
    print((' ' * level * 4) + Bcolors.OKBLUE + "[INFO] " + Bcolors.ENDC + text)


def error(text, level=1):
    print((' ' * level * 4) + Bcolors.FAIL + "[ERR]  " + Bcolors.ENDC + text)


def get_progress_bar(message="", level=1, limit=20):
    prefix = (' ' * level * 4) + Bcolors.OKBLUE + "[INFO] " + Bcolors.ENDC + message
    return LoggerBar(prefix, max=limit)
