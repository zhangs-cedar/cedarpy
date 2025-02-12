from pync import Notifier


def s_print(title):
    """
    A simple print function
    """
    Notifier.notify(title)
