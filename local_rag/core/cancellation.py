"""Cooperative cancellation shared by local inference and streamed requests."""


class RequestCancelled(Exception):
    pass


def check_cancelled(event):
    if event is not None and event.is_set():
        raise RequestCancelled()
