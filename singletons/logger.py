import logging

_logger = None


def get_logger():
    global _logger
    if _logger is None:
        logging.basicConfig(level=logging.INFO)
        _logger = logging.getLogger(__name__)
    return _logger
