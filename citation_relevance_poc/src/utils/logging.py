from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a named logger after installing the project default format.

    `basicConfig` is intentionally lightweight; if an application has already
    configured logging, Python will leave the existing handlers in place.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger(name)
