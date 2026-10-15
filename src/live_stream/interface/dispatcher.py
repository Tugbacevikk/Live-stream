"""Webhook göndericisi süreci:  python -m live_stream.interface.dispatcher"""
from __future__ import annotations

import logging

from live_stream.interface.composition import store_from_settings
from live_stream.infrastructure.webhook import OutboxDispatcher
from live_stream.interface.settings import Settings


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = store_from_settings(Settings(), pool_size=1)
    store.migrate()
    s = Settings()
    OutboxDispatcher(store, secret=s.webhook_secret, max_attempts=s.webhook_max_attempts).run_forever()


if __name__ == "__main__":
    main()
