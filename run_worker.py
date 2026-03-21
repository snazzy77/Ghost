import os

from redis import Redis
from rq import Connection, Queue, SimpleWorker, Worker
from rq.timeouts import TimerDeathPenalty

from app.config import REDIS_URL
from app.db import init_db


class WindowsSimpleWorker(SimpleWorker):
    # Use thread timer timeouts instead of SIGALRM on Windows.
    death_penalty_class = TimerDeathPenalty


def main() -> None:
    init_db()
    redis_conn = Redis.from_url(REDIS_URL)
    with Connection(redis_conn):
        worker_cls = WindowsSimpleWorker if os.name == "nt" else Worker
        worker = worker_cls([Queue("ghost-train")])
        worker.work()


if __name__ == "__main__":
    main()
