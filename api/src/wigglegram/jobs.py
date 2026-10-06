"""Bounded in-process job queue. Heavy work runs in worker threads so the event loop stays free."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .storage import new_id

log = logging.getLogger(__name__)


class QueueFull(Exception):
    pass


@dataclass
class Job:
    id: str
    payload: Any
    status: str = "queued"  # queued | running | done | error
    stage: str | None = None
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None


class JobError(Exception):
    """An error whose message is safe to show to the user."""


class JobQueue:
    def __init__(self, handler: Callable[[Job], dict], max_pending: int, workers: int = 1, ttl_s: float = 3600):
        self.handler = handler
        self.max_pending = max_pending
        self.workers = workers
        self.ttl_s = ttl_s
        self.jobs: dict[str, Job] = {}
        self._pending: list[str] = []
        self._queue: asyncio.Queue[str] | None = None
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        self._queue = asyncio.Queue()
        self._tasks = [asyncio.create_task(self._worker()) for _ in range(self.workers)]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def submit(self, payload: Any) -> Job:
        if len(self._pending) >= self.max_pending:
            raise QueueFull
        job = Job(id=new_id(), payload=payload)
        self.jobs[job.id] = job
        self._pending.append(job.id)
        self._queue.put_nowait(job.id)
        return job

    def position(self, job: Job) -> int | None:
        """0-based place in line while queued."""
        try:
            return self._pending.index(job.id)
        except ValueError:
            return None

    @property
    def pending(self) -> int:
        return len(self._pending)

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            self._pending.remove(job_id)
            job = self.jobs.get(job_id)
            if job is None:
                continue
            job.status = "running"
            try:
                job.result = await asyncio.to_thread(self.handler, job)
                job.status = "done"
            except JobError as e:
                job.status, job.error = "error", str(e)
            except Exception:
                log.exception("job %s failed", job_id)
                job.status, job.error = "error", "Processing failed."
            finally:
                job.finished = time.time()

    def sweep(self) -> None:
        cutoff = time.time() - self.ttl_s
        for jid in [j.id for j in self.jobs.values() if j.finished and j.finished < cutoff]:
            del self.jobs[jid]
