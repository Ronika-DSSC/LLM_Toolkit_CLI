"""
Lightweight in-process job queue for the two processing pipelines.

Each pipeline (image / text) gets its own bounded worker pool so that:
  - A burst of images doesn't starve text extraction, or vice versa.
  - Concurrency can be capped per pipeline independently 
  - Uploading a whole folder doesn't try to run hundreds of jobs at once;
    work queues up behind the worker pool and drains in order.

Worker counts are configurable via environment variables so they can be
tuned per-deployment without a code change:
    IMAGE_PIPELINE_WORKERS (default: 2)
    TEXT_PIPELINE_WORKERS  (default: 4)
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

IMAGE_PIPELINE_WORKERS = int(os.environ.get("IMAGE_PIPELINE_WORKERS", "2"))
TEXT_PIPELINE_WORKERS = int(os.environ.get("TEXT_PIPELINE_WORKERS", "4"))

STATUS_QUEUED = "Queued"
STATUS_RUNNING = "Running"
STATUS_DONE = "Done"
STATUS_ERROR = "Error"

PIPELINE_IMAGE = "image"
PIPELINE_TEXT = "text"


@dataclass
class Job:
    job_id: str
    file_path: str
    file_name: str
    file_type: str
    pipeline: str  # PIPELINE_IMAGE | PIPELINE_TEXT
    run_id: str
    status: str = STATUS_QUEUED
    result: Any = None
    error: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class PipelineQueue:
    """
    Wraps two ThreadPoolExecutors (one per pipeline) and tracks job state
    in a thread-safe dict so the UI can poll progress without blocking.
    """

    def __init__(
        self,
        image_worker_fn: Callable[[str], Any],
        text_worker_fn: Callable[[str, str], Any],
        image_workers: int = IMAGE_PIPELINE_WORKERS,
        text_workers: int = TEXT_PIPELINE_WORKERS,
    ) -> None:
        self._image_worker_fn = image_worker_fn
        self._text_worker_fn = text_worker_fn

        self._image_executor = ThreadPoolExecutor(
            max_workers=image_workers, thread_name_prefix="image-pipeline"
        )
        self._text_executor = ThreadPoolExecutor(
            max_workers=text_workers, thread_name_prefix="text-pipeline"
        )

        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, file_path: str, file_name: str, file_type: str, pipeline: str, run_id: str) -> str:
        """Queue a single file for processing. Returns a job_id to poll."""
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            file_path=file_path,
            file_name=file_name,
            file_type=file_type,
            pipeline=pipeline,
            run_id=run_id,
        )
        with self._lock:
            self._jobs[job_id] = job

        executor = self._image_executor if pipeline == PIPELINE_IMAGE else self._text_executor
        executor.submit(self._run, job_id)
        return job_id

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = STATUS_RUNNING
            job.started_at = time.time()

        try:
            if job.pipeline == PIPELINE_IMAGE:
                result = self._image_worker_fn(job.file_path,job.run_id)
            else:
                result = self._text_worker_fn(job.file_path, job.file_type, job.run_id)
            with self._lock:
                job.status = STATUS_DONE
                job.result = result
        except Exception as e:  # noqa: BLE001 - capture, don't crash the worker thread
            with self._lock:
                job.status = STATUS_ERROR
                job.error = str(e)
        finally:
            with self._lock:
                job.finished_at = time.time()

    def get_jobs(self, job_ids: list[str]) -> list[Job]:
        with self._lock:
            return [self._jobs[j] for j in job_ids if j in self._jobs]

    def all_done(self, job_ids: list[str]) -> bool:
        return all(j.status in (STATUS_DONE, STATUS_ERROR) for j in self.get_jobs(job_ids))
