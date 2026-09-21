"""Background workers, so a long SOM run never freezes the window."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..i18n import Text, render


class Worker(QObject):
    """Runs one callable off the GUI thread and reports progress."""

    progress = Signal(str, float)      # message, 0..1
    finished = Signal(object)          # result
    failed = Signal(str, str)          # short message, traceback

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    @Slot()
    def run(self) -> None:
        try:
            result = self._fn(*self._args, progress=self._emit, **self._kwargs)
        except Exception as exc:                    # surfaced in the log pane
            self.failed.emit(_message(exc), traceback.format_exc())
            return
        self.finished.emit(result)

    def _emit(self, message, fraction=None) -> None:
        if fraction is None and isinstance(message, (int, float)):
            message, fraction = "", float(message)
        # A Qt signal carries a plain str, which would drop a Text's translation,
        # so the message is rendered in the interface language before it leaves.
        self.progress.emit(render(message), float(fraction or 0.0))


def _message(exc: Exception) -> str:
    """An exception's message in the interface language, where it has one."""
    if len(exc.args) == 1 and isinstance(exc.args[0], Text):
        return exc.args[0].render()
    return str(exc)


class TaskRunner(QObject):
    """Owns the thread/worker pair and keeps them alive for the call's duration.

    The completion handlers are real slots on this object, which lives in the
    GUI thread.  Connecting plain Python callables instead would give a direct
    connection that executes inside the worker thread -- and the cleanup step
    would then call ``QThread.wait()`` on the very thread it is running in,
    which deadlocks.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._on_done = None
        self._on_error = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self, fn, on_done, on_progress=None, on_error=None, *args, **kwargs) -> None:
        if self.busy:
            raise RuntimeError("A task is already running.")

        self._on_done = on_done
        self._on_error = on_error

        thread = QThread(self)
        worker = Worker(fn, *args, **kwargs)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._finished)
        worker.failed.connect(self._failed)
        if on_progress:
            worker.progress.connect(on_progress)

        self._thread, self._worker = thread, worker
        thread.start()

    # ------------------------------------------------------------------
    @Slot(object)
    def _finished(self, result) -> None:
        callback = self._on_done
        self._teardown()
        if callback:
            callback(result)

    @Slot(str, str)
    def _failed(self, message: str, tb: str) -> None:
        callback = self._on_error
        self._teardown()
        if callback:
            callback(message, tb)

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = None
        self._worker = None
        self._on_done = None
        self._on_error = None


__all__ = ["Worker", "TaskRunner"]
