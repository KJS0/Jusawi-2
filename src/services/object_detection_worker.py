from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal  # type: ignore[import]
from ..utils.logging_setup import get_logger  # type: ignore

from .object_detection import Yolo11Detector, Det


class DetectionWorker(QObject):
    """
    Simple QThread-based worker that runs object detection on an image file path.
    Emits finished(dets) or error(msg) and then thread should be cleaned up by owner.
    """

    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, image_path: str, score_thr: float = 0.5, max_results: Optional[int] = None):
        super().__init__()
        self.image_path = image_path
        self.score_thr = float(score_thr)
        self.max_results = max_results
        self._thread: Optional[QThread] = None
        try:
            self.log = get_logger("svc.object_detection_worker")
        except Exception:
            self.log = None  # type: ignore

    def start(self) -> None:
        t = QThread()
        self.moveToThread(t)
        t.started.connect(self._run)
        # Ensure proper cleanup
        self.finished.connect(lambda _: self._cleanup())
        self.error.connect(lambda _msg: self._cleanup())
        t.start()
        self._thread = t

    def _cleanup(self) -> None:
        try:
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait()
        except Exception:
            pass
        self._thread = None

    def _run(self) -> None:
        try:
            if self.log:
                try:
                    self.log.info("worker_begin | path=%s | thr=%.3f | max=%s", self.image_path, self.score_thr, str(self.max_results))
                except Exception:
                    pass
            dets: List[Det] = Yolo11Detector.get().detect_file(
                self.image_path, score_thr=self.score_thr, max_results=self.max_results
            )
            # Emit plain dicts for Qt signal compatibility across threads if needed
            out = [{"bbox": d.bbox, "label": d.label, "score": d.score} for d in dets]
            self.finished.emit(out)
            if self.log:
                try:
                    self.log.info("worker_end | path=%s | count=%d", self.image_path, len(out))
                except Exception:
                    pass
        except Exception as e:
            try:
                if self.log:
                    self.log.exception("worker_error | path=%s | err=%s", self.image_path, str(e))
            except Exception:
                pass
            self.error.emit(str(e))


