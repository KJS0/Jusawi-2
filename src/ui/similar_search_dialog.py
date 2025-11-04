from __future__ import annotations

import os
from typing import List, Tuple
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QProgressDialog  # type: ignore[import]
from PyQt6.QtGui import QIcon, QPixmap, QColor, QPalette  # type: ignore[import]
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer  # type: ignore[import]

try:
    from PIL import Image, ImageQt  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageQt = None  # type: ignore

from ..services.similarity_service import SimilarityIndex
from ..utils.logging_setup import get_logger

_log = get_logger("ui.SimilarDialog")


class _Worker(QThread):
    progress = pyqtSignal(int, str)
    done = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, anchor: str, folder: str, svc: SimilarityIndex, top_k: int = 100):
        super().__init__()
        self._anchor = anchor
        self._folder = folder
        self._svc = svc
        self._top_k = top_k
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self.progress.emit(10, "폴더 인덱싱…")
            # 인덱싱 단계: 내부에서 캐시/스캔 처리
            if self._cancel:
                self.failed.emit("취소됨")
                return
            self.progress.emit(60, "유사도 계산…")
            if self._cancel:
                self.failed.emit("취소됨")
                return
            # auto: 대용량 폴더는 ANN, 그 외에는 pHash→CLIP
            res = self._svc.similar_auto(self._anchor, self._folder, top_k=self._top_k, mode="auto")
            if self._cancel:
                self.failed.emit("취소됨")
                return
            self.progress.emit(100, "완료")
            self.done.emit(res)
        except Exception as e:
            self.failed.emit(str(e))


class SimilarSearchDialog(QDialog):
    def __init__(self, parent, anchor_path: str, folder: str):
        super().__init__(parent)
        self.setWindowTitle("유사 사진 찾기")
        self._anchor = anchor_path
        self._folder = folder
        self._svc = SimilarityIndex()
        self._worker: _Worker | None = None
        self._busy: QDialog | None = None
        self._busy_label: QLabel | None = None

        lay = QVBoxLayout(self)
        self.listw = QListWidget(self)
        try:
            # 탐색기 스타일 썸네일 그리드 구성 (4열 x 3행, 총 12개 초기 표시)
            cols, rows = 4, 3
            icon = QSize(160, 160)
            # 텍스트 두 줄(+여유)을 위한 그리드 높이 확장
            grid = QSize(190, 230)
            spacing = 6
            border = 16  # 여유 마진
            width = cols * grid.width() + (cols - 1) * spacing + border
            height = rows * grid.height() + (rows - 1) * spacing + border

            self.listw.setViewMode(self.listw.ViewMode.IconMode)
            self.listw.setIconSize(icon)
            self.listw.setResizeMode(self.listw.ResizeMode.Adjust)
            self.listw.setMovement(self.listw.Movement.Static)
            self.listw.setUniformItemSizes(True)
            self.listw.setWrapping(True)
            self.listw.setGridSize(grid)
            try:
                # 너무 긴 파일명은 가운데 생략으로 표시
                self.listw.setTextElideMode(Qt.TextElideMode.ElideMiddle)
            except Exception:
                pass
            self.listw.setSpacing(spacing)
            # 4열 강제: 리스트 너비를 그리드 폭에 맞춰 고정
            self.listw.setMinimumWidth(width)
            self.listw.setMaximumWidth(width)
            self.listw.setMinimumHeight(height)
            self.listw.setMaximumHeight(height)
        except Exception:
            pass
        # 다크 테마와 일관된 위젯 스타일 적용
        try:
            self.setStyleSheet(
                "QDialog { background-color: #373737; color: #EAEAEA; }"
                " QLabel { color: #EAEAEA; }"
                " QPushButton { color: #EAEAEA; background-color: transparent; border: 1px solid #555; padding: 4px 8px; border-radius: 4px; }"
                " QListWidget { background-color: #1F1F1F; color: #EAEAEA; border: 1px solid #333; }"
                " QListWidget, QListWidget::viewport { background-color: #1F1F1F; color: #EAEAEA; }"
                " QListWidget::item { background-color: #1F1F1F; color: #EAEAEA; }"
                " QListWidget::item:hover { background-color: #2A2A2A; }"
                " QListWidget::item:selected { background-color: #3A3A3A; color: #FFFFFF; }"
                " QScrollBar:vertical { background: #2B2B2B; width: 12px; }"
                " QScrollBar::handle:vertical { background: #555; min-height: 24px; border-radius: 6px; }"
                " QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { background: transparent; height: 0px; }"
                " QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: #2B2B2B; }"
            )
            # 윈도우 테마에서 뷰포트 배경이 흰색으로 남는 문제 방지: 팔레트/오토필 지정
            try:
                pal = self.listw.viewport().palette()
                pal.setColor(QPalette.ColorRole.Base, QColor("#1F1F1F"))
                pal.setColor(QPalette.ColorRole.Window, QColor("#1F1F1F"))
                pal.setColor(QPalette.ColorRole.Text, QColor("#EAEAEA"))
                self.listw.viewport().setPalette(pal)
                self.listw.viewport().setAutoFillBackground(True)
            except Exception:
                pass
            try:
                # 위젯 로컬 스타일 재강조(일부 플랫폼에서 상위 스타일이 누락되는 경우 대비)
                self.listw.setStyleSheet(
                    "QListWidget { background-color: #1F1F1F; color: #EAEAEA; border: 1px solid #333; }"
                    " QListWidget::viewport { background-color: #1F1F1F; }"
                    " QListWidget::item { background-color: #1F1F1F; color: #EAEAEA; }"
                    " QListWidget::item:selected { background-color: #3A3A3A; color: #FFFFFF; }"
                )
            except Exception:
                pass
        except Exception:
            pass
        lay.addWidget(self.listw, 1)
        self.listw.itemDoubleClicked.connect(self._on_open)

        self._pending: list[tuple[str, float]] = []
        self._batch_size = 12
        self._start_search()

    def _start_search(self):
        self.listw.clear()
        try:
            _log.info("similar_dialog_start | anchor=%s | dir=%s", os.path.basename(self._anchor), os.path.basename(self._folder))
        except Exception:
            pass
        # 심플 로딩창(진행 막대 없음)
        try:
            self._busy = QDialog(self)
            self._busy.setWindowTitle("검색")
            self._busy.setWindowModality(Qt.WindowModality.ApplicationModal)
            v = QVBoxLayout(self._busy)
            self._busy_label = QLabel("검색 중... (탐색 과정에서 시간이 걸릴 수 있습니다)", self._busy)
            try:
                self._busy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            except Exception:
                pass
            v.addWidget(self._busy_label)
            try:
                self._busy.setStyleSheet(
                    "QDialog { background-color: #373737; color: #EAEAEA; }"
                    " QLabel { color: #EAEAEA; }"
                )
            except Exception:
                pass
            self._busy.show()
        except Exception:
            pass
        # 워커 스레드 시작
        self._worker = _Worker(self._anchor, self._folder, self._svc, top_k=12)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, val: int, msg: str):
        try:
            if self._busy_label:
                # 메시지는 고정 문구 유지 요구에 따라 업데이트하지 않아도 됨
                # 필요 시 다음 라인을 활성화:
                # self._busy_label.setText(msg)
                pass
        except Exception:
            pass

    def _on_done(self, res: List[Tuple[str, float]]):
        try:
            if self._busy:
                self._busy.close()
        except Exception:
            pass
        self._busy = None
        self._worker = None
        # 배치로 추가해 UI 끊김 완화
        self._pending = list(res)
        try:
            _log.info("similar_dialog_done | count=%d", len(self._pending))
        except Exception:
            pass
        self._drain_batch()

    def _drain_batch(self):
        if not self._pending:
            return
        chunk = self._pending[:self._batch_size]
        self._pending = self._pending[self._batch_size:]
        # grid와 아이콘/텍스트 레이아웃 파라미터는 초기 구성과 동일하게 사용
        grid_size = QSize(190, 230)
        for path, score in chunk:
            it = QListWidgetItem(f"{os.path.basename(path)}  ({score:.3f})")
            pm = QPixmap(path)
            if pm.isNull() and Image is not None and ImageQt is not None:
                try:
                    with Image.open(path) as im:
                        im.thumbnail((160,160))
                        pm = QPixmap.fromImage(ImageQt.ImageQt(im))  # type: ignore
                except Exception:
                    pass
            if not pm.isNull():
                it.setIcon(QIcon(pm))
            try:
                it.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
                it.setSizeHint(grid_size)
                # 텍스트/배경 일관된 다크 테마
                it.setForeground(QColor("#EAEAEA"))
            except Exception:
                pass
            it.setData(Qt.ItemDataRole.UserRole, path)
            self.listw.addItem(it)
        if self._pending:
            QTimer.singleShot(0, self._drain_batch)

    def _on_failed(self, err: str):
        try:
            if self._busy:
                self._busy.close()
        except Exception:
            pass
        self._busy = None
        self._worker = None
        try:
            self.listw.clear()
            self.listw.addItem(QListWidgetItem(f"검색 실패: {err}"))
        except Exception:
            pass
        try:
            _log.warning("similar_dialog_fail | err=%s", err)
        except Exception:
            pass

    def _on_cancel(self):
        try:
            if self._worker:
                self._worker.cancel()
        except Exception:
            pass
        try:
            _log.info("similar_dialog_cancel")
        except Exception:
            pass

    def _on_open(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        try:
            self.parent().load_image(path, source='similar')
            self.accept()
        except Exception:
            pass



    def keyPressEvent(self, e):
        try:
            k = getattr(e, 'key')() if hasattr(e, 'key') else None
        except Exception:
            k = None
        try:
            if k == Qt.Key.Key_Escape:
                # 검색 취소 후 닫기
                self._on_cancel()
                try:
                    self.reject()
                except Exception:
                    pass
                return
            if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                it = self.listw.currentItem()
                if it is not None:
                    self._on_open(it)
                    return
            if k in (
                Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down,
                Qt.Key.Key_PageUp, Qt.Key.Key_PageDown, Qt.Key.Key_Home, Qt.Key.Key_End,
            ):
                try:
                    self.listw.setFocus()
                    # QListWidget 기본 탐색 동작에 위임
                    self.listw.keyPressEvent(e)
                except Exception:
                    pass
                return
        except Exception:
            pass
        try:
            super().keyPressEvent(e)
        except Exception:
            pass
