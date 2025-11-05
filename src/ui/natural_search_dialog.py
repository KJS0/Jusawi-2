from __future__ import annotations

import os
from typing import List, Tuple

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QListWidget, QListWidgetItem, QMessageBox
)  # type: ignore[import]
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal  # type: ignore[import]
from PyQt6.QtGui import QShortcut, QKeySequence  # type: ignore[import]

from ..services.online_search_service import OnlineEmbeddingIndex
from ..utils.logging_setup import get_logger

_log = get_logger("ui.NaturalSearchDialog")


class _SearchWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, index: OnlineEmbeddingIndex, files: List[str], query: str, top_k: int | None):
        super().__init__()
        self._index = index
        self._files = files
        self._query = query
        self._top_k = top_k
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            res = self._index.search(
                image_paths=self._files,
                query_text=self._query,
                top_k=self._top_k,
                progress_cb=lambda p, m: self.progress.emit(int(p), str(m)),
                is_cancelled=lambda: bool(self._cancelled),
            )
            self.finished.emit(res)
        except Exception as e:
            self.failed.emit(str(e))


class NaturalSearchDialog(QDialog):
    def __init__(self, parent=None, files: List[str] | None = None, initial_query: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("자연어 검색")
        self._files = files or []
        # 뷰어 설정 기반 OpenAI 모델/키 전달
        try:
            viewer = parent
        except Exception:
            viewer = None
        try:
            model = str(getattr(viewer, "_embed_model", "text-embedding-3-small")) if viewer is not None else "text-embedding-3-small"
        except Exception:
            model = "text-embedding-3-small"
        try:
            api_key = str(getattr(viewer, "_ai_openai_api_key", "")) if viewer is not None else ""
            if not api_key:
                api_key = None
        except Exception:
            api_key = None
        try:
            tag_w = int(getattr(viewer, "_search_tag_weight", 2)) if viewer is not None else 2
        except Exception:
            tag_w = 2
        try:
            vmodel = str(getattr(viewer, "_verify_model", "gpt-5-nano")) if viewer is not None else "gpt-5-nano"
        except Exception:
            vmodel = "gpt-5-nano"
        self._index = OnlineEmbeddingIndex(model=model, api_key=api_key, tag_weight=tag_w, verify_model=vmodel)
        self._thread: QThread | None = None
        self._worker: _SearchWorker | None = None
        self._pix_cache: dict[str, object] = {}

        root = QVBoxLayout(self)
        try:
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(6)
        except Exception:
            pass

        self.query_edit = QTextEdit(self)
        try:
            self.query_edit.setPlaceholderText("예) 해질녘 해변에서 역광으로 산책하는 사람")
            # 질의 에리어 높이를 작게(결과 영역 확보)
            self.query_edit.setFixedHeight(64)
        except Exception:
            pass
        root.addWidget(QLabel("질의"))
        root.addWidget(self.query_edit)
        try:
            if initial_query:
                self.query_edit.setPlainText(str(initial_query))
        except Exception:
            pass

        btn_row = QHBoxLayout()
        self.search_btn = QPushButton("검색")
        self.search_btn.clicked.connect(self._on_search)
        btn_row.addWidget(self.search_btn)
        # 재검증 모드/상위 N 컨트롤 제거(전체 후보 전수 재검증 고정)
        # from PyQt6.QtWidgets import QComboBox, QSpinBox  # type: ignore[import]
        # self.combo_verify_mode = QComboBox(self)
        # self.combo_verify_mode.addItems(["엄격", "보통", "느슨함"])  # strict/normal/loose
        # self.spin_verify_topn = QSpinBox(self); self.spin_verify_topn.setRange(0, 500); self.spin_verify_topn.setValue(20)
        btn_row.addStretch(1)
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        self.list_widget = QListWidget(self)
        try:
            # 아이콘/리스트 모드 및 썸네일 크기: 설정 반영
            from PyQt6.QtCore import QSize  # type: ignore[import]
            vm = QListWidget.ViewMode.IconMode
            icon_px = 192
            viewer = self.parent()
            try:
                if viewer is not None:
                    vm_str = str(getattr(viewer, "_search_result_view_mode", "grid"))
                    vm = QListWidget.ViewMode.IconMode if vm_str == "grid" else QListWidget.ViewMode.ListMode
                    icon_px = int(getattr(viewer, "_search_result_thumb_size", 192))
            except Exception:
                pass
            self.list_widget.setViewMode(vm)
            self.list_widget.setIconSize(QSize(icon_px, icon_px))
            self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
            self.list_widget.setMovement(QListWidget.Movement.Static)
            self.list_widget.setSpacing(10)
            # 다크 테마 일관성(배경/텍스트)
            self.list_widget.setStyleSheet(
                "QListWidget { background-color: #1F1F1F; color: #EAEAEA; border: 1px solid #333; }"
                " QListWidget::item { color: #EAEAEA; }"
                " QListWidget::item:selected { background-color: #2B2B2B; color: #FFFFFF; }"
            )
        except Exception:
            pass
        root.addWidget(self.list_widget, 1)

        # 로딩바 없는 메시지 전용 다이얼로그
        class _BusyDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("검색")
                self.setModal(True)
                try:
                    self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
                except Exception:
                    pass
                lay = QVBoxLayout(self)
                try:
                    lay.setContentsMargins(16, 16, 16, 16)
                except Exception:
                    pass
                lbl = QLabel("검색 중... (탐색 과정에서 시간이 걸릴 수 있습니다)", self)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lay.addWidget(lbl)
                try:
                    self.setStyleSheet(
                        "QDialog { background-color: #2B2B2B; color: #EAEAEA; }"
                        " QLabel { color: #EAEAEA; }"
                    )
                except Exception:
                    pass
                try:
                    self.setFixedSize(360, 120)
                except Exception:
                    pass

        self._busy = _BusyDialog(self)
        # 기본값: 부모 뷰어 설정 반영 항목 제거(전체 후보 재검증 고정)

        # 단축키: Enter/Ctrl+Enter 검색, Esc 닫기
        try:
            sc_enter = QShortcut(QKeySequence("Return"), self)
            sc_enter.activated.connect(self._on_search)
            sc_ctrl_enter = QShortcut(QKeySequence("Ctrl+Return"), self)
            sc_ctrl_enter.activated.connect(self._on_search)
            sc_esc = QShortcut(QKeySequence("Escape"), self)
            sc_esc.activated.connect(self.accept)
        except Exception:
            pass

        # 다크 테마 일관성 적용(로컬 스타일시트, 뷰어 테마와 조화)
        try:
            self.setStyleSheet(
                "QDialog { background-color: #373737; color: #EAEAEA; }"
                " QLabel { color: #EAEAEA; }"
                " QTextEdit { background-color: #2B2B2B; color: #EAEAEA; border: 1px solid #444; }"
                " QComboBox, QSpinBox { background-color: #2B2B2B; color: #EAEAEA; border: 1px solid #444; }"
                " QPushButton { color: #EAEAEA; background-color: transparent; border: 1px solid #555; padding: 4px 8px; border-radius: 4px; }"
                " QListWidget { background-color: #1F1F1F; color: #EAEAEA; border: 1px solid #333; }"
            )
        except Exception:
            pass

    def _on_search(self):
        q = (self.query_edit.toPlainText() or "").strip()
        if not q:
            QMessageBox.information(self, "자연어 검색", "질의를 입력하세요.")
            return
        if not self._files:
            QMessageBox.information(self, "자연어 검색", "현재 폴더에 이미지가 없습니다.")
            return
        if self._thread is not None:
            return
        try:
            self._busy.show()
        except Exception:
            pass

        # 최근 질의 저장(전역 재검색용)
        try:
            viewer = self.parent()
            if viewer is not None:
                setattr(viewer, "_last_natural_query", q)
        except Exception:
            pass

        self._thread = QThread(self)
        try:
            viewer = self.parent()
            tk = int(getattr(viewer, "_search_top_k", len(self._files))) if viewer is not None else len(self._files)
        except Exception:
            tk = len(self._files)
        if tk <= 0:
            tk = len(self._files)

        self._worker = _SearchWorker(self._index, self._files, q, tk)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _on_progress(self, p: int, msg: str):
        # 퍼센트/세부 메시지 갱신 없이 고정 문구 유지
        return

    def _on_finished(self, results: List[Tuple[str, float]]):
        self._cleanup_worker()
        try:
            self._busy.hide()
        except Exception:
            pass
        # 필터/정렬/표시 옵션 적용
        # 설정 기반 임베딩/검증 옵션이 반영되도록 재조합 필요 시 향후 확장 여지
        results2 = self._apply_viewer_prefs(results)
        # 필터로 인해 모두 제외되었을 경우, 원본 결과를 그대로 보여주고 안내
        if results and not results2:
            try:
                from PyQt6.QtWidgets import QMessageBox  # type: ignore[import]
                QMessageBox.information(self, "자연어 검색", "필터 조건으로 인해 결과가 모두 제외되었습니다. 필터를 조정하세요.")
            except Exception:
                pass
            self._render_results(results)
            return
        self._render_results(results2)
        # 요구사항: 자연어 검색 결과는 필름스트립(인덱스)에 표시하지 않음
        # (기존 viewer.filmstrip.set_items 경로를 제거)

    def _on_failed(self, err: str):
        self._cleanup_worker()
        try:
            self._busy.hide()
        except Exception:
            pass
        QMessageBox.warning(self, "자연어 검색", f"실패: {err}")

    def _cleanup_worker(self):
        try:
            if self._worker:
                try:
                    self._worker.cancel()
                except Exception:
                    pass
                self._worker.deleteLater()
        except Exception:
            pass
        try:
            if self._thread:
                self._thread.quit()
                self._thread.wait(1000)
                self._thread.deleteLater()
        except Exception:
            pass
        self._thread = None
        self._worker = None

    def _render_results(self, results: List[Tuple[str, float]]):
        self.list_widget.clear()
        try:
            from PyQt6.QtGui import QIcon, QPixmap  # type: ignore[import]
            from PyQt6.QtCore import QSize  # type: ignore[import]
        except Exception:
            QIcon = None  # type: ignore
            QPixmap = None  # type: ignore
            QSize = None  # type: ignore
        show_score = True
        try:
            viewer = self.parent()
            if viewer is not None:
                show_score = bool(getattr(viewer, "_search_show_score", True))
        except Exception:
            show_score = True
        # 현재 아이콘 크기 설정 값 재적용(동적 반영)
        icon_px = 192
        try:
            viewer = self.parent()
            if viewer is not None:
                icon_px = int(getattr(viewer, "_search_result_thumb_size", 192))
        except Exception:
            icon_px = 192
        def _load_icon(path: str, box: int):
            # 캐시 우선
            try:
                pm = self._pix_cache.get(path)
                if pm is not None and not pm.isNull():
                    if QIcon is not None:
                        return QIcon(pm)
                    return None
            except Exception:
                pass
            # 1차: QPixmap 직접 로드
            try:
                if QPixmap is not None:
                    pm1 = QPixmap(path)
                    if not pm1.isNull():
                        if pm1.width() > box or pm1.height() > box:
                            pm1 = pm1.scaled(box, box, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        self._pix_cache[path] = pm1
                        return QIcon(pm1) if QIcon is not None else None
            except Exception:
                pass
            # 2차: QImageReader로 스케일 로드(포맷/EXIF 회전 반영)
            try:
                from PyQt6.QtGui import QImageReader  # type: ignore
                rdr = QImageReader(path)
                rdr.setAutoTransform(True)
                try:
                    sz = rdr.size()
                    ow, oh = int(sz.width()), int(sz.height())
                    if ow > 0 and oh > 0:
                        if ow < oh:
                            tw = box
                            th = int(box * oh / max(1, ow))
                        else:
                            tw = int(box * ow / max(1, oh))
                            th = box
                        from PyQt6.QtCore import QSize as _QSize  # type: ignore
                        rdr.setScaledSize(_QSize(tw, th))
                except Exception:
                    pass
                img = rdr.read()
                if not img.isNull():
                    pm2 = QPixmap.fromImage(img)
                    if not pm2.isNull():
                        if pm2.width() > box or pm2.height() > box:
                            pm2 = pm2.scaled(box, box, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        self._pix_cache[path] = pm2
                        return QIcon(pm2) if QIcon is not None else None
            except Exception:
                pass
            return None

        for path, score in results:
            base = os.path.basename(path)
            label = f"{base}\n{score:.3f}" if show_score else base
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, path)
            # 썸네일 생성 시도
            icon = _load_icon(path, int(icon_px))
            try:
                if icon is not None:
                    item.setIcon(icon)
                    if QSize is not None:
                        # 아이콘+라벨 높이 추정치를 설정(여백 포함)
                        item.setSizeHint(QSize(int(icon_px + 20), int(icon_px + (40 if show_score else 28))))
            except Exception:
                pass
            self.list_widget.addItem(item)

    def _apply_viewer_prefs(self, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        # 필터 전면 비활성화: 결과는 그대로 두고 정렬만 적용
        try:
            viewer = self.parent()
        except Exception:
            viewer = None
        paths_scores = results[:]

        # 정렬
        try:
            sort_order = str(getattr(viewer, "_search_sort_order", "similarity")) if viewer is not None else "similarity"
        except Exception:
            sort_order = "similarity"
        if sort_order == "similarity":
            return paths_scores
        elif sort_order == "date_desc":
            def _ts_key(p: str) -> float:
                try:
                    from ..services.exif_utils import extract_with_pillow  # type: ignore
                    meta = extract_with_pillow(p)
                    dt = str(meta.get("datetime") or "")
                    if dt:
                        import datetime as _dt
                        # EXIF 'YYYY:MM:DD HH:MM:SS'
                        y = int(dt[0:4]); m = int(dt[5:7]); d = int(dt[8:10])
                        hh = int(dt[11:13]); mm = int(dt[14:16]); ss = int(dt[17:19])
                        return _dt.datetime(y, m, d, hh, mm, ss).timestamp()
                except Exception:
                    pass
                try:
                    return float(os.path.getmtime(p))
                except Exception:
                    return 0.0
            try:
                return sorted(paths_scores, key=lambda x: _ts_key(x[0]), reverse=True)
            except Exception:
                return paths_scores
        elif sort_order == "name_asc":
            try:
                return sorted(paths_scores, key=lambda x: os.path.basename(x[0]).lower())
            except Exception:
                return paths_scores
        return paths_scores

    def _on_open_selected(self):
        items = self.list_widget.selectedItems()
        if not items:
            return
        path = items[0].data(Qt.ItemDataRole.UserRole)
        try:
            # 부모 뷰어가 있으면 열기 시도
            viewer = self.parent()
            if viewer and hasattr(viewer, "load_image"):
                viewer.load_image(path, source='search')
                self.accept()
                return
        except Exception:
            pass
        # 폴백: 경로만 알림
        QMessageBox.information(self, "열기", str(path))