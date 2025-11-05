from __future__ import annotations

import os
from typing import Any, Dict

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QWidget, QTabWidget, QMessageBox, QLineEdit
)  # type: ignore[import]
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal  # type: ignore[import]
from PyQt6.QtGui import QShortcut, QKeySequence  # type: ignore[import]
import socket

from ..services.ai_analysis_service import AIAnalysisService, AnalysisContext, AIConfig
from ..utils.logging_setup import get_logger

_log = get_logger("ui.AIAnalysisDialog")


class _AnalysisWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict, str)
    failed = pyqtSignal(str)

    def __init__(self, service: AIAnalysisService, image_path: str, ctx: AnalysisContext, is_cancelled_callable):
        super().__init__()
        self._service = service
        self._image_path = image_path
        self._ctx = ctx
        self._cancel = is_cancelled_callable

    def run(self):
        try:
            data = self._service.analyze(
                self._image_path,
                context=self._ctx,
                progress_cb=lambda p, m: self.progress.emit(int(p), str(m)),
                is_cancelled=lambda: (bool(self._cancel()) or QThread.currentThread().isInterruptionRequested()),
            )
            # 취소 시에도 일관되게 finished로 넘겨 UI가 정리되도록 함
            self.finished.emit(data, "ok")
        except Exception as e:
            self.failed.emit(str(e))


class AIAnalysisDialog(QDialog):
    def __init__(self, parent=None, image_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("AI 분석")
        self._image_path = image_path or ""
        self._service = AIAnalysisService()
        self._data: Dict[str, Any] = {}
        self._thread: QThread | None = None
        self._worker: _AnalysisWorker | None = None
        self._cancel_flag = False
        self._close_pending = False
        self._viewer = parent

        root = QVBoxLayout(self)
        try:
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(6)
        except Exception:
            pass
        # 다크 테마 스타일(일관성)
        try:
            self.setStyleSheet(
                "QDialog { background-color: #2B2B2B; }\n"
                "QLabel { color: #EAEAEA; }\n"
                "QLineEdit, QTextEdit { background-color: #1E1E1E; color: #EAEAEA; border: 1px solid #444; }\n"
                "QPushButton { color: #EAEAEA; }\n"
                "QProgressDialog { background-color: #2B2B2B; color: #EAEAEA; }\n"
            )
        except Exception:
            pass

        # 분리 표시: 짧은 캡션 / 긴 캡션 / 태그
        # 짧은 캡션
        root.addWidget(QLabel("짧은 캡션"))
        self.sc_edit = QLineEdit(self)
        try:
            self.sc_edit.setReadOnly(True)
        except Exception:
            pass
        root.addWidget(self.sc_edit)
        # 긴 캡션
        root.addWidget(QLabel("긴 캡션"))
        self.lc_edit = QTextEdit(self)
        try:
            self.lc_edit.setReadOnly(True)
        except Exception:
            pass
        root.addWidget(self.lc_edit, 1)
        # 태그
        root.addWidget(QLabel("태그"))
        self.tags_edit = QLineEdit(self)
        try:
            self.tags_edit.setReadOnly(True)
        except Exception:
            pass
        root.addWidget(self.tags_edit)

        # Buttons (요구 순서: 분석 실행, 검색 열기, 닫기)
        btn_row = QHBoxLayout()
        self.analyze_btn = QPushButton("분석 실행")
        self.analyze_btn.clicked.connect(self._on_analyze)
        btn_row.addWidget(self.analyze_btn)

        # 퀵액션: 자연어 검색 열기(결과 기반)
        self.search_btn = QPushButton("검색 열기")
        self.search_btn.clicked.connect(self._open_search_with_result)
        btn_row.addWidget(self.search_btn)


        # 요청에 따라 내보내기/파일명 변경 버튼은 제거

        btn_row.addStretch(1)
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self._on_close)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        # 버튼 스타일/정책을 메인 UI와 일관되게 적용
        try:
            from PyQt6.QtWidgets import QSizePolicy  # type: ignore[import]
            _btn_style = "color: #EAEAEA;"
            for _b in [
                self.analyze_btn,
                self.search_btn,
                close_btn,
            ]:
                try:
                    _b.setStyleSheet(_btn_style)
                    _b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                except Exception:
                    pass
        except Exception:
            pass

        # 로딩바 없는 메시지 전용 다이얼로그
        class _BusyDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("AI 분석")
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
                lbl = QLabel("생성 중...", self)
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
                    self.setFixedSize(320, 120)
                except Exception:
                    pass
        self._busy = _BusyDialog(self)

        # Esc로 분석 취소(다이얼로그 내 우선)
        try:
            self._esc_shortcut = QShortcut(QKeySequence("Escape"), self)
            self._esc_shortcut.activated.connect(self._on_cancel)
        except Exception:
            pass

        if self._image_path:
            try:
                self._on_analyze()
            except Exception:
                pass

    def _on_cancel(self):
        self._cancel_flag = True
        try:
            self._busy.hide()
        except Exception:
            pass
        # 즉시 진행 중 작업 종료를 유도(UI는 즉시 반환)
        try:
            if self._thread and self._thread.isRunning():
                self._thread.requestInterruption()
                # UI 업데이트 시그널을 미리 끊어, 취소 직후 UI가 멈추도록 함
                if self._worker:
                    try:
                        self._worker.progress.disconnect()
                    except Exception:
                        pass
                    try:
                        self._worker.finished.disconnect()
                    except Exception:
                        pass
                    try:
                        self._worker.failed.disconnect()
                    except Exception:
                        pass
        except Exception:
            pass

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_flag)

    def _on_analyze(self):
        if not self._image_path or not os.path.exists(self._image_path):
            QMessageBox.warning(self, "AI 분석", "유효한 파일이 없습니다.")
            return
        if self._thread is not None:
            return
        self._cancel_flag = False

        # 뷰어 설정을 AIConfig로 구성(환경변수 의존 제거) + 실제 온라인 여부 확인
        cfg = self._build_config_from_viewer()
        try:
            if self._is_probably_online():
                # 온라인이면 강제로 offline_mode 해제(잘못된 설정값 무시)
                try:
                    cfg.offline_mode = False
                except Exception:
                    pass
            else:
                QMessageBox.warning(self, "AI 분석", "오프라인에서는 이 기능을 실행할 수 없습니다.")
                return
        except Exception:
            # 온라인 점검 실패 시, 보수적으로 계속 진행(서비스에서 실제 오류 처리)
            pass

        try:
            self._service.apply_config(cfg)
        except Exception:
            pass

        ctx = self._build_ctx_from_viewer()

        # 작업 스레드는 부모를 가지지 않게 하여 수명 관리 충돌을 방지
        self._thread = QThread()
        self._worker = _AnalysisWorker(self._service, self._image_path, ctx, self._is_cancelled)
        self._worker.moveToThread(self._thread)
        # started 시그널 연결은 단일 슬롯로만 연결되도록 보장
        try:
            self._thread.started.disconnect()
        except Exception:
            pass
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()
        try:
            self._busy.show()
        except Exception:
            pass

    def _is_probably_online(self) -> bool:
        try:
            s = socket.create_connection(("api.openai.com", 443), timeout=1.2)
            try:
                s.close()
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _on_progress(self, p: int, msg: str):
        # 로딩 메시지 고정("생성 중...") — 진행률 업데이트는 표시하지 않음
        return

    def _on_finished(self, data: Dict[str, Any], status: str):
        self._cleanup_worker()
        self._data = data or {}
        self._render(data or {})
        # 인덱싱: 생성된 태그/주제를 자연어 검색 인덱스에 반영
        try:
            from ..services.online_search_service import OnlineEmbeddingIndex  # type: ignore
            idx = OnlineEmbeddingIndex()
            tags = [str(t) for t in (self._data.get("tags") or [])]
            subj = [str(s) for s in (self._data.get("subjects") or [])]
            sc = str(self._data.get("short_caption") or "")
            lc = str(self._data.get("long_caption") or "")
            if self._image_path:
                idx.upsert_tags_subjects(self._image_path, tags, subj, short_caption=sc, long_caption=lc)
                # 현재 폴더 범위에서 해당 파일 색인 보강
                folder = os.path.dirname(self._image_path)
                files = [os.path.join(folder, n) for n in os.listdir(folder) if os.path.isfile(os.path.join(folder, n))]
                idx.ensure_index([self._image_path], progress_cb=None)
        except Exception:
            pass
        try:
            self._busy.hide()
        except Exception:
            pass
        # 폴백으로 반환된 경우 자세한 원인 안내
        try:
            if isinstance(self._data, dict):
                notes = str(self._data.get("notes") or "")
            else:
                notes = ""
        except Exception:
            notes = ""
        if notes:
            try:
                QMessageBox.information(self, "AI 분석", f"참고: {notes}")
            except Exception:
                pass

    def _on_failed(self, err: str):
        self._cleanup_worker()
        try:
            self._busy.hide()
        except Exception:
            pass
        # 서비스에 축적된 상세 오류도 함께 표시(가능 시)
        detail = ""
        try:
            if hasattr(self._service, "get_last_error"):
                detail = str(self._service.get_last_error() or "")
        except Exception:
            detail = ""
        msg = f"분석 실패: {err}"
        if detail:
            msg += f"\n\n원인: {detail}"
        QMessageBox.warning(self, "AI 분석", msg)

    def _cleanup_worker(self):
        try:
            if self._worker:
                try:
                    # 모든 시그널 연결 해제
                    self._worker.progress.disconnect()
                except Exception:
                    pass
                try:
                    self._worker.finished.disconnect()
                except Exception:
                    pass
                try:
                    self._worker.failed.disconnect()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if self._thread:
                try:
                    self._thread.started.disconnect()
                except Exception:
                    pass
                # 안전 종료 루프: 인터럽트→quit→wait
                try:
                    self._thread.requestInterruption()
                except Exception:
                    pass
                self._thread.quit()
                # 여유 시간 넉넉히 대기
                self._thread.wait(5000)
        except Exception:
            pass
        try:
            if self._worker:
                self._worker.deleteLater()
        except Exception:
            pass
        try:
            if self._thread:
                # 부모가 없으므로 명시적 deleteLater 호출 후 지연 없이 정리되도록 함
                self._thread.deleteLater()
        except Exception:
            pass
        self._thread = None
        self._worker = None

    def _render(self, data: Dict[str, Any]):
        # 분리된 필드 업데이트
        sc = str(data.get("short_caption") or "").strip()
        lc = str(data.get("long_caption") or "").strip()
        tags = [str(t) for t in (data.get("tags") or [])]
        try:
            self.sc_edit.setText(sc)
        except Exception:
            pass
        try:
            self.lc_edit.setPlainText(lc)
        except Exception:
            pass
        try:
            self.tags_edit.setText(", ".join(tags))
        except Exception:
            pass

        # JSON 보기/저장은 제거됨

    def _on_close(self):
        if self._thread is not None and self._thread.isRunning():
            # 즉시 취소 후 바로 닫기(스레드는 백그라운드에서 정리되도록 함)
            self._on_cancel()
            try:
                self.accept()
            except Exception:
                pass
            return
        # 이미 종료됨
        self._cleanup_worker()
        self.accept()

    def _current_strings(self) -> tuple[str, str, list[str], list[str]]:
        try:
            sc = str(self._data.get("short_caption") or "").strip()
        except Exception:
            sc = ""
        try:
            lc = str(self._data.get("long_caption") or "").strip()
        except Exception:
            lc = ""
        try:
            tags = [str(t) for t in (self._data.get("tags") or [])]
        except Exception:
            tags = []
        try:
            subj = [str(s) for s in (self._data.get("subjects") or [])]
        except Exception:
            subj = []
        return sc, lc, tags, subj

    def _copy_to_clipboard(self, which: str):
        try:
            sc, lc, tags, subj = self._current_strings()
            from PyQt6.QtWidgets import QApplication  # type: ignore[import]
            cb = QApplication.clipboard()
            if which == "caption" or which == "short":
                text = sc
            elif which == "long":
                text = lc
            elif which == "tags":
                text = ", ".join(tags)
            else:
                text = ""
            cb.setText(text or "")
            try:
                QMessageBox.information(self, "복사", "클립보드에 복사되었습니다.")
            except Exception:
                pass
        except Exception:
            pass

    def _open_search_with_result(self):
        try:
            sc, lc, tags, subj = self._current_strings()
            query = sc if sc else (", ".join(tags) if tags else lc)
            if not query:
                query = ""
            # 현재 폴더 파일 목록 확보
            files = []
            try:
                if self._image_path:
                    folder = os.path.dirname(self._image_path)
                    files = [os.path.join(folder, n) for n in os.listdir(folder) if os.path.isfile(os.path.join(folder, n))]
            except Exception:
                files = []
            from .natural_search_dialog import NaturalSearchDialog
            d = NaturalSearchDialog(self._viewer, files=files, initial_query=query)
            d.resize(800, 600)
            d.exec()
        except Exception:
            pass

    # 배치 버튼/핸들러 제거됨


    def _build_ctx_from_viewer(self) -> AnalysisContext:
        try:
            lang = str(getattr(self._viewer, "_ai_language", "ko") or "ko")
        except Exception:
            lang = "ko"
        try:
            tone = str(getattr(self._viewer, "_ai_tone", "중립") or "중립")
        except Exception:
            tone = "중립"
        try:
            purpose = str(getattr(self._viewer, "_ai_purpose", "archive") or "archive")
        except Exception:
            purpose = "archive"
        try:
            sc = int(getattr(self._viewer, "_ai_short_words", 16))
        except Exception:
            sc = 16
        try:
            lc = int(getattr(self._viewer, "_ai_long_chars", 120))
        except Exception:
            lc = 120
        ctx = AnalysisContext(
            purpose=purpose,
            tone=tone,
            language=lang,
            long_caption_chars=lc,
            short_caption_words=sc,
        )
        return ctx

    def _build_config_from_viewer(self) -> AIConfig:
        cfg = AIConfig()
        try:
            cfg.api_key = str(getattr(self._viewer, "_ai_openai_api_key", "") or "")
        except Exception:
            cfg.api_key = ""
        # 모델/프로바이더는 고정(옵션 확장 대비 기본값 유지)
        cfg.provider = "openai"
        try:
            cfg.fast_mode = bool(getattr(self._viewer, "_ai_fast_mode", False))
        except Exception:
            cfg.fast_mode = False
        try:
            cfg.offline_mode = bool(getattr(self._viewer, "_offline_mode", False))
        except Exception:
            cfg.offline_mode = False
        try:
            cfg.exif_level = str(getattr(self._viewer, "_ai_exif_level", "full"))
        except Exception:
            cfg.exif_level = "full"
        try:
            cfg.retry_count = int(getattr(self._viewer, "_ai_retry_count", 2))
        except Exception:
            cfg.retry_count = 2
        try:
            cfg.retry_delay_s = float(int(getattr(self._viewer, "_ai_retry_delay_ms", 800)) / 1000.0)
        except Exception:
            cfg.retry_delay_s = 0.8
        try:
            cfg.cache_enable = True
            cfg.cache_ttl_s = int(getattr(self._viewer, "_ai_cache_ttl_s", 0)) if hasattr(self._viewer, "_ai_cache_ttl_s") else 0
        except Exception:
            pass
        try:
            cfg.http_timeout_s = float(getattr(self._viewer, "_ai_http_timeout_s", 120))
        except Exception:
            pass
        # 이미지 전처리 고정값 유지(필요 시 설정으로 노출 가능)
        return cfg


