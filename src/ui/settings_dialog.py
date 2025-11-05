from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QDialogButtonBox, QWidget, QTabWidget, QPushButton, QScrollArea
)
from .settings.core_page import CoreSettingsPage


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("설정")
        self._viewer_for_keys = None

        root = QVBoxLayout(self)
        try:
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)
        except Exception:
            pass
        self.tabs = QTabWidget(self)
        root.addWidget(self.tabs)

        # 통합 페이지(탭 없이 한 탭만 사용)
        self.core_tab = QWidget(self)
        self.tabs.addTab(self.core_tab, "설정")
        core_layout = QVBoxLayout(self.core_tab)
        try:
            core_layout.setContentsMargins(0, 0, 0, 0)
            core_layout.setSpacing(0)
        except Exception:
            pass
        self.core_page = CoreSettingsPage(None)
        try:
            _sa = QScrollArea(self.core_tab)
            _sa.setWidgetResizable(True)
            _sa.setFrameShape(QScrollArea.Shape.NoFrame)
            _sa.setWidget(self.core_page)
            core_layout.addWidget(_sa)
        except Exception:
            core_layout.addWidget(self.core_page)

        # 하단: 기본 설정 버튼 + 확인/취소
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        self.btn_reset_all = QPushButton("기본 설정으로 재설정", self)
        try:
            self.btn_reset_all.clicked.connect(self._on_reset_all_defaults)
        except Exception:
            pass
        bottom_row.addWidget(self.btn_reset_all)
        bottom_row.addStretch(1)
        root.addLayout(bottom_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        def _apply_and_accept():
            try:
                viewer = self.parent()
                if viewer is not None:
                    self.apply_to_viewer(viewer)
                    viewer.save_settings()
            except Exception:
                pass
            self.accept()
        buttons.accepted.connect(_apply_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        # 기본 크기(요청 사이즈 근접값)
        try:
            self.setMinimumSize(520, 380)
            self.resize(560, 420)
            try:
                # 탭 바 숨김: 단일 설정창처럼 보이도록
                self.tabs.setTabBarAutoHide(True)
                self.tabs.tabBar().hide()
            except Exception:
                pass
        except Exception:
            pass
        return

    # populate from viewer
    def load_from_viewer(self, viewer):
        # viewer 참조 저장(지연 로드시 사용)
        self._viewer_for_keys = viewer
        try:
            if hasattr(self, "core_page"):
                self.core_page.load_from_viewer(viewer)
        except Exception:
            pass
        # 자연 정렬 UI 규칙 동기화
        try:
            self._on_sort_name_changed()
        except Exception:
            pass

    # commit back to viewer (does not save)
    def apply_to_viewer(self, viewer):
        if hasattr(self, "core_page"):
            self.core_page.apply_to_viewer(viewer)

    def _on_sort_name_changed(self):
        return

    # ----- helpers -----
    def _on_reset_defaults(self):
        return

    def _on_reset_all_defaults(self):
        # 통합 페이지 기본값만 초기화
        try:
            if hasattr(self, "core_page"):
                self.core_page.reset_to_defaults()
            viewer = getattr(self, "_viewer_for_keys", None)
            if viewer is not None:
                self.apply_to_viewer(viewer)
                try:
                    viewer.save_settings()
                except Exception:
                    pass
        except Exception:
            pass

    def _on_accept(self):
        self.accept()

    def _on_import_yaml(self):
        # 제거됨
        return

    def _on_export_yaml(self):
        # 제거됨
        return

    def _on_tab_changed(self, index: int):
        return

    def _build_keys_tab(self):
        return

    def _collect_mapping_from_ui(self, validate_against_fixed: bool = True):
        return {}

    def _normalize_single_key(self, editor):
        return

    def _on_key_changed(self, editor, defaults: list, row_idx: int):
        return

    def _show_key_warning(self, title: str, text: str, editor):
        return

    def _seq_to_text(self, seq) -> str:
        return ""

    def _normalize_default_text(self, defaults: list) -> str:
        if not defaults:
            return ""
        return str(defaults[0])

    # 이벤트 필터: 배타 포커스 + Backspace/Delete로 해제 지원
    def eventFilter(self, obj, event):
        return super().eventFilter(obj, event)


class _LabeledRow(QHBoxLayout):
    def __init__(self, label_text: str):
        super().__init__()
        self.addStretch(1)


def _spin(min_v: int, max_v: int, value: int):
    return None


