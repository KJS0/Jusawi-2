from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QCheckBox, QComboBox, QSpinBox, QFormLayout,
)

from .base import SettingsPage


class GeneralSettingsPage(SettingsPage):
    """일반 탭 구현: 기존 SettingsDialog 일반 섹션을 모듈화.

    UI 요소 id/역할은 기존 다이얼로그 구현을 그대로 따릅니다.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        try:
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(8)
        except Exception:
            pass

        # 파일 열기 관련
        self.chk_scan_after_open = QCheckBox("열기 후 폴더 자동 스캔", self)
        self.chk_remember_last_dir = QCheckBox("마지막 사용 폴더 기억", self)
        root.addWidget(self.chk_scan_after_open)
        root.addWidget(self.chk_remember_last_dir)

        # 세션/최근 옵션
        root.addWidget(QLabel("세션/최근", self))
        self.combo_startup_restore = QComboBox(self)
        self.combo_startup_restore.addItems(["항상 복원", "묻기", "복원 안 함"])  # always/ask/never
        self.spin_recent_max = QSpinBox(self); self.spin_recent_max.setRange(1, 100); self.spin_recent_max.setSuffix(" 개")
        self.chk_recent_auto_prune = QCheckBox("존재하지 않는 항목 자동 정리", self)
        form_recent = QFormLayout()
        form_recent.addRow("시작 시 세션 복원", self.combo_startup_restore)
        form_recent.addRow("최근 목록 최대 개수", self.spin_recent_max)
        form_recent.addRow("존재하지 않음 자동 정리", self.chk_recent_auto_prune)
        root.addLayout(form_recent)

        # 애니메이션(오버레이 관련 UI 제거)
        root.addWidget(QLabel("애니메이션", self))
        self.chk_anim_autoplay = QCheckBox("자동 재생", self)
        self.chk_anim_loop = QCheckBox("루프 재생", self)
        self.chk_anim_keep_state = QCheckBox("파일 전환 시 재생 상태 유지", self)
        self.chk_anim_pause_unfocus = QCheckBox("비활성화 시 자동 일시정지", self)
        self.chk_anim_click_toggle = QCheckBox("이미지 클릭으로 재생/일시정지", self)
        form_anim = QFormLayout()
        form_anim.addRow("재생 상태 유지", self.chk_anim_keep_state)
        form_anim.addRow("비활성화 시 일시정지", self.chk_anim_pause_unfocus)
        form_anim.addRow("클릭으로 재생/일시정지", self.chk_anim_click_toggle)
        root.addWidget(self.chk_anim_autoplay)
        root.addWidget(self.chk_anim_loop)
        root.addLayout(form_anim)

        # 디렉터리 탐색(정렬 고정: Windows 탐색기식). 노출 옵션 축소.
        root.addWidget(QLabel("디렉터리 탐색", self))
        self.chk_wrap_ends = QCheckBox("끝에서 반대쪽으로 순환 이동", self)
        self.spin_nav_throttle = QSpinBox(self); self.spin_nav_throttle.setRange(0, 2000); self.spin_nav_throttle.setSuffix(" ms")
        self.chk_film_center = QCheckBox("필름스트립 항목을 자동 중앙 정렬", self)
        self.combo_zoom_policy = QComboBox(self); self.combo_zoom_policy.addItems(["전환 시 초기화", "보기 모드 유지", "배율 유지"])
        form = QFormLayout()
        form.addRow("끝에서의 동작", self.chk_wrap_ends)
        form.addRow("연속 탐색 속도", self.spin_nav_throttle)
        form.addRow("필름스트립 중앙 정렬", self.chk_film_center)
        form.addRow("확대 상태 유지", self.combo_zoom_policy)
        root.addLayout(form)

        # 드래그 앤 드롭 옵션 제거됨(기능 비활성화)

        # TIFF
        self.chk_tiff_first_page = QCheckBox("항상 첫 페이지로 열기", self)
        row_tiff = QFormLayout()
        row_tiff.addRow("TIFF", self.chk_tiff_first_page)
        root.addLayout(row_tiff)

    # 데이터 바인딩
    def load_from_viewer(self, viewer: Any) -> None:  # noqa: ANN401
        try:
            self.chk_scan_after_open.setChecked(bool(getattr(viewer, "_open_scan_dir_after_open", True)))
        except Exception:
            self.chk_scan_after_open.setChecked(True)
        try:
            self.chk_remember_last_dir.setChecked(bool(getattr(viewer, "_remember_last_open_dir", True)))
        except Exception:
            self.chk_remember_last_dir.setChecked(True)
        try:
            self.chk_anim_autoplay.setChecked(bool(getattr(viewer, "_anim_autoplay", True)))
        except Exception:
            self.chk_anim_autoplay.setChecked(True)
        try:
            self.chk_anim_loop.setChecked(bool(getattr(viewer, "_anim_loop", True)))
        except Exception:
            self.chk_anim_loop.setChecked(True)
        try:
            self.chk_anim_keep_state.setChecked(bool(getattr(viewer, "_anim_keep_state_on_switch", False)))
        except Exception:
            self.chk_anim_keep_state.setChecked(False)
        try:
            self.chk_anim_pause_unfocus.setChecked(bool(getattr(viewer, "_anim_pause_on_unfocus", False)))
        except Exception:
            self.chk_anim_pause_unfocus.setChecked(False)
        try:
            self.chk_anim_click_toggle.setChecked(bool(getattr(viewer, "_anim_click_toggle", False)))
        except Exception:
            self.chk_anim_click_toggle.setChecked(False)
        # 세션/최근
        try:
            pol = str(getattr(viewer, "_startup_restore_policy", "always"))
            self.combo_startup_restore.setCurrentIndex({"always":0, "ask":1, "never":2}.get(pol, 0))
        except Exception:
            self.combo_startup_restore.setCurrentIndex(0)
        try:
            self.spin_recent_max.setValue(int(getattr(viewer, "_recent_max_items", 10)))
        except Exception:
            self.spin_recent_max.setValue(10)
        try:
            self.chk_recent_auto_prune.setChecked(bool(getattr(viewer, "_recent_auto_prune_missing", True)))
        except Exception:
            self.chk_recent_auto_prune.setChecked(True)
        # 정렬/필터
        # 정렬 옵션 제거됨(윈도우 탐색기식 고정)
        try:
            self.chk_tiff_first_page.setChecked(bool(getattr(viewer, "_tiff_open_first_page_only", True)))
        except Exception:
            self.chk_tiff_first_page.setChecked(True)
        # 추가 옵션
        try:
            self.chk_wrap_ends.setChecked(bool(getattr(viewer, "_nav_wrap_ends", False)))
        except Exception:
            self.chk_wrap_ends.setChecked(False)
        try:
            self.spin_nav_throttle.setValue(int(getattr(viewer, "_nav_min_interval_ms", 100)))
        except Exception:
            self.spin_nav_throttle.setValue(100)
        try:
            self.chk_film_center.setChecked(bool(getattr(viewer, "_filmstrip_auto_center", True)))
        except Exception:
            self.chk_film_center.setChecked(True)
        try:
            zp = str(getattr(viewer, "_zoom_policy", "mode"))
            self.combo_zoom_policy.setCurrentIndex({"reset":0, "mode":1, "scale":2}.get(zp, 1))
        except Exception:
            self.combo_zoom_policy.setCurrentIndex(1)

    def apply_to_viewer(self, viewer: Any) -> None:  # noqa: ANN401
        try:
            viewer._open_scan_dir_after_open = bool(self.chk_scan_after_open.isChecked())
        except Exception:
            pass
        try:
            viewer._remember_last_open_dir = bool(self.chk_remember_last_dir.isChecked())
        except Exception:
            pass
        try:
            viewer._anim_autoplay = bool(self.chk_anim_autoplay.isChecked())
            viewer._anim_loop = bool(self.chk_anim_loop.isChecked())
            viewer._anim_keep_state_on_switch = bool(self.chk_anim_keep_state.isChecked())
            viewer._anim_pause_on_unfocus = bool(self.chk_anim_pause_unfocus.isChecked())
            viewer._anim_click_toggle = bool(self.chk_anim_click_toggle.isChecked())
            # 오버레이 관련 속성은 더 이상 사용하지 않음
            viewer._anim_overlay_enabled = False
        except Exception:
            pass
        try:
            pol_idx = int(self.combo_startup_restore.currentIndex())
            viewer._startup_restore_policy = ("always" if pol_idx == 0 else ("ask" if pol_idx == 1 else "never"))
        except Exception:
            pass
        try:
            viewer._recent_max_items = int(self.spin_recent_max.value())
            viewer._recent_auto_prune_missing = bool(self.chk_recent_auto_prune.isChecked())
        except Exception:
            pass
        # 정렬 옵션 제거됨
        try:
            viewer._tiff_open_first_page_only = bool(self.chk_tiff_first_page.isChecked())
        except Exception:
            pass
        try:
            viewer._nav_wrap_ends = bool(self.chk_wrap_ends.isChecked())
            viewer._nav_min_interval_ms = int(self.spin_nav_throttle.value())
            viewer._filmstrip_auto_center = bool(self.chk_film_center.isChecked())
            idx = int(self.combo_zoom_policy.currentIndex())
            viewer._zoom_policy = ("reset" if idx == 0 else ("mode" if idx == 1 else "scale"))
        except Exception:
            pass

    def reset_to_defaults(self) -> None:
        try:
            self.chk_scan_after_open.setChecked(True)
            self.chk_remember_last_dir.setChecked(True)
            self.chk_anim_autoplay.setChecked(True)
            self.chk_anim_loop.setChecked(True)
            self.chk_anim_keep_state.setChecked(False)
            self.chk_anim_pause_unfocus.setChecked(False)
            self.chk_anim_click_toggle.setChecked(False)
            self.combo_startup_restore.setCurrentIndex(0)
            self.spin_recent_max.setValue(10)
            self.chk_recent_auto_prune.setChecked(True)
            # 정렬 옵션 제거됨
            self.chk_wrap_ends.setChecked(False)
            self.spin_nav_throttle.setValue(100)
            self.chk_film_center.setChecked(True)
            self.combo_zoom_policy.setCurrentIndex(1)
            self.chk_tiff_first_page.setChecked(True)
        except Exception:
            pass


