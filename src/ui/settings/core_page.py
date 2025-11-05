from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QCheckBox, QComboBox, QFormLayout, QLineEdit
)

from .base import SettingsPage


class CoreSettingsPage(SettingsPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        try:
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(8)
        except Exception:
            pass

        # (통합) 일반 섹션 라벨 제거
        self.chk_scan_after_open = QCheckBox("열기 후 폴더 자동 스캔", self)
        self.chk_remember_last_dir = QCheckBox("마지막 사용 폴더 기억", self)
        form_general = QFormLayout()
        self.combo_startup_restore = QComboBox(self)
        # never/always/ask 순서
        self.combo_startup_restore.addItems(["복원 안 함", "항상 복원", "묻기"])
        root.addWidget(self.chk_scan_after_open)
        root.addWidget(self.chk_remember_last_dir)
        # 사용자 요구 순서에 맞춰 순서를 재배치
        # 1) 애니메이션 두 항목, 2) 사진 미리 불러오기, 3) 세션 복원, 4) 보기/색상, 5) API 키

        # (통합) 애니메이션 라벨 제거
        self.chk_anim_autoplay = QCheckBox("애니메이션 자동 재생", self)
        self.chk_anim_loop = QCheckBox("애니메이션 반복 재생", self)
        root.addWidget(self.chk_anim_autoplay)
        root.addWidget(self.chk_anim_loop)

        # (통합) 프리패치/키 라벨 제거
        self.chk_prefetch_thumbs = QCheckBox("사진 미리 불러오기", self)
        form_keys = QFormLayout()
        self.ed_kakao_key = QLineEdit(self)
        self.ed_kakao_key.setPlaceholderText("카카오 REST API 키")
        self.ed_google_key = QLineEdit(self)
        self.ed_google_key.setPlaceholderText("구글 지도 API 키")
        form_keys.addRow("카카오 API 키", self.ed_kakao_key)
        form_keys.addRow("구글 API 키", self.ed_google_key)
        root.addWidget(self.chk_prefetch_thumbs)

        # (통합) 보기/색상 라벨 제거
        form_view = QFormLayout()
        self.combo_default_view = QComboBox(self)
        self.combo_default_view.addItems(["화면 맞춤", "가로 맞춤", "세로 맞춤", "실제 크기"])
        self.combo_preview_target = QComboBox(self)
        self.combo_preview_target.addItems(["sRGB", "Display P3", "Adobe RGB"])
        form_view.addRow("기본 보기 모드", self.combo_default_view)
        form_view.addRow("미리보기 색영역", self.combo_preview_target)
        # 세션 복원은 목록 중간에 위치
        form_general.addRow("시작 시 세션 복원", self.combo_startup_restore)
        root.addLayout(form_general)
        root.addLayout(form_view)

        # (통합) AI 라벨 제거
        form_ai = QFormLayout()
        self.ed_ai_api_key = QLineEdit(self)
        self.ed_ai_api_key.setPlaceholderText("OpenAI API Key")
        form_ai.addRow("OpenAI API 키", self.ed_ai_api_key)
        root.addLayout(form_ai)
        # 마지막으로 지도 API 키
        root.addLayout(form_keys)

    def load_from_viewer(self, viewer: Any) -> None:  # noqa: ANN401
        self.chk_scan_after_open.setChecked(bool(getattr(viewer, "_open_scan_dir_after_open", True)))
        self.chk_remember_last_dir.setChecked(bool(getattr(viewer, "_remember_last_open_dir", True)))
        pol = str(getattr(viewer, "_startup_restore_policy", "never"))
        self.combo_startup_restore.setCurrentIndex({"never":0, "always":1, "ask":2}.get(pol, 0))
        self.chk_anim_autoplay.setChecked(bool(getattr(viewer, "_anim_autoplay", True)))
        self.chk_anim_loop.setChecked(bool(getattr(viewer, "_anim_loop", True)))
        self.chk_prefetch_thumbs.setChecked(bool(getattr(viewer, "_enable_thumb_prefetch", True)))
        self.ed_kakao_key.setText(str(getattr(viewer, "_map_kakao_api_key", "") or ""))
        self.ed_google_key.setText(str(getattr(viewer, "_map_google_api_key", "") or ""))
        dvm = str(getattr(viewer, "_default_view_mode", "fit"))
        self.combo_default_view.setCurrentIndex({"fit":0, "fit_width":1, "fit_height":2, "actual":3}.get(dvm, 0))
        tgt = str(getattr(viewer, "_preview_target", "sRGB"))
        self.combo_preview_target.setCurrentIndex({"sRGB":0, "Display P3":1, "Adobe RGB":2}.get(tgt, 0))
        self.ed_ai_api_key.setText(str(getattr(viewer, "_ai_openai_api_key", "") or ""))

    def apply_to_viewer(self, viewer: Any) -> None:  # noqa: ANN401
        viewer._open_scan_dir_after_open = bool(self.chk_scan_after_open.isChecked())
        viewer._remember_last_dir = bool(self.chk_remember_last_dir.isChecked())
        idx = int(self.combo_startup_restore.currentIndex())
        viewer._startup_restore_policy = ("never" if idx == 0 else ("always" if idx == 1 else "ask"))
        viewer._anim_autoplay = bool(self.chk_anim_autoplay.isChecked())
        viewer._anim_loop = bool(self.chk_anim_loop.isChecked())
        viewer._enable_thumb_prefetch = bool(self.chk_prefetch_thumbs.isChecked())
        viewer._map_kakao_api_key = str(self.ed_kakao_key.text()).strip()
        viewer._map_google_api_key = str(self.ed_google_key.text()).strip()
        dvm = int(self.combo_default_view.currentIndex())
        viewer._default_view_mode = ("fit" if dvm == 0 else ("fit_width" if dvm == 1 else ("fit_height" if dvm == 2 else "actual")))
        tgt = int(self.combo_preview_target.currentIndex())
        viewer._preview_target = ("sRGB" if tgt == 0 else ("Display P3" if tgt == 1 else "Adobe RGB"))
        viewer._ai_openai_api_key = str(self.ed_ai_api_key.text()).strip()

    def reset_to_defaults(self) -> None:
        self.chk_scan_after_open.setChecked(True)
        self.chk_remember_last_dir.setChecked(True)
        self.combo_startup_restore.setCurrentIndex(0)
        self.chk_anim_autoplay.setChecked(True)
        self.chk_anim_loop.setChecked(True)
        self.chk_prefetch_thumbs.setChecked(True)
        self.combo_default_view.setCurrentIndex(0)
        self.combo_preview_target.setCurrentIndex(0)
        self.ed_kakao_key.setText("")
        self.ed_google_key.setText("")
        self.ed_ai_api_key.setText("")


