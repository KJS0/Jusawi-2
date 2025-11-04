from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QDialogButtonBox, QCheckBox, QWidget, QTabWidget, QTableWidget, QTableWidgetItem, QPushButton, QHeaderView, QApplication, QFormLayout, QFileDialog, QDoubleSpinBox, QScrollArea
)
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QKeySequenceEdit
from ..shortcuts.shortcuts_manager import COMMANDS, get_effective_keymap, save_custom_keymap
import os
from .settings import (
    GeneralSettingsPage,
    AISettingsPage,
    PerformanceSettingsPage,
    ViewSettingsPage,
    ColorSettingsPage,
    FullscreenSettingsPage,
)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("설정")
        self._reset_keys_to_defaults = False
        self._key_warning_active = False

        root = QVBoxLayout(self)
        try:
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)
        except Exception:
            pass
        self.tabs = QTabWidget(self)
        root.addWidget(self.tabs)


        # ----- 일반 탭 ----- (모듈화)
        self.general_tab = QWidget(self)
        self.tabs.addTab(self.general_tab, "일반")
        gen_root = QVBoxLayout(self.general_tab)
        try:
            gen_root.setContentsMargins(0, 0, 0, 0)
            gen_root.setSpacing(0)
        except Exception:
            pass
        self.general_page = GeneralSettingsPage(None)
        # 스크롤 영역으로 감싸기
        try:
            _sa = QScrollArea(self.general_tab)
            _sa.setWidgetResizable(True)
            _sa.setFrameShape(QScrollArea.Shape.NoFrame)
            _sa.setWidget(self.general_page)
            gen_root.addWidget(_sa)
        except Exception:
            gen_root.addWidget(self.general_page)
        # 기존 코드 호환을 위한 속성 매핑(외부 로직/필터 등록 재사용)
        for _name in [
            "chk_scan_after_open", "chk_remember_last_dir",
            "combo_startup_restore", "spin_recent_max", "chk_recent_auto_prune",
            "chk_anim_autoplay", "chk_anim_loop", "chk_anim_keep_state",
            "chk_anim_pause_unfocus", "chk_anim_click_toggle",
            "chk_anim_overlay_enable", "chk_anim_overlay_show_index",
            "combo_anim_overlay_pos", "spin_anim_overlay_opacity",
            "combo_sort_mode", "combo_sort_name", "chk_exclude_hidden",
            "chk_wrap_ends", "spin_nav_throttle", "chk_film_center",
            "combo_zoom_policy", "chk_drop_allow_folder", "chk_drop_parent_scan",
            "chk_drop_overlay", "chk_drop_confirm", "spin_drop_threshold",
            "chk_tiff_first_page",
        ]:
            try:
                setattr(self, _name, getattr(self.general_page, _name))
            except Exception:
                pass

        try:
            self.combo_sort_name.currentIndexChanged.connect(lambda _idx: self._on_sort_name_changed())
        except Exception:
            pass

        # ----- AI 탭 ----- (모듈화)
        self.ai_tab = QWidget(self)
        self.tabs.addTab(self.ai_tab, "AI")
        ai_layout = QVBoxLayout(self.ai_tab)
        try:
            ai_layout.setContentsMargins(0, 0, 0, 0)
            ai_layout.setSpacing(0)
        except Exception:
            pass
        self.ai_page = AISettingsPage(None)
        try:
            _sa2 = QScrollArea(self.ai_tab)
            _sa2.setWidgetResizable(True)
            _sa2.setFrameShape(QScrollArea.Shape.NoFrame)
            _sa2.setWidget(self.ai_page)
            ai_layout.addWidget(_sa2)
        except Exception:
            ai_layout.addWidget(self.ai_page)
        for _name in [
            "chk_ai_auto", "chk_ai_auto_drop", "chk_ai_auto_nav", "chk_ai_skip_cached",
            "spin_ai_delay", "combo_ai_language", "combo_ai_tone", "combo_ai_purpose",
            "spin_ai_short_words", "spin_ai_long_chars", "chk_ai_fast_mode",
            "combo_ai_exif_level", "spin_ai_retry_count", "spin_ai_retry_delay", "ed_ai_api_key",
        ]:
            try:
                setattr(self, _name, getattr(self.ai_page, _name))
            except Exception:
                pass

        # TIFF는 일반 페이지 내부에 포함됨

        # ----- 성능/프리패치 탭 ----- (모듈화)
        self.perf_tab = QWidget(self)
        self.tabs.addTab(self.perf_tab, "성능/프리패치")
        perf_layout = QVBoxLayout(self.perf_tab)
        try:
            perf_layout.setContentsMargins(0, 0, 0, 0)
            perf_layout.setSpacing(0)
        except Exception:
            pass
        self.perf_page = PerformanceSettingsPage(None)
        try:
            _sa3 = QScrollArea(self.perf_tab)
            _sa3.setWidgetResizable(True)
            _sa3.setFrameShape(QScrollArea.Shape.NoFrame)
            _sa3.setWidget(self.perf_page)
            perf_layout.addWidget(_sa3)
        except Exception:
            perf_layout.addWidget(self.perf_page)
        for _name in [
            "chk_prefetch_thumbs", "spin_preload_radius", "chk_prefetch_map",
            "combo_preload_direction", "spin_preload_priority", "spin_preload_concurrency",
            "spin_preload_retry", "spin_preload_retry_delay", "spin_upgrade_delay",
            "dbl_preview_headroom", "chk_disable_scaled_below_100", "chk_preserve_visual_size_on_dpr",
            "spin_img_cache_mb", "spin_scaled_cache_mb", "spin_cache_auto_shrink_pct", "spin_cache_gc_interval",
            "spin_thumb_quality", "ed_thumb_dir", "btn_thumb_dir",
            "chk_pregen_scales", "ed_pregen_scales",
            "chk_preload_idle_only", "spin_prefetch_on_dir_enter", "spin_slideshow_prefetch",
        ]:
            try:
                setattr(self, _name, getattr(self.perf_page, _name))
            except Exception:
                pass

        # 단축키 탭 제거
        self._keys_ready = False
        # 단축키 입력부(키 설정) 섹션 제거 유지 — 별도 탭/대화에서만 다루도록 고정

        # ----- 보기 탭 ----- (모듈화)
        self.view_tab = QWidget(self)
        self.tabs.addTab(self.view_tab, "보기")
        view_layout = QVBoxLayout(self.view_tab)
        try:
            view_layout.setContentsMargins(0, 0, 0, 0)
            view_layout.setSpacing(0)
        except Exception:
            pass
        self.view_page = ViewSettingsPage(None)
        try:
            _sa4 = QScrollArea(self.view_tab)
            _sa4.setWidgetResizable(True)
            _sa4.setFrameShape(QScrollArea.Shape.NoFrame)
            _sa4.setWidget(self.view_page)
            view_layout.addWidget(_sa4)
        except Exception:
            view_layout.addWidget(self.view_page)
        for _name in [
            "combo_default_view", "spin_min_scale", "spin_max_scale", "chk_fixed_steps",
            "spin_zoom_step", "spin_precise_step", "chk_smooth", "spin_fit_margin",
            "chk_wheel_requires_ctrl", "chk_alt_precise", "combo_dbl", "combo_mid",
            "chk_refit_on_tf", "chk_anchor_preserve", "chk_preserve_visual_dpr",
        ]:
            try:
                setattr(self, _name, getattr(self.view_page, _name))
            except Exception:
                pass

        # ----- 색상 관리 탭 ----- (모듈화)
        self.color_tab = QWidget(self)
        self.tabs.addTab(self.color_tab, "색상")
        color_layout = QVBoxLayout(self.color_tab)
        try:
            color_layout.setContentsMargins(0, 0, 0, 0)
            color_layout.setSpacing(0)
        except Exception:
            pass
        self.color_page = ColorSettingsPage(None)
        try:
            _sa5 = QScrollArea(self.color_tab)
            _sa5.setWidgetResizable(True)
            _sa5.setFrameShape(QScrollArea.Shape.NoFrame)
            _sa5.setWidget(self.color_page)
            color_layout.addWidget(_sa5)
        except Exception:
            color_layout.addWidget(self.color_page)
        for _name in [
            "chk_icc_ignore", "combo_assumed", "combo_target", "combo_fallback",
            "chk_anim_convert", "chk_thumb_convert",
        ]:
            try:
                setattr(self, _name, getattr(self.color_page, _name))
            except Exception:
                pass
        # ----- 전체화면/오버레이 탭 ----- (모듈화)
        self.fullscreen_tab = QWidget(self)
        self.tabs.addTab(self.fullscreen_tab, "전체화면")
        fs_layout = QVBoxLayout(self.fullscreen_tab)
        try:
            fs_layout.setContentsMargins(0, 0, 0, 0)
            fs_layout.setSpacing(0)
        except Exception:
            pass
        self.fullscreen_page = FullscreenSettingsPage(None)
        try:
            _sa6 = QScrollArea(self.fullscreen_tab)
            _sa6.setWidgetResizable(True)
            _sa6.setFrameShape(QScrollArea.Shape.NoFrame)
            _sa6.setWidget(self.fullscreen_page)
            fs_layout.addWidget(_sa6)
        except Exception:
            fs_layout.addWidget(self.fullscreen_page)
        for _name in [
            "spin_fs_auto_hide", "spin_cursor_hide", "combo_fs_viewmode",
            "chk_fs_show_filmstrip", "chk_fs_safe_exit", "chk_overlay_default",
        ]:
            try:
                setattr(self, _name, getattr(self.fullscreen_page, _name))
            except Exception:
                pass

        # 단축키 탭 제거됨(별도 도움말/다이얼로그에서만 다룸)

        # 하단: 우측 확인/취소만 표시
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        # 기본 설정으로 재설정 버튼(좌측)
        self.btn_reset_all = QPushButton("기본 설정으로 재설정", self)
        try:
            self.btn_reset_all.clicked.connect(self._on_reset_all_defaults)
        except Exception:
            pass
        bottom_row.addWidget(self.btn_reset_all)
        bottom_row.addStretch(1)
        root.addLayout(bottom_row)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        # 확인 버튼: 색상 설정 포함하여 전체 적용
        def _apply_all_and_accept():
            try:
                self._apply_color_settings()
            except Exception:
                pass
            try:
                # 기존 단축키 저장/검증 로직은 별도 _on_accept에 있음 → 호출 유지
                self._on_accept()
            except Exception:
                # 키 탭이 준비되지 않았으면 그냥 닫기
                try:
                    self.accept()
                except Exception:
                    pass
            # 변경된 설정을 즉시 저장/동기화
            try:
                viewer = getattr(self, "_viewer_for_keys", None)
                if viewer is not None and hasattr(viewer, "save_settings"):
                    viewer.save_settings()
            except Exception:
                pass
        buttons.accepted.connect(_apply_all_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _apply_values_from_viewer(self):
        viewer = self.parent() if hasattr(self, 'parent') else None
        if viewer is None:
            return
        # 색상 탭: 페이지 위임 로드
        try:
            if hasattr(self, "color_page"):
                self.color_page.load_from_viewer(viewer)
        except Exception:
            pass

    def _apply_color_settings(self):
        viewer = self.parent() if hasattr(self, 'parent') else None
        if viewer is None:
            return
        # 색상 탭: 페이지 위임 저장
        try:
            if hasattr(self, "color_page"):
                self.color_page.apply_to_viewer(viewer)
        except Exception:
            pass
        # 이미지 서비스에도 즉시 반영
        try:
            if hasattr(viewer, 'image_service') and viewer.image_service is not None:
                svc = viewer.image_service
                svc._icc_ignore_embedded = bool(getattr(viewer, "_icc_ignore_embedded", False))
                svc._assumed_colorspace = str(getattr(viewer, "_assumed_colorspace", "sRGB"))
                svc._preview_target = str(getattr(viewer, "_preview_target", "sRGB"))
                svc._fallback_policy = str(getattr(viewer, "_fallback_policy", "ignore"))
        except Exception:
            pass
        try:
            viewer.save_settings()
        except Exception:
            pass

        # 탭 변경 시 크기 최적화
        try:
            self.tabs.currentChanged.connect(self._on_tab_changed)
        except Exception:
            pass
        # 초기 크기: 일반 탭에 맞춰 컴팩트하게
        try:
            self._on_tab_changed(0)
        except Exception:
            pass

        # 스크롤 영역/레이아웃 구성 이후, 휠 가드용 필터 설치 준비
        try:
            self._wheel_guard_targets = []
        except Exception:
            self._wheel_guard_targets = []
        # 일반 탭 구성 마지막에 각 입력 위젯을 휠가드 대상으로 등록
        try:
            for w in [
                self.spin_recent_max,
                self.combo_startup_restore,
                self.chk_recent_auto_prune,
                self.chk_scan_after_open,
                self.chk_remember_last_dir,
                self.chk_anim_autoplay,
                self.chk_anim_loop,
                self.combo_sort_mode,
                self.combo_sort_name,
                self.chk_exclude_hidden,
                self.chk_wrap_ends,
                self.spin_nav_throttle,
                self.chk_film_center,
                self.combo_zoom_policy,
                self.chk_drop_allow_folder,
                self.chk_drop_parent_scan,
                self.chk_drop_overlay,
                self.chk_drop_confirm,
                self.spin_drop_threshold,
                self.chk_prefetch_thumbs,
                self.spin_preload_radius,
                self.chk_prefetch_map,
                # AI 탭 요소는 별도 등록 아래에서 수행
                self.chk_tiff_first_page,
            ]:
                if w:
                    try:
                        w.installEventFilter(self)
                        self._wheel_guard_targets.append(w)
                    except Exception:
                        pass
        except Exception:
            pass
        # AI 탭 휠 가드 등록
        try:
            for w in [
                self.chk_ai_auto,
                self.chk_ai_auto_drop,
                self.chk_ai_auto_nav,
                self.chk_ai_skip_cached,
                self.spin_ai_delay,
                self.combo_ai_language,
                self.combo_ai_tone,
                self.combo_ai_purpose,
                self.spin_ai_short_words,
                self.spin_ai_long_chars,
                self.chk_ai_fast_mode,
                self.combo_ai_exif_level,
                self.spin_ai_retry_count,
                self.spin_ai_retry_delay,
            ]:
                if w:
                    try:
                        w.installEventFilter(self)
                        self._wheel_guard_targets.append(w)
                    except Exception:
                        pass
        except Exception:
            pass

    # 외부에서 호출: 단축키 탭으로 전환하고 첫 편집기로 포커스 이동
    def focus_shortcuts_tab(self):
        # 단축키 탭이 제거되었으므로 도움말 다이얼로그를 여는 방향으로 유지하거나 무시
        try:
            from .dialogs import open_shortcuts_help  # lazy import
            open_shortcuts_help(self.parent())
        except Exception:
            pass

    # populate from viewer
    def load_from_viewer(self, viewer):
        # viewer 참조 저장(지연 로드시 사용)
        self._viewer_for_keys = viewer
        # 각 탭 페이지 위임 로드
        try:
            if hasattr(self, "color_page"):
                self.color_page.load_from_viewer(viewer)
        except Exception:
            pass
        try:
            if hasattr(self, "general_page"):
                self.general_page.load_from_viewer(viewer)
        except Exception:
            pass
        try:
            if hasattr(self, "ai_page"):
                self.ai_page.load_from_viewer(viewer)
        except Exception:
            pass
        try:
            if hasattr(self, "perf_page"):
                self.perf_page.load_from_viewer(viewer)
        except Exception:
            pass
        try:
            if hasattr(self, "view_page"):
                self.view_page.load_from_viewer(viewer)
        except Exception:
            pass
        try:
            if hasattr(self, "fullscreen_page"):
                self.fullscreen_page.load_from_viewer(viewer)
        except Exception:
            pass
        # 자연 정렬 UI 규칙 동기화
        try:
            self._on_sort_name_changed()
        except Exception:
            pass

    # commit back to viewer (does not save)
    def apply_to_viewer(self, viewer):
        # 키 저장(중복/금지 키 검증 포함)
        mapping = None
        if self._reset_keys_to_defaults:
            # 키 탭이 열려있지 않아도 기본값으로 저장
            mapping = {}
            for cmd in COMMANDS:
                if cmd.lock_key:
                    continue
                mapping[cmd.id] = cmd.default_keys[:1]
        elif getattr(self, "_keys_ready", False):
            mapping = self._collect_mapping_from_ui(validate_against_fixed=True)
            if mapping is None:
                return
        if mapping is not None:
            save_custom_keymap(getattr(viewer, "settings", None), mapping)
        self._reset_keys_to_defaults = False
        # 각 탭 페이지 위임 적용
        try:
            if hasattr(self, "general_page"):
                self.general_page.apply_to_viewer(viewer)
        except Exception:
            pass
        try:
            if hasattr(self, "ai_page"):
                self.ai_page.apply_to_viewer(viewer)
        except Exception:
            pass
        try:
            if hasattr(self, "view_page"):
                self.view_page.apply_to_viewer(viewer)
        except Exception:
            pass
        try:
            if hasattr(self, "fullscreen_page"):
                self.fullscreen_page.apply_to_viewer(viewer)
        except Exception:
            pass
        try:
            if hasattr(self, "perf_page"):
                self.perf_page.apply_to_viewer(viewer)
                # 썸네일 캐시 즉시 반영(필름스트립 생성된 경우)
            if hasattr(viewer, 'filmstrip') and viewer.filmstrip is not None and getattr(viewer.filmstrip, '_cache', None) is not None:
                try:
                        viewer.filmstrip._cache.quality = int(getattr(viewer, "_thumb_cache_quality", 85))
                        if getattr(viewer, "_thumb_cache_dir", ""):
                            viewer.filmstrip._cache.root = getattr(viewer, "_thumb_cache_dir", "")
                except Exception:
                    pass
        except Exception:
            pass

    def _on_sort_name_changed(self):
        try:
            is_explorer = (self.combo_sort_name.currentIndex() == 0)
            # 윈도우 탐색기 정렬 선택 시 자동으로 정렬 기준을 파일명으로 강제하고 비활성화
            if is_explorer:
                try:
                    if int(self.combo_sort_mode.currentIndex()) != 1:
                        self.combo_sort_mode.setCurrentIndex(1)  # 파일명
                except Exception:
                    self.combo_sort_mode.setCurrentIndex(1)
                self.combo_sort_mode.setEnabled(False)
            else:
                self.combo_sort_mode.setEnabled(True)
        except Exception:
            pass

    # ----- helpers -----
    def _on_reset_defaults(self):
        # 단축키 탭 기본값: 레지스트리의 default_keys로 되돌림(없으면 비움)
        # 테이블은 COMMANDS 순으로 채워져 있음
        if getattr(self, "_keys_ready", False) and hasattr(self, "keys_table"):
            row = 0
            for cmd in COMMANDS:
                editor = self.keys_table.cellWidget(row, 3) if row < self.keys_table.rowCount() else None
                if isinstance(editor, QKeySequenceEdit):
                    defaults = cmd.default_keys[:]
                    editor.setKeySequence(QKeySequence(defaults[0]) if defaults else QKeySequence())
                row += 1
        self._reset_keys_to_defaults = True

    def _on_reset_all_defaults(self):
        # 모든 탭 기본값으로 초기화하고 즉시 적용/저장
        try:
            if hasattr(self, "general_page"):
                self.general_page.reset_to_defaults()
            if hasattr(self, "ai_page"):
                self.ai_page.reset_to_defaults()
            if hasattr(self, "perf_page"):
                self.perf_page.reset_to_defaults()
            if hasattr(self, "view_page"):
                self.view_page.reset_to_defaults()
            if hasattr(self, "fullscreen_page"):
                self.fullscreen_page.reset_to_defaults()
            # 색상 탭은 사용자 환경에 따라 보수적으로 유지
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
        # 적용 전에 유효성 검사 수행 후 통과하면 accept
        # viewer에 직접 접근하지 않고 저장할 매핑만 검증
        if not self._reset_keys_to_defaults and not getattr(self, "_keys_ready", False):
            # 키 탭이 초기화되지 않았다면 단축키 검증은 건너뜀
            self.accept()
            return
        mapping = self._collect_mapping_from_ui(validate_against_fixed=True)
        if mapping is None:
            return
        self.accept()

    def _on_import_yaml(self):
        # 제거됨
        return

    def _on_export_yaml(self):
        # 제거됨
        return

    def _on_tab_changed(self, index: int):
        # 일반 탭은 작게, 단축키 탭은 내용 기반으로 크게
        try:
            tab_text = self.tabs.tabText(index)
        except Exception:
            tab_text = ""
        if tab_text == "단축키":
            # 지연 초기화
            if not getattr(self, "_keys_ready", False):
                try:
                    self._build_keys_tab()
                except Exception:
                    pass
            try:
                # 단축키 탭 크기 재조정
                header = self.keys_table.horizontalHeader()
                header.resizeSections(QHeaderView.ResizeMode.ResizeToContents)
            except Exception:
                pass
            # 기본 넉넉한 크기
            self.resize(max(860, self.width()), max(520, self.height()))
        else:
            # 일반 탭: 컴팩트 크기 강제(작게도 가능하도록 최소 크기 낮추기)
            try:
                self.setMinimumSize(420, 280)
            except Exception:
                pass
            self.resize(520, 360)

        # 자연 정렬 UI 규칙 동기화
        try:
            self.combo_sort_mode.setEnabled(self.combo_sort_name.currentIndex() != 0)
        except Exception:
            pass

    def _build_keys_tab(self):
        # 실제 키 탭 UI 구성 및 데이터 로드
        keys_layout = QVBoxLayout(self.keys_tab)
        self.keys_table = QTableWidget(0, 4, self.keys_tab)
        self.keys_table.setHorizontalHeaderLabels(["조건", "명령", "설명", "단축키"])
        try:
            # 표 셀 선택/편집 비활성화(편집기는 별도 위젯으로 사용)
            from PyQt6.QtWidgets import QAbstractItemView
            self.keys_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.keys_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        except Exception:
            pass
        keys_layout.addWidget(self.keys_table)
        self._keys_ready = True
        # 데이터 채우기
        viewer = getattr(self, "_viewer_for_keys", None)
        eff = get_effective_keymap(getattr(viewer, "settings", None))
        self.keys_table.setRowCount(0)
        self._key_editors = []
        self._editor_meta = {}
        for cmd in [c for c in COMMANDS if c.id != "reset_to_100"]:
            row = self.keys_table.rowCount()
            self.keys_table.insertRow(row)
            cond_text = "고정" if cmd.lock_key else "-"
            it0 = QTableWidgetItem(cond_text)
            it1 = QTableWidgetItem(cmd.label)
            it2 = QTableWidgetItem(cmd.desc)
            # 입력/선택 불가, 표시만
            try:
                it0.setFlags(Qt.ItemFlag.ItemIsEnabled)
                it1.setFlags(Qt.ItemFlag.ItemIsEnabled)
                it2.setFlags(Qt.ItemFlag.ItemIsEnabled)
            except Exception:
                pass
            self.keys_table.setItem(row, 0, it0)
            self.keys_table.setItem(row, 1, it1)
            self.keys_table.setItem(row, 2, it2)
            # 읽기 전용 텍스트로 표시(여러 키가 있을 경우 ; 로 연결)
            seqs = eff.get(cmd.id, []) or []
            txt = "; ".join([str(s) for s in seqs]) if seqs else ""
            it3 = QTableWidgetItem(txt)
            try:
                it3.setFlags(Qt.ItemFlag.ItemIsEnabled)
            except Exception:
                pass
            self.keys_table.setItem(row, 3, it3)
        # 컬럼/크기 조정
        try:
            header = self.keys_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
            self.keys_table.resizeColumnsToContents()
        except Exception:
            pass

    def _collect_mapping_from_ui(self, validate_against_fixed: bool = True):
        from PyQt6.QtWidgets import QMessageBox
        # 열람 전용: 현재 테이블의 텍스트를 그대로 저장 대상으로 수집
        mapping: dict[str, list[str]] = {}
        if getattr(self, "_keys_ready", False) and hasattr(self, "keys_table"):
            for row in range(self.keys_table.rowCount()):
                cmd = COMMANDS[row]
                item = self.keys_table.item(row, 3)
                txt = item.text().strip() if item else ""
                mapping[cmd.id] = [txt] if txt else []

        # 유효성 검사: 중복, 고정키 충돌, 예약키 금지
        if validate_against_fixed:
            used: dict[str, str] = {}
            # 고정키(F1, Escape 등) 집합
            fixed_keys = set()
            for cmd in COMMANDS:
                if cmd.lock_key:
                    for k in cmd.default_keys:
                        if k:
                            fixed_keys.add(k)
            # 예약키(확장 가능)
            # Space는 애니메이션 전용 고정키이므로 일반 명령으로 저장 시 충돌 처리
            reserved = {"Alt+F4"}

            # 중복 및 금지 검사
            for cmd in COMMANDS:
                if cmd.lock_key:
                    continue
                keys = mapping.get(cmd.id, []) or []
                if not keys:
                    continue
                k = keys[0]
                if k in reserved:
                    QMessageBox.warning(self, "단축키 오류", f"예약된 단축키 '{k}' 는 사용할 수 없습니다.")
                    return None
                if k in fixed_keys:
                    # Space는 조용히 무시
                    if k == "Space":
                        mapping[cmd.id] = []
                        continue
                    QMessageBox.warning(self, "단축키 충돌", f"'{k}' 는 시스템/고정 단축키와 충돌합니다.")
                    return None
                if k in used:
                    # 다른 명령과 중복
                    other = used[k]
                    QMessageBox.warning(self, "단축키 중복", f"'{k}' 가 '{other}' 와(과) 중복됩니다. 다른 키를 지정하세요.")
                    return None
                used[k] = cmd.label

        return mapping

    def _normalize_single_key(self, editor: QKeySequenceEdit):
        try:
            seq = editor.keySequence()
            if not seq or seq.isEmpty():
                return
            # 표준 텍스트로 변환
            try:
                from PyQt6.QtGui import QKeySequence as _QS
                text = seq.toString(_QS.SequenceFormat.PortableText)
            except Exception:
                text = seq.toString()
            # 여러 파트가 "," 로 구분되어 들어올 수 있으므로 마지막 파트만 유지
            parts = [p.strip() for p in text.split(',') if p.strip()]
            last_part = parts[-1] if parts else ''
            # 수정키만 있는 조합 문자열 집합(PortableText 기준)
            mod_only = {
                "Ctrl", "Shift", "Alt", "Meta",
                "Ctrl+Shift", "Ctrl+Alt", "Ctrl+Meta",
                "Shift+Alt", "Shift+Meta", "Alt+Meta",
                "Ctrl+Shift+Alt", "Ctrl+Shift+Meta", "Ctrl+Alt+Meta",
                "Shift+Alt+Meta", "Ctrl+Shift+Alt+Meta"
            }
            if last_part in mod_only:
                editor.blockSignals(True)
                editor.setKeySequence(QKeySequence())
                editor.blockSignals(False)
                return
            # 최종 1개 시퀀스로 고정
            editor.blockSignals(True)
            editor.setKeySequence(QKeySequence(last_part))
            editor.blockSignals(False)
        except Exception:
            pass

    def _on_key_changed(self, editor: QKeySequenceEdit, defaults: list, row_idx: int):
        # 정규화 수행
        self._normalize_single_key(editor)
        # 중복/금지/고정 충돌 즉시 검사
        try:
            cur_txt = self._seq_to_text(editor.keySequence())
            def_txt = self._normalize_default_text(defaults)
            # 빈 값이면 버튼만 동기화하고 끝
            if not cur_txt:
                btn = self.keys_table.cellWidget(row_idx, 4)
                if isinstance(btn, QPushButton):
                    btn.setEnabled(False)
                return
            # 고정키/예약키 집합
            fixed_keys = set()
            for cmd in COMMANDS:
                if cmd.lock_key:
                    for k in cmd.default_keys:
                        if k:
                            fixed_keys.add(k)
            reserved = {"Alt+F4"}
            # 현재 행 외 사용중 키 수집
            used = set()
            for r in range(self.keys_table.rowCount()):
                if r == row_idx:
                    continue
                item = self.keys_table.item(r, 3)
                k = item.text().strip() if item else ""
                if k:
                    used.add(k)
            if cur_txt in reserved:
                self._show_key_warning("단축키 오류", f"예약된 단축키 '{cur_txt}' 는 사용할 수 없습니다.", editor)
                meta = getattr(self, "_editor_meta", {}).get(editor, {})
                prev_txt = meta.get("prev", "") if isinstance(meta, dict) else ""
                editor.blockSignals(True)
                editor.setKeySequence(QKeySequence(prev_txt) if prev_txt else QKeySequence())
                editor.blockSignals(False)
                cur_txt = prev_txt
            elif cur_txt in fixed_keys:
                # Space는 조용히 무시, 그 외에는 경고
                if cur_txt == "Space":
                    meta = getattr(self, "_editor_meta", {}).get(editor, {})
                    prev_txt = meta.get("prev", "") if isinstance(meta, dict) else ""
                    editor.blockSignals(True)
                    editor.setKeySequence(QKeySequence(prev_txt) if prev_txt else QKeySequence())
                    editor.blockSignals(False)
                    cur_txt = prev_txt
                else:
                    self._show_key_warning("단축키 충돌", f"'{cur_txt}' 는 시스템/고정 단축키와 충돌합니다.", editor)
                    meta = getattr(self, "_editor_meta", {}).get(editor, {})
                    prev_txt = meta.get("prev", "") if isinstance(meta, dict) else ""
                    editor.blockSignals(True)
                    editor.setKeySequence(QKeySequence(prev_txt) if prev_txt else QKeySequence())
                    editor.blockSignals(False)
                    cur_txt = prev_txt
            elif cur_txt in used:
                self._show_key_warning("단축키 중복", f"'{cur_txt}' 가 이미 다른 명령에 할당되어 있습니다.", editor)
                meta = getattr(self, "_editor_meta", {}).get(editor, {})
                prev_txt = meta.get("prev", "") if isinstance(meta, dict) else ""
                editor.blockSignals(True)
                editor.setKeySequence(QKeySequence(prev_txt) if prev_txt else QKeySequence())
                editor.blockSignals(False)
                cur_txt = prev_txt
            # 기본값 버튼 상태 업데이트
            btn = self.keys_table.cellWidget(row_idx, 4)
            if isinstance(btn, QPushButton):
                btn.setEnabled(bool(def_txt) and cur_txt != def_txt)
        except Exception:
            pass

    def _show_key_warning(self, title: str, text: str, editor: QKeySequenceEdit | None):
        # 동일 시점 중복 팝업 방지 및 포커스 이동 보장
        try:
            if self._key_warning_active:
                return
            self._key_warning_active = True
            from PyQt6.QtWidgets import QMessageBox
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle(title)
            box.setText(text)
            box.setWindowModality(Qt.WindowModality.ApplicationModal)
            try:
                box.raise_()
                box.activateWindow()
            except Exception:
                pass
            box.exec()
        except Exception:
            pass
        finally:
            self._key_warning_active = False
            try:
                if editor is not None:
                    editor.setFocus()
            except Exception:
                pass

    def _seq_to_text(self, seq: QKeySequence) -> str:
        try:
            from PyQt6.QtGui import QKeySequence as _QS
            return seq.toString(_QS.SequenceFormat.PortableText) if seq and not seq.isEmpty() else ""
        except Exception:
            return seq.toString() if seq and not seq.isEmpty() else ""

    def _normalize_default_text(self, defaults: list) -> str:
        if not defaults:
            return ""
        return str(defaults[0])

    # 이벤트 필터: 배타 포커스 + Backspace/Delete로 해제 지원
    def eventFilter(self, obj, event):
        try:
            from PyQt6.QtCore import QEvent  # type: ignore
            et = event.type()
            # 키 시퀀스 에디터 기존 로직 유지
            if isinstance(obj, QKeySequenceEdit):
                if et == QEvent.Type.FocusIn:
                    for ed in getattr(self, "_key_editors", []) or []:
                        if ed is not obj and ed.hasFocus():
                            try:
                                ed.clearFocus()
                            except Exception:
                                pass
                elif et == QEvent.Type.KeyPress:
                    key = getattr(event, 'key', None)
                    if key and int(key()) in (0x01000003, 0x01000007):  # Backspace/Delete
                        obj.setKeySequence(QKeySequence())
                        meta = getattr(self, "_editor_meta", {}).get(obj, None)
                        if meta is not None:
                            self._on_key_changed(obj, meta.get("defaults", []), int(meta.get("row", 0)))
                        return True
                return super().eventFilter(obj, event)
            # 휠 가드: 포커스 없는 경우 스핀/콤보/체크는 값 변경 금지
            if et == QEvent.Type.Wheel:
                from PyQt6.QtWidgets import QAbstractSpinBox, QComboBox, QCheckBox  # type: ignore
                if isinstance(obj, (QAbstractSpinBox, QComboBox, QCheckBox)):
                    if not obj.hasFocus():
                        return True
        except Exception:
            pass
        return super().eventFilter(obj, event)


class _LabeledRow(QHBoxLayout):
    def __init__(self, label_text: str):
        super().__init__()
        try:
            self.setContentsMargins(0, 0, 0, 0)
            self.setSpacing(0)
        except Exception:
            pass
        self._label = QLabel(label_text)
        try:
            self._label.setMinimumWidth(0)
            self._label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        except Exception:
            pass
        self.addWidget(self._label)
        self._holder = QWidget()
        self._holder_layout = QHBoxLayout(self._holder)
        try:
            self._holder_layout.setContentsMargins(0, 0, 0, 0)
            self._holder_layout.setSpacing(0)
        except Exception:
            pass
        self.addWidget(self._holder, 1)

    def set_widget(self, w: QWidget):
        # clear holder
        while self._holder_layout.count():
            item = self._holder_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._holder_layout.addWidget(w)


def _spin(min_v: int, max_v: int, value: int) -> QSpinBox:
    s = QSpinBox()
    s.setRange(min_v, max_v)
    s.setValue(value)
    return s


