import os
import sys
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QApplication, QMainWindow, QLabel, QSizePolicy, QMenu, QTextEdit, QGraphicsOpacityEffect  # type: ignore[import]
from PyQt6.QtCore import QTimer, Qt, QSettings, QPointF, QEvent, QUrl, pyqtSignal, QObject, QPoint, QEasingCurve, QPropertyAnimation  # type: ignore[import]
from PyQt6.QtGui import QKeySequence, QShortcut, QImage, QAction, QPixmap, QMovie, QColorSpace, QDesktopServices  # type: ignore[import]

from .image_view import ImageView
from .filmstrip import FilmstripView
from . import commands as viewer_cmd
from . import animation_controller as anim
from .state import TransformState, ViewerState
from . import display_scaling as ds
from .dirty_guard import handle_dirty_before_action as dg_handle
from .dir_utils import rescan_current_dir as dir_rescan
from . import dir_scan as dir_scan_ext
from .theme import apply_ui_theme_and_spacing as apply_theme
from . import file_commands as file_cmd
from . import image_loader as img_loader
from . import history as hist
from ..utils.file_utils import open_file_dialog_util, cleanup_leftover_temp_and_backup
from ..utils.delete_utils import move_to_trash_windows
from ..storage.mru_store import normalize_path, update_mru
from ..storage.mru_store import process_mru
from .z_dnd_bridge import handle_dropped_files as dnd_handle_files
from .z_dnd_bridge import handle_dropped_folders as dnd_handle_folders
from .z_dnd_bridge import drag_enter as dnd_drag_enter, drag_move as dnd_drag_move, drop as dnd_drop
from .fullscreen_controller import enter_fullscreen as fs_enter_fullscreen, exit_fullscreen as fs_exit_fullscreen
from .menu_builder import rebuild_recent_menu as build_recent_menu
from .menu_builder import rebuild_log_menu as build_log_menu
from ..shortcuts.shortcuts_manager import apply_shortcuts as apply_shortcuts_ext
from ..dnd.event_filters import DnDEventFilter
from ..services.session_service import save_last_session as save_session_ext, restore_last_session as restore_session_ext
from ..storage.settings_store import load_settings as load_settings_ext, save_settings as save_settings_ext
from ..utils.navigation import NavigationController, show_prev_image as nav_show_prev_image, show_next_image as nav_show_next_image, load_image_at_current_index as nav_load_image_at_current_index, update_button_states as nav_update_button_states
from .title_status import update_window_title as ts_update_window_title, update_status_left as ts_update_status_left, update_status_right as ts_update_status_right
from . import info_panel
from .layout_builder import build_top_and_status_bars
from .shortcuts_utils import set_global_shortcuts_enabled
from .view_utils import clear_display as view_clear_display
from .utils_misc import clamp as util_clamp, enable_dnd_on as util_enable_dnd_on, setup_global_dnd as util_setup_global_dnd, handle_escape as util_handle_escape
from . import lifecycle
from . import log_actions
from ..dnd.dnd_setup import setup_global_dnd as setup_global_dnd_ext, enable_dnd as enable_dnd_ext
from ..services.image_service import ImageService
from . import dialogs as dlg
from ..utils.logging_setup import get_logger, get_log_dir, export_logs_zip, suggest_logs_zip_name, open_logs_folder
from ..services.ratings_store import get_image as ratings_get_image, upsert_image as ratings_upsert_image  # type: ignore
from . import rating_bar
from . import transform_ui
from . import map_click

class JusawiViewer(QMainWindow):
    def __init__(self, skip_session_restore: bool = False):
        super().__init__()
        # 세션 복원 스킵 여부(명령줄로 파일/폴더가 지정된 경우 사용)
        self._skip_session_restore = bool(skip_session_restore)
        self.setWindowTitle("Jusawi")
        self.log = get_logger("ui.JusawiViewer")
        # 창 기본 초기화 크기/위치 정책: 복원 비활성화, 기본 크기 지정
        self._restore_window_geometry = False
        try:
            self.resize(1280, 800)
            self.move(80, 60)
        except Exception:
            pass

        self.current_image_path = None
        self.image_files_in_dir = []
        self.current_image_index = -1
        self.load_successful = False
        # 줌 상태(레거시 변수는 유지하되, ImageView가 관리)
        self.scale_factor = 1.0
        self.fit_mode = True
        self.min_scale = 0.01
        self.max_scale = 16.0
        
        # 전체화면 및 슬라이드쇼 상태 관리
        self.is_fullscreen = False
        self.is_slideshow_active = False
        self.button_layout = None  # 나중에 설정
        self.previous_window_state = None  # 전체화면 이전 상태 저장

        # 상태 캐시(우측 상태 표시용)
        self._last_cursor_x = 0
        self._last_cursor_y = 0
        self._last_scale = 1.0
        self._last_view_mode = 'fit'
        self._last_center = QPointF(0.0, 0.0)
        # 설정 기본값
        self._refit_on_transform = True
        self._fit_margin_pct = 0
        self._wheel_zoom_requires_ctrl = True
        self._wheel_zoom_alt_precise = True
        self._use_fixed_zoom_steps = False
        self._zoom_step_factor = 1.25
        self._precise_zoom_step_factor = 1.1
        self._double_click_action = 'toggle'  # toggle|fit|fit_width|fit_height|actual|none
        self._middle_click_action = 'none'    # none|toggle|fit|actual
        # 제스처 내비게이션(트랙패드 두 손가락 좌우 스와이프 -> 이전/다음)
        self._gesture_nav_enabled = True
        self._gesture_nav_threshold = 240  # 누적 delta 임계값(약 2단계)
        self._gesture_nav_cooldown_ms = 300
        self._gesture_accum_x = 0
        self._gesture_last_trigger_ms = 0

        # 편집/변환 상태
        self._tf_rotation = 0  # 0/90/180/270
        self._tf_flip_h = False
        self._tf_flip_v = False
        self._is_dirty = False
        self._save_policy = 'discard'  # 'discard' | 'ask' | 'overwrite' | 'save_as'
        self._jpeg_quality = 95
        # 편집 히스토리(Undo/Redo)
        self._history_undo = []  # list[tuple[int,bool,bool]]
        self._history_redo = []  # list[tuple[int,bool,bool]]
        # 애니메이션 재생 상태
        self._anim_is_playing = False
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(100)  # 기본 10fps
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._movie = None  # type: QMovie | None
        # 옵션: QMovie 프레임도 sRGB로 변환(성능 비용 존재). 기본 활성화
        self._convert_movie_frames_to_srgb = True

        # 설정 저장(QSettings)
        self.settings = QSettings("Jusawi", "Jusawi")
        self.recent_files = []  # list[dict]
        self.recent_folders = []  # list[dict]
        self.last_open_dir = ""

        # QMainWindow의 중앙 위젯 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # 배경/스타일은 theme.apply_ui_theme_and_spacing에서 일괄 적용
        central_widget.setStyleSheet("")
        # Drag & Drop 비활성화
        self.setAcceptDrops(False)
        central_widget.setAcceptDrops(False)
        try:
            central_widget.removeEventFilter(self)
        except Exception:
            pass

        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self._normal_margins = (5, 5, 5, 5)

        # ImageView (QGraphicsView 기반)
        self.image_display_area = ImageView(central_widget)
        self.image_display_area.scaleChanged.connect(self.on_scale_changed)
        self.image_display_area.cursorPosChanged.connect(self.on_cursor_pos_changed)
        # 명시적 min/max 스케일 설정
        self.image_display_area.set_min_max_scale(self.min_scale, self.max_scale)
        # ImageView 및 내부 뷰포트 DnD 비활성화
        self.image_display_area.setAcceptDrops(False)
        try:
            self.image_display_area.viewport().setAcceptDrops(False)
            self.image_display_area.viewport().removeEventFilter(self)
        except Exception:
            pass
        try:
            self.image_display_area.removeEventFilter(self)
        except Exception:
            pass
        # 메인 콘텐츠 영역: 이미지 + 정보 패널(우측) → QSplitter로 가변 너비
        from PyQt6.QtWidgets import QSplitter  # type: ignore[import]
        self.content_widget = QWidget(central_widget)
        self.content_layout = QHBoxLayout(self.content_widget)
        try:
            self.content_layout.setContentsMargins(0, 0, 0, 0)
            self.content_layout.setSpacing(6)
        except Exception:
            pass
        # 스플리터 제거 후, 이미지 영역은 스프링으로 확장
        self.content_layout.addWidget(self.image_display_area, 1)
        # 정보 패널 (텍스트 + 지도 미리보기) → 내부도 QSplitter로 가변 높이
        self.info_panel = QWidget(self.content_widget)
        self.info_panel_layout = QVBoxLayout(self.info_panel)
        try:
            self.info_panel_layout.setContentsMargins(0, 0, 0, 0)
            self.info_panel_layout.setSpacing(6)
        except Exception:
            pass
        self.info_text = QTextEdit(self.info_panel)
        try:
            self.info_text.setReadOnly(True)
            self.info_text.setMinimumWidth(280)
        except Exception:
            pass
        self.info_map_label = QLabel(self.info_panel)
        try:
            self.info_map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.info_map_label.setText("여기에 지도가 표시됩니다.")
            try:
                self.info_text.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            except Exception:
                pass
        except Exception:
            pass
        # 스플리터 롤백: 직접 위젯 배치
        self.info_panel_layout.addWidget(self.info_text)
        self.info_panel_layout.addWidget(self.info_map_label)
        self.info_panel.setVisible(False)
        self.content_layout.addWidget(self.info_panel)
        # 수평 레이아웃으로 이미지/정보 패널 표시
        self.main_layout.addWidget(self.content_widget, 1)

        # 하단 필름 스트립
        self.filmstrip = FilmstripView(self)
        try:
            self.filmstrip.currentIndexChanged.connect(self._on_filmstrip_index_changed)
        except Exception:
            pass
        # 하단 필름 스트립을 고정 배치로 롤백
        self.main_layout.addWidget(self.filmstrip, 0)
        try:
            # 시작 시에는 이미지가 없으므로 보이지 않게
            self.filmstrip.setVisible(False)
        except Exception:
            pass
        # 전체 캐시 리셋 시 썸네일 메모리도 함께 초기화할 수 있도록 핸들러 제공
        try:
            self._clear_filmstrip_cache = lambda: getattr(self.filmstrip, "_cache", None) and self.filmstrip._cache.clear()
        except Exception:
            self._clear_filmstrip_cache = lambda: None

        # 필름 스트립 컨트롤 바(별점/플래그)
        try:
            rating_bar.create(self)
        except Exception:
            pass
        # 시작 시 평점 바 숨김
        try:
            if hasattr(self, '_rating_flag_bar') and self._rating_flag_bar is not None:
                self._rating_flag_bar.setVisible(False)
        except Exception:
            pass

        # 이미지 서비스
        self.image_service = ImageService(self)
        self.image_service.loaded.connect(self._on_image_loaded)
        # 마지막 DPR 기억(배율 일관성 유지용)
        try:
            self._last_dpr = float(self.image_display_area.viewport().devicePixelRatioF())
        except Exception:
            self._last_dpr = 1.0
        # 자유 줌 상태에서 DPR 변경 시 배율(%)을 그대로 유지: True면 시각 크기 보존(배율 변함), False면 배율 고정
        # 사진 뷰어 UX 기준: 배율 표시는 불변이 더 직관적 -> 기본 False
        self._preserve_visual_size_on_dpr_change = False
        # DPR 전환 가드 및 해제 타이머(중복 재적용/재디바운스 방지)
        self._in_dpr_transition = False
        self._dpr_guard_timer = QTimer(self)
        self._dpr_guard_timer.setSingleShot(True)
        self._dpr_guard_timer.timeout.connect(lambda: setattr(self, "_in_dpr_transition", False))
        # 썸네일(다운샘플) 표시 후 원본으로 1회 업그레이드 타이머
        self._fullres_upgrade_timer = QTimer(self)
        self._fullres_upgrade_timer.setSingleShot(True)
        self._fullres_upgrade_timer.timeout.connect(self._upgrade_to_fullres_if_needed)
        # 100% 이하 배율에서도 스케일 디코딩/캐시를 적극 활용하여 대용량 표시 성능 개선
        self._disable_scaled_cache_below_100 = False
        # DPR/모니터 변경 시 재적용 트리거 설정 (표시 후에도 보장되도록 별도 보조 메서드 사용)
        self._screen_signal_connected = False
        try:
            self._ensure_screen_signal_connected()
        except Exception:
            pass
        # 뷰포트 스케일 적용 디바운스 타이머
        self._scale_apply_timer = QTimer(self)
        self._scale_apply_timer.setSingleShot(True)
        self._scale_apply_timer.setInterval(30)
        self._scale_apply_timer.timeout.connect(self._apply_scaled_pixmap_now)
        self._scale_apply_delay_ms = 30
        # 자동 업그레이드 지연(ms) 및 일시정지 플래그
        self._fullres_upgrade_delay_ms = 120
        self._pause_auto_upgrade = False
        # 프리뷰 헤드룸 배율(1.0~1.2)
        self._preview_headroom = 1.0
        # 원본 풀해상도 이미지 보관(저장/고배율 표시용)
        self._fullres_image = None
        # 현재 픽스맵이 스케일 프리뷰인지 여부(원본 업그레이드 필요 판단용)
        self._is_scaled_preview = False
        # 직전 프레임이 원본으로 업그레이드 된 직후인지(스케일 적용 전 깜빡임 방지)
        self._just_upgraded_fullres = False
        # 프리로드 설정(다음/이전 1장씩)
        self._preload_radius = 1

        # 자연어 검색/임베딩 관련 기능 제거됨

        build_top_and_status_bars(self)

        # 회전 버튼은 위에서 생성됨

        self.update_status_left()
        self.update_status_right()

        # 단축키: 별점 0..5, 플래그 Z/X/C
        try:
            rating_bar.register_shortcuts(self)
        except Exception:
            pass

        # 정보 패널/지도 관련 초기화
        info_panel.setup_info_panel(self)

        # 초기 레이아웃 사이즈 조정
        try:
            self._update_info_panel_sizes()
        except Exception:
            pass

        # 키보드 단축키 설정
        self.setup_shortcuts()
        
        # 이미지 라벨에 초기 포커스 설정
        self.image_display_area.setFocus()
        
        self.update_button_states()

        # 전역 DnD 지원: 주요 위젯에 일괄 적용
        self._setup_global_dnd()

        # 전체화면 진입 전 스타일 복원용 변수
        self._stylesheet_before_fullscreen = None

        # 토스트 비활성화: 상태바/다이얼로그만 사용
        self.toast = None

        # 정보 오버레이 라벨 및 UI/커서 자동 숨김 타이머 초기화
        try:
            self._info_overlay = QLabel(self.image_display_area.viewport())
            self._info_overlay.setVisible(False)
            self._info_overlay.setStyleSheet("background-color: rgba(0,0,0,120); color: #EAEAEA; padding: 6px 8px; border-radius: 4px; font-size: 12px;")
            self._info_overlay.move(10, 10)
        except Exception:
            self._info_overlay = None
        self._overlay_visible = False
        self._ui_chrome_visible = True
        self._ui_auto_hide_timer = QTimer(self)
        self._ui_auto_hide_timer.setSingleShot(True)
        self._ui_auto_hide_timer.timeout.connect(self._on_ui_auto_hide_timer)
        self._cursor_hide_timer = QTimer(self)
        self._cursor_hide_timer.setSingleShot(True)
        self._cursor_hide_timer.timeout.connect(self._hide_cursor_if_fullscreen)
        # 전체화면 오버레이 위치 고정 타이머(이동/줌 중에도 붙게 유지)
        self._overlay_pos_timer = QTimer(self)
        self._overlay_pos_timer.setInterval(33)
        self._overlay_pos_timer.timeout.connect(self._position_fullscreen_overlays)

        # 프리로드 유휴 타이머
        self._idle_prefetch_timer = QTimer(self)
        self._idle_prefetch_timer.setSingleShot(True)
        self._idle_prefetch_timer.setInterval(600)  # 기본 600ms 무입력 시 유휴로 간주
        try:
            self._idle_prefetch_timer.timeout.connect(lambda: (not getattr(self, "_preload_only_when_idle", False)) or self._preload_neighbors())
        except Exception:
            pass

        # 설정 로드 및 최근/세션 복원
        self.load_settings()
        # 오버레이 관련 기능 비활성화(설정값이 있더라도 강제 비활성)
        try:
            self._anim_overlay_enabled = False
            self._fs_show_filmstrip_overlay = False
            self._overlay_enabled_default = False
            self._overlay_visible = False
            if getattr(self, "_info_overlay", None) is not None:
                self._info_overlay.setVisible(False)
        except Exception:
            pass
        # YAML에서 고급 캐시 설정이 제공된 경우 이미지 서비스에 적용
        try:
            img_max = getattr(self, "_img_cache_max_bytes", None)
            scaled_max = getattr(self, "_scaled_cache_max_bytes", None)
            if hasattr(self, "image_service") and self.image_service is not None:
                try:
                    self.image_service.set_cache_limits(img_max, scaled_max)
                except Exception:
                    pass
                # 캐시 자동 축소/정리 타이머 적용
                try:
                    if getattr(self, "_cache_gc_interval_s", 0) and int(self._cache_gc_interval_s) > 0:
                        if not hasattr(self, "_cache_gc_timer") or self._cache_gc_timer is None:
                            self._cache_gc_timer = QTimer(self)
                            self._cache_gc_timer.setSingleShot(False)
                            self._cache_gc_timer.timeout.connect(self._on_cache_gc_timer)
                        self._cache_gc_timer.start(max(5, int(self._cache_gc_interval_s)) * 1000)
                except Exception:
                    pass
        except Exception:
            pass
        self.rebuild_recent_menu()
        if not getattr(self, "_skip_session_restore", False):
            self.restore_last_session()
        # 남은 임시/백업 파일 정리 (최근 폴더 기준, 실패 무시)
        try:
            if self.last_open_dir and os.path.isdir(self.last_open_dir):
                cleanup_leftover_temp_and_backup(self.last_open_dir)
        except Exception:
            pass
        # UI 환경 설정 적용
        try:
            self._apply_ui_theme_and_spacing()
            self._preferred_view_mode = getattr(self, "_default_view_mode", 'fit')
        except Exception:
            self._preferred_view_mode = 'fit'
        # 배타 포커스를 위한 전역 단축키 관리
        self._global_shortcuts_enabled = True
        # 내비게이션 컨트롤러
        try:
            self._nav = NavigationController(self)
        except Exception:
            self._nav = None

        # 색상 A/B 비교용 상태
        self._color_ab_show_original = False

        # AI 분석 기본값은 load_settings에서 config.yaml/QSettings로 로드하므로
        # 여기서 초기화하지 않습니다. (초기 실행 시 YAML 값이 덮어써지는 문제 방지)
        self._chain_ai_active = False

    def clamp(self, value, min_v, max_v):
        return util_clamp(value, min_v, max_v)

    def _enable_dnd_on(self, widget):
        util_enable_dnd_on(widget, self)

    def _setup_global_dnd(self):
        util_setup_global_dnd(self)

    def _set_global_shortcuts_enabled(self, enabled: bool) -> None:
        set_global_shortcuts_enabled(self, enabled)

    def _on_cache_gc_timer(self) -> None:
        try:
            # 저메모리 감지(Windows 메모리 여유 등은 복잡하므로 간단히 프로세스 RSS 기준 임계 적용 가능)
            auto_pct = int(getattr(self, "_cache_auto_shrink_pct", 50))
            auto_pct = max(10, min(90, auto_pct))
            # 상한을 auto_pct%만큼 축소하여 일시적으로 비운 뒤 상한 복원
            if hasattr(self.image_service, "_img_cache") and hasattr(self.image_service, "_scaled_cache"):
                try:
                    orig_img_max = int(self.image_service._img_cache._max_bytes)
                    orig_scaled_max = int(self.image_service._scaled_cache._max_bytes)
                    new_img_max = max(1, int(orig_img_max * (100 - auto_pct) / 100))
                    new_scaled_max = max(1, int(orig_scaled_max * (100 - auto_pct) / 100))
                    self.image_service._img_cache.shrink_to(new_img_max)
                    self.image_service._scaled_cache.shrink_to(new_scaled_max)
                    # 복원
                    self.image_service._img_cache._max_bytes = orig_img_max
                    self.image_service._scaled_cache._max_bytes = orig_scaled_max
                except Exception:
                    pass
        except Exception:
            pass

    # ----- Undo/Redo 히스토리 -----
    def _capture_state(self):
        return hist.capture_state(self)

    def _restore_state(self, state) -> None:
        hist.restore_state(self, state)

    def _history_push(self) -> None:
        hist.history_push(self)

    # 전체 Viewer 상태 스냅샷/복원(옵셔널 사용 지점에서 활용)
    def snapshot_viewer_state(self) -> ViewerState:
        return ViewerState.snapshot_from(self)

    def restore_viewer_state(self, state: ViewerState) -> None:
        try:
            if not isinstance(state, ViewerState):
                return
            state.restore_into(self)
            # 경로/인덱스에 맞춰 이미지/뷰 갱신
            if 0 <= self.current_image_index < len(self.image_files_in_dir):
                self.load_image_at_current_index()
            self._apply_transform_to_view()
            self.update_button_states()
            self.update_status_right()
        except Exception:
            pass

    # 실행 취소/다시 실행 기능 제거됨

    # Settings: 저장/로드
    def load_settings(self):
        load_settings_ext(self)

    def save_settings(self):
        save_settings_ext(self)

    # 최근 항목 유틸은 mru_store로 분리됨

    def rebuild_recent_menu(self):
        build_recent_menu(self)
        try:
            build_log_menu(self)
        except Exception:
            pass

    def _open_recent_folder(self, dir_path: str):
        if not dir_path or not os.path.isdir(dir_path):
            self.statusBar().showMessage("폴더가 존재하지 않습니다.", 3000)
            # 존재하지 않는 항목은 목록에서 제거
            self.recent_folders = [it for it in self.recent_folders if normalize_path(it.get("path","")) != normalize_path(dir_path)]
            self.save_settings()
            self.rebuild_recent_menu()
            return
        self.scan_directory(dir_path)
        if 0 <= self.current_image_index < len(self.image_files_in_dir):
            self.load_image(self.image_files_in_dir[self.current_image_index])
        else:
            try:
                self.clear_display()
            except Exception:
                pass
            self.statusBar().showMessage("폴더에 표시할 이미지가 없습니다.", 3000)

    def clear_recent(self):
        self.recent_files = []
        self.recent_folders = []
        self.save_settings()
        self.rebuild_recent_menu()

    # 옵션: 존재하지 않는 항목을 한 번에 정리
    def prune_missing_from_recent(self):
        try:
            self.recent_files = [it for it in (self.recent_files or []) if os.path.isfile(it.get("path", "") if isinstance(it, dict) else str(it))]
        except Exception:
            pass
        try:
            self.recent_folders = [it for it in (self.recent_folders or []) if os.path.isdir(it.get("path", "") if isinstance(it, dict) else str(it))]
        except Exception:
            pass
        self.save_settings()
        self.rebuild_recent_menu()

    # 로그: 폴더 열기/ZIP 내보내기
    def _open_logs_folder(self):
        log_actions.open_logs_folder(self)

    def _export_logs_zip(self):
        log_actions.export_logs_zip(self)

    # Drag & Drop 지원: 유틸
    def _handle_dropped_files(self, files):
        dnd_handle_files(self, files)

    def _handle_dropped_folders(self, folders):
        dnd_handle_folders(self, folders)

    # Drag & Drop 이벤트 핸들러
    def dragEnterEvent(self, event):
        dnd_drag_enter(self, event)

    def dragMoveEvent(self, event):
        dnd_drag_move(self, event)

    def dropEvent(self, event):
        dnd_drop(self, event)

    # 상태 관련 계산 유틸은 status_utils로 분리됨

    def update_status_left(self):
        ts_update_status_left(self)

    def update_status_right(self):
        ts_update_status_right(self)

    def on_scale_changed(self, scale: float):
        self._last_scale = scale
        self.update_status_right()
        # 마지막 보기 모드 기록
        try:
            self._last_view_mode = getattr(self.image_display_area, "_view_mode", 'fit')
        except Exception:
            pass
        try:
            self._update_info_overlay_text()
        except Exception:
            pass
        # 디바운스 후 스케일별 다운샘플 적용
        try:
            if getattr(self, "_in_dpr_transition", False):
                return
            self._scale_apply_timer.start(getattr(self, "_scale_apply_delay_ms", 30))
        except Exception:
            pass

    def on_cursor_pos_changed(self, x: int, y: int):
        self._last_cursor_x = x
        self._last_cursor_y = y
        self.update_status_right()
        try:
            self._on_user_activity()
        except Exception:
            pass

    # ----- 애니메이션 컨트롤 -----
    def _is_current_file_animation(self) -> bool:
        return anim.is_current_file_animation(self)

    def anim_prev_frame(self):
        anim.prev_frame(self)

    def anim_next_frame(self):
        anim.next_frame(self)

    def anim_jump_back_10(self):
        try:
            from . import animation_controller as _anim
            _anim.jump_frames(self, -10)
        except Exception:
            pass

    def anim_jump_forward_10(self):
        try:
            from . import animation_controller as _anim
            _anim.jump_frames(self, +10)
        except Exception:
            pass

    def anim_toggle_play(self):
        anim.toggle_play(self)

    def _on_anim_tick(self):
        anim.on_tick(self)

    def _on_movie_frame(self, frame_index: int):
        anim.on_movie_frame(self, frame_index)

    # 세션 저장/복원
    def save_last_session(self):
        save_session_ext(self)

    def restore_last_session(self):
        restore_session_ext(self)

    def reset_zoom(self, fit=True):
        self.fit_mode = bool(fit)
        if self.fit_mode:
            self.image_display_area.fit_to_window()
        else:
            self.scale_factor = 1.0
        self.update_button_states()

    def fit_to_window(self):
        viewer_cmd.fit_to_window(self)

    def fit_to_width(self):
        viewer_cmd.fit_to_width(self)

    def fit_to_height(self):
        viewer_cmd.fit_to_height(self)

    def zoom_in(self):
        viewer_cmd.zoom_in(self)

    def zoom_out(self):
        viewer_cmd.zoom_out(self)

    def on_wheel_zoom(self, delta_y, ctrl, vp_anchor):
        # ImageView가 휠 이벤트를 처리하므로 이 메서드는 사용하지 않습니다.
        pass

    def update_window_title(self, file_path=None):
        """창 제목 업데이트"""
        ts_update_window_title(self, file_path)

    # ----- 변환 상태 관리 -----
    def _apply_transform_to_view(self):
        return transform_ui.apply_transform_to_view(self)

    def _mark_dirty(self, dirty: bool = True):
        self._is_dirty = bool(dirty)
        self.update_window_title(self.current_image_path)
        self.update_status_right()

    def get_transform_status_text(self) -> str:
        return transform_ui.get_transform_status_text(self)

    def rotate_cw_90(self):
        viewer_cmd.rotate_cw_90(self)

    def rotate_ccw_90(self):
        viewer_cmd.rotate_ccw_90(self)

    def rotate_180(self):
        viewer_cmd.rotate_180(self)

    def flip_horizontal(self):
        viewer_cmd.flip_horizontal(self)

    def flip_vertical(self):
        viewer_cmd.flip_vertical(self)

    def rotate_cycle(self):
        viewer_cmd.rotate_cycle(self)

    def reset_transform(self):
        viewer_cmd.reset_transform(self)

    def reset_transform_state(self):
        # 제거된 기능 (더 이상 사용하지 않음)
        pass

    # ImageService 콜백/적용
    def _on_image_loaded(self, path: str, img: QImage, success: bool, error: str):
        # 현재는 동기 로딩만 사용하므로 호출되지 않음
        if not success:
            return
        self._apply_loaded_image(path, img, source='async')

    def _apply_loaded_image(self, path: str, img: QImage, source: str):
        img_loader.apply_loaded_image(self, path, img, source)
        try:
            rating_bar.refresh(self)
        except Exception:
            pass
        # 플래그/별점 상태가 첫 프레임에서 누락되지 않게 재시도 예약
        try:
            from PyQt6.QtCore import QTimer  # type: ignore[import]
            QTimer.singleShot(0, lambda: rating_bar.refresh(self))
        except Exception:
            pass

    def _on_screen_changed(self, screen):
        ds.on_screen_changed(self, screen)

    def _on_dpi_changed(self, *args):
        ds.on_dpi_changed(self, *args)
        try:
            self._update_info_panel_sizes()
        except Exception:
            pass

    def _begin_dpr_transition(self, guard_ms: int = 160):
        ds.begin_dpr_transition(self, guard_ms)

    def _ensure_screen_signal_connected(self):
        ds.ensure_screen_signal_connected(self)
        try:
            self._update_info_panel_sizes()
        except Exception:
            pass

    def showEvent(self, event):
        super().showEvent(event)
        lifecycle.on_show(self, event)
        try:
            self._update_info_panel_sizes()
        except Exception:
            pass
        try:
            # 좌우 합계 100% 강제: 스플리터 폭 기준으로 남는 픽셀까지 분배
            if getattr(self, "_content_splitter", None):
                total = max(2, int(self._content_splitter.width()))
                left = int(total * 0.58)
                right = total - left
                # 최소 폭 보장 후 초과분을 반영
                min_left, min_right = 200, 200
                if left < min_left:
                    right -= (min_left - left)
                    left = min_left
                if right < min_right:
                    left -= (min_right - right)
                    right = min_right
                left = max(1, left)
                right = max(1, total - left)
                self._content_splitter.setSizes([left, right])
        except Exception:
            pass

    def event(self, e):
        try:
            from PyQt6.QtCore import QEvent  # type: ignore
            et = e.type()
            # 입력 이벤트가 발생하면 유휴 타이머를 재시작하여 일정 시간 후 프리로드
            if et in (QEvent.Type.MouseMove, QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress, QEvent.Type.Wheel):
                try:
                    if getattr(self, "_preload_only_when_idle", False):
                        self._idle_prefetch_timer.start()
                except Exception:
                    pass
            # 비활성화 시 자동 일시정지(옵션)
            if et in (QEvent.Type.WindowDeactivate, QEvent.Type.FocusOut, QEvent.Type.ApplicationStateChange):
                try:
                    if bool(getattr(self, "_anim_pause_on_unfocus", False)):
                        mv = getattr(self, "_movie", None)
                        if mv is not None:
                            try:
                                from PyQt6.QtGui import QMovie  # type: ignore
                                if mv.state() == QMovie.MovieState.Running:
                                    mv.setPaused(True)
                                    self._anim_is_playing = False
                            except Exception:
                                pass
                        else:
                            if bool(getattr(self, "_anim_is_playing", False)):
                                try:
                                    self._anim_timer.stop()
                                except Exception:
                                    pass
                                self._anim_is_playing = False
                except Exception:
                    pass
        except Exception:
            pass
        try:
            return super().event(e)
        except Exception:
            return False

    def keyPressEvent(self, event):
        try:
            from . import event_handlers as evt
            if evt.handle_key_press(self, event):
                event.accept()
                return
        except Exception:
            pass
        super().keyPressEvent(event)

    def _maybe_gesture_nav(self, wheel_event) -> bool:
        from .gesture_nav import maybe_gesture_nav
        return maybe_gesture_nav(self, wheel_event)

    def _preload_neighbors(self):
        img_loader.preload_neighbors(self)

    def setup_shortcuts(self):
        """키보드 단축키 설정"""
        apply_shortcuts_ext(self)
        # 숫자/문자 키 우선 처리 외에 보기/줌 단축키가 정상 동작하도록 포커스 설정
        try:
            self.image_display_area.setFocus()
        except Exception:
            pass

    # ----- 색상 보기 A/B 토글 -----
    def toggle_color_ab(self):
        try:
            self._color_ab_show_original = not bool(getattr(self, "_color_ab_show_original", False))
        except Exception:
            self._color_ab_show_original = False
        # 이미지 서비스에 정책 반영
        try:
            if hasattr(self, "image_service") and self.image_service is not None:
                self.image_service._color_view_mode = 'original' if self._color_ab_show_original else 'managed'
        except Exception:
            pass
        # 현재 이미지 재적용(캐시를 우회하도록 무효화 후 재로딩)
        try:
            cur = getattr(self, "current_image_path", None)
            if cur and os.path.isfile(cur):
                try:
                    self.image_service.invalidate_path(cur)
                except Exception:
                    pass
                self.load_image(cur, source='ab-toggle')
                try:
                    self.statusBar().showMessage("원본 색상 보기" if self._color_ab_show_original else "관리된 색상 보기", 1500)
                except Exception:
                    pass
        except Exception:
            pass

    # ----- 최근 파일 빠른 실행 핸들러 (Alt+1..9) -----
    def _open_recent_by_index(self, idx: int) -> None:
        try:
            items = getattr(self, "recent_files", []) or []
            if not items:
                return
            if 0 <= idx < len(items):
                it = items[idx]
                path = it.get("path") if isinstance(it, dict) else str(it)
                if path:
                    self.load_image(path, source='recent')
        except Exception:
            pass

    def _open_recent_1(self): self._open_recent_by_index(0)
    def _open_recent_2(self): self._open_recent_by_index(1)
    def _open_recent_3(self): self._open_recent_by_index(2)
    def _open_recent_4(self): self._open_recent_by_index(3)
    def _open_recent_5(self): self._open_recent_by_index(4)
    def _open_recent_6(self): self._open_recent_by_index(5)
    def _open_recent_7(self): self._open_recent_by_index(6)
    def _open_recent_8(self): self._open_recent_by_index(7)
    def _open_recent_9(self): self._open_recent_by_index(8)

    # 마지막 닫은 이미지 다시 열기
    def reopen_last_closed_image(self) -> None:
        try:
            path = getattr(self, "_last_closed_image_path", "") or ""
            if path and os.path.isfile(path):
                self.load_image(path, source='reopen')
            else:
                try:
                    self.statusBar().showMessage("다시 열 수 있는 이미지가 없습니다.", 2000)
                except Exception:
                    pass
        except Exception:
            pass

    # 보기 공유 토글 제거

    # ----- 사용자 요청 단축키 핸들러 -----
    def reload_current_image(self):
        file_cmd.reload_current_image(self)

    def clear_caches(self) -> None:
        try:
            if hasattr(self, "image_service") and self.image_service is not None:
                try:
                    self.image_service.clear_all_caches()
                except Exception:
                    pass
            try:
                self._clear_filmstrip_cache()
            except Exception:
                pass
            try:
                self.statusBar().showMessage("캐시를 비웠습니다.", 2000)
            except Exception:
                pass
        except Exception:
            pass

    # open_recent_list 단축키 제거됨

    def _apply_ui_theme_and_spacing(self):
        apply_theme(self)
        try:
            self._update_info_panel_sizes()
        except Exception:
            pass
        # filmstrip 테마 위임만 유지
        try:
            is_light = (getattr(self, "_resolved_theme", "dark") == "light")
            if hasattr(self, 'filmstrip') and self.filmstrip is not None:
                try:
                    self.filmstrip.apply_theme(is_light)
                except Exception:
                    pass
        except Exception:
            pass

    def _refresh_rating_flag_bar(self):
        return rating_bar.refresh(self)

    def _on_set_rating(self, n: int):
        return rating_bar.set_rating(self, n)
    
    # ----- 새 단축키 핸들러 -----
    def upgrade_fullres_now(self) -> None:
        try:
            if self._fullres_upgrade_timer.isActive():
                self._fullres_upgrade_timer.stop()
        except Exception:
            pass
        try:
            prev_pause = bool(getattr(self, "_pause_auto_upgrade", False))
            self._pause_auto_upgrade = False
            ds.upgrade_to_fullres_if_needed(self)
            self._pause_auto_upgrade = prev_pause
        except Exception:
            pass

    # (Shift+U 제거) 자동 업그레이드 토글 핸들러 제거

    def revert_to_preview(self) -> None:
        try:
            path = self.current_image_path or ""
            if not path:
                return
            if self._is_current_file_animation() or getattr(self, "_movie", None):
                return
            view = self.image_display_area
            # 앵커 보존: 뷰포트 중심의 아이템 좌표 저장
            item_anchor_point = None
            try:
                pix_item = getattr(view, "_pix_item", None)
                if pix_item:
                    vp_center = view.viewport().rect().center()
                    scene_center = view.mapToScene(vp_center)
                    item_anchor_point = pix_item.mapFromScene(scene_center)
            except Exception:
                item_anchor_point = None
            # 업그레이드 단계에서도 동일 앵커를 사용하도록 저장
            try:
                setattr(self, "_pending_anchor_point", item_anchor_point)
            except Exception:
                pass
            vm = str(getattr(view, "_view_mode", "fit") or "fit")
            from PyQt6.QtGui import QPixmap  # type: ignore[import]
            try:
                dpr = float(view.viewport().devicePixelRatioF())
            except Exception:
                try:
                    dpr = float(self.devicePixelRatioF())
                except Exception:
                    dpr = 1.0
            vw = max(1, int(view.viewport().width()))
            vh = max(1, int(view.viewport().height()))
            headroom = float(getattr(self, "_preview_headroom", 1.0) or 1.0)
            img = self.image_service.get_scaled_for_viewport(path, vw, vh, view_mode=vm, dpr=dpr, headroom=headroom)
            if img is None or img.isNull():
                return
            pm = QPixmap.fromImage(img)
            view.updatePixmapFrame(pm)
            try:
                ow = int(getattr(self, "_fullres_image", None).width()) if getattr(self, "_fullres_image", None) is not None else pm.width()
                sw = max(1, int(pm.width()))
                src_scale = min(1.0, float(sw) / float(max(1, ow)))
            except Exception:
                src_scale = 1.0
            view.set_source_scale(src_scale)
            # 보기 모드 재적용 및 앵커 재중앙
            try:
                if vm in ("fit", "fit_width", "fit_height"):
                    view.apply_current_view_mode()
                if item_anchor_point is not None and getattr(view, "_pix_item", None):
                    new_scene_point = view._pix_item.mapToScene(item_anchor_point)
                    view.centerOn(new_scene_point)
            except Exception:
                pass
            self._is_scaled_preview = True
            if not getattr(self, "_pause_auto_upgrade", False):
                try:
                    if self._fullres_upgrade_timer.isActive():
                        self._fullres_upgrade_timer.stop()
                    delay = int(getattr(self, "_fullres_upgrade_delay_ms", 120))
                    self._fullres_upgrade_timer.start(max(0, delay))
                except Exception:
                    pass
        except Exception:
            pass

    def _on_set_flag(self, f: str):
        return rating_bar.set_flag(self, f)

    def toggle_fullscreen(self):
        """전체화면 모드 토글"""
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def toggle_ui_chrome(self):
        """툴바/상태바/필름스트립/평점바 표시 전환"""
        new_visible = not bool(getattr(self, "_ui_chrome_visible", True))
        from .fs_overlays import apply_ui_chrome_visibility
        apply_ui_chrome_visibility(self, new_visible, temporary=False)
        if self.is_fullscreen and new_visible and int(getattr(self, "_fs_auto_hide_ms", 0)) > 0:
            try:
                self._ui_auto_hide_timer.start(int(self._fs_auto_hide_ms))
            except Exception:
                pass

    def toggle_info_overlay(self):
        # 오버레이 비활성화: 토글은 동작하지 않음
        try:
            self._overlay_visible = False
            if getattr(self, "_info_overlay", None) is not None:
                self._info_overlay.setVisible(False)
        except Exception:
            pass

    def mousePressEvent(self, event):
        try:
            if map_click.handle_mouse_press(self, event):
                event.accept()
                return
        except Exception:
            pass
        # 이미지 클릭으로 재생/일시정지(옵션)
        try:
            from PyQt6.QtCore import Qt as _Qt  # type: ignore
            if bool(getattr(self, "_anim_click_toggle", False)):
                if event.button() == _Qt.MouseButton.LeftButton and callable(getattr(self, "_is_current_file_animation", None)) and self._is_current_file_animation():
                    try:
                        self.anim_toggle_play()
                        event.accept()
                        return
                    except Exception:
                        pass
        except Exception:
            pass
        super().mousePressEvent(event)

    def enter_fullscreen(self):
        """전체화면 모드 진입 (애니메이션 없이)"""
        fs_enter_fullscreen(self)
        try:
            self._ensure_fs_overlays_created()
            try:
                self._fs_toolbar_h = int(self.button_bar.sizeHint().height()) if hasattr(self, 'button_bar') and self.button_bar else None
            except Exception:
                self._fs_toolbar_h = None
            try:
                self._fs_filmstrip_h = int(self.filmstrip.sizeHint().height()) if hasattr(self, 'filmstrip') and self.filmstrip else None
            except Exception:
                self._fs_filmstrip_h = None
            self._position_fullscreen_overlays()
            try:
                self._overlay_pos_timer.start()
            except Exception:
                pass
        except Exception:
            pass
        # 전체화면 진입 시 보기 모드 적용
        try:
            mode = str(getattr(self, "_fs_enter_view_mode", "keep"))
        except Exception:
            mode = "keep"
        try:
            if mode in ("fit", "fit_width", "fit_height", "actual"):
                if mode == "fit":
                    self.image_display_area.fit_to_window()
                elif mode == "fit_width":
                    self.image_display_area.fit_to_width()
                elif mode == "fit_height":
                    self.image_display_area.fit_to_height()
                else:
                    self.image_display_area.reset_to_100()
        except Exception:
            pass
        # 필름스트립/평점바 초기 상태
        try:
            if hasattr(self, 'filmstrip') and self.filmstrip is not None:
                self.filmstrip.setVisible(bool(getattr(self, "_fs_show_filmstrip_overlay", False)))
        except Exception:
            pass
        try:
            if hasattr(self, '_rating_flag_bar') and self._rating_flag_bar is not None:
                self._rating_flag_bar.setVisible(False)
        except Exception:
            pass
        # 진입 시 UI 크롬은 기본 숨김(자동 숨김이 없으면 유지)
        from .fs_overlays import apply_ui_chrome_visibility, start_auto_hide_timers
        apply_ui_chrome_visibility(self, False, temporary=True)
        start_auto_hide_timers(self)

    def exit_fullscreen(self):
        """전체화면 모드 종료 (제목표시줄 보장)"""
        fs_exit_fullscreen(self)
        try:
            self._overlay_pos_timer.stop()
        except Exception:
            pass
        # 진행 중인 오버레이 애니메이션 강제 중지
        try:
            if hasattr(self, "_anim_toolbar") and self._anim_toolbar:
                self._anim_toolbar.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "_anim_filmstrip") and self._anim_filmstrip:
                self._anim_filmstrip.stop()
        except Exception:
            pass
        try:
            if hasattr(self, 'filmstrip') and self.filmstrip is not None:
                self.filmstrip.setVisible(True)
        except Exception:
            pass
        try:
            if hasattr(self, '_rating_flag_bar') and self._rating_flag_bar is not None:
                self._rating_flag_bar.setVisible(True)
        except Exception:
            pass
        # UI 크롬/커서 복원 및 타이머 중지
        from .fs_overlays import apply_ui_chrome_visibility
        apply_ui_chrome_visibility(self, True, temporary=False)
        try:
            self._ui_auto_hide_timer.stop()
            self._cursor_hide_timer.stop()
            from .fs_overlays import restore_cursor, restore_overlays_to_layout
            restore_cursor(self)
            restore_overlays_to_layout(self)
            # 강제 재배치 및 보이기(간헐적 비가시성 회피)
            try:
                if hasattr(self, 'filmstrip') and self.filmstrip is not None:
                    self.filmstrip.setVisible(True)
                    self.filmstrip.updateGeometry()
                    self.filmstrip.repaint()
                    from PyQt6.QtCore import QTimer  # type: ignore[import]
                    QTimer.singleShot(0, lambda: (self.filmstrip.setVisible(True), self.filmstrip.raise_(), self.filmstrip.repaint()))
            except Exception:
                pass
        except Exception:
            pass

    def handle_escape(self):
        # 안전 종료 규칙: 전체화면에서 Esc를 누르면 우선 UI만 표시
        try:
            if bool(getattr(self, "_fs_safe_exit", True)) and self.is_fullscreen:
                if not bool(getattr(self, "_ui_chrome_visible", True)):
                    from .fs_overlays import apply_ui_chrome_visibility, start_auto_hide_timers
                    apply_ui_chrome_visibility(self, True, temporary=True)
                    start_auto_hide_timers(self)
                    return
        except Exception:
            pass
        util_handle_escape(self)

    def start_slideshow(self, interval_ms: int = 2000):
        """슬라이드쇼 시작(간단한 placeholder). interval_ms 간격으로 다음 이미지로 이동."""
        try:
            self.is_slideshow_active = True
            from PyQt6.QtCore import QTimer  # type: ignore
            if not hasattr(self, "_slideshow_timer") or self._slideshow_timer is None:
                self._slideshow_timer = QTimer(self)
                self._slideshow_timer.timeout.connect(lambda: getattr(self._nav, "show_next_image", lambda: None)())
            self._slideshow_timer.start(max(500, int(interval_ms)))
            # 슬라이드쇼 시작 예열 N장
            try:
                n = int(getattr(self, "_slideshow_prefetch_count", 0))
            except Exception:
                n = 0
            if n and n > 0:
                idx = int(getattr(self, "current_image_index", -1))
                files = getattr(self, "image_files_in_dir", []) or []
                paths: list[str] = []
                for off in range(1, n + 1):
                    j = idx + off
                    if 0 <= j < len(files):
                        paths.append(files[j])
                if paths:
                    prio = int(getattr(self, "_preload_priority", -1))
                    self.image_service.preload(paths, priority=prio)
        except Exception:
            pass

    def stop_slideshow(self):
        """슬라이드쇼 종료 (향후 구현을 위한 placeholder)"""
        self.is_slideshow_active = False
        # 슬라이드쇼 타이머가 있다면 여기서 정지
        try:
            if hasattr(self, "_slideshow_timer") and self._slideshow_timer is not None:
                self._slideshow_timer.stop()
        except Exception:
            pass

    def delete_current_image(self):
        file_cmd.delete_current_image(self)
        try:
            # 삭제 직전의 경로를 마지막 닫은 이미지 힌트로 보관
            if getattr(self, "current_image_path", None):
                self._last_closed_image_path = self.current_image_path
        except Exception:
            pass

    # 삭제 기능은 delete_utils로 분리됨
    def _undo_last_delete(self):
        # 휴지통 복원은 OS/라이브러리 의존성이 높아 표준화가 어려움.
        # 우선 휴지통을 열어 사용자가 즉시 복원할 수 있게 안내.
        try:
            import subprocess, sys
            if sys.platform == "win32":
                subprocess.Popen(["explorer.exe", "shell:RecycleBinFolder"])  # 휴지통 열기
        except Exception:
            pass
        # 잠시 후 디렉터리 재스캔 시도(복원 반영)
        try:
            QTimer.singleShot(1200, getattr(self, "_rescan_current_dir", lambda: None))
        except Exception:
            try:
                self._rescan_current_dir()
            except Exception:
                pass

    def clear_display(self):
        view_clear_display(self)

    def open_file(self):
        file_cmd.open_file(self)

    # ----- 파일/폴더 열기 관련 핸들러 -----
    def open_folder(self) -> None:
        file_cmd.open_folder(self)


    def load_image(self, file_path, source='other'):
        return img_loader.load_image(self, file_path, source)

    # ----- 저장 흐름 -----
    def save_current_image(self) -> bool:
        return file_cmd.save_current_image(self)

    def save_current_image_as(self) -> bool:
        return file_cmd.save_current_image_as(self)

    # ----- 스케일별 다운샘플 적용 -----
    def _apply_scaled_pixmap_now(self):
        ds.apply_scaled_pixmap_now(self)

    def _upgrade_to_fullres_if_needed(self):
        if getattr(self, "_pause_auto_upgrade", False):
            return
        ds.upgrade_to_fullres_if_needed(self)

    def _handle_dirty_before_action(self) -> bool:
        return dg_handle(self)

    # ----- 설정 다이얼로그 -----
    def open_settings_dialog(self):
        dlg.open_settings_dialog(self)

    def open_shortcuts_help(self):
        dlg.open_shortcuts_help(self)

    # EXIF 열람 기능 제거됨

    def open_ai_analysis_dialog(self):
        dlg.open_ai_analysis_dialog(self)

    def open_batch_ai_dialog(self):
        dlg.open_batch_ai_dialog(self)

    def open_natural_search_dialog(self):
        dlg.open_natural_search_dialog(self)

    def rerun_last_natural_search(self):
        try:
            q = str(getattr(self, "_last_natural_query", "") or "")
        except Exception:
            q = ""
        try:
            dlg.open_natural_search_dialog(self, initial_query=(q if q else None))
        except Exception:
            try:
                dlg.open_natural_search_dialog(self)
            except Exception:
                pass

    # ----- AI 단축키 핸들러 -----
    def toggle_ai_language(self):
        try:
            cur = str(getattr(self, "_ai_language", "ko") or "ko")
        except Exception:
            cur = "ko"
        try:
            new_lang = "en" if cur.startswith("ko") else "ko"
            self._ai_language = new_lang
            try:
                self.statusBar().showMessage(f"AI 언어: {'한국어' if new_lang=='ko' else '영어'}", 1500)
            except Exception:
                pass
            # 즉시 저장
            self.save_settings()
        except Exception:
            pass

    def start_chain_ai_analysis(self):
        """현재 사진부터 순차적으로 AI 분석 다이얼로그를 실행하고, 다음 사진으로 이동을 반복한다."""
        if not (self.image_files_in_dir and 0 <= self.current_image_index < len(self.image_files_in_dir)):
            try:
                self.statusBar().showMessage("분석할 사진이 없습니다.", 2000)
            except Exception:
                pass
            return
        # 재진입 방지
        if getattr(self, "_chain_ai_active", False):
            return
        self._chain_ai_active = True

        def _step():
            try:
                if not getattr(self, "_chain_ai_active", False):
                    return
                if not (self.image_files_in_dir and 0 <= self.current_image_index < len(self.image_files_in_dir)):
                    self._chain_ai_active = False
                    return
                # 현재 사진 분석(모달)
                self.open_ai_analysis_dialog()
                # 다음 사진으로 이동
                if not getattr(self, "_chain_ai_active", False):
                    return
                if self.current_image_index + 1 < len(self.image_files_in_dir):
                    self.show_next_image()
                    from PyQt6.QtCore import QTimer  # type: ignore[import]
                    QTimer.singleShot(0, _step)
                else:
                    self._chain_ai_active = False
                    try:
                        self.statusBar().showMessage("연쇄 분석 완료", 2000)
                    except Exception:
                        pass
            except Exception:
                self._chain_ai_active = False

        _step()

    def toggle_chain_ai_analysis(self):
        try:
            if getattr(self, "_chain_ai_active", False):
                self._chain_ai_active = False
                try:
                    self.statusBar().showMessage("연쇄 분석 중지", 1500)
                except Exception:
                    pass
                return
        except Exception:
            pass
        self.start_chain_ai_analysis()

    def open_similar_search_dialog(self):
        cur = self.current_image_path or ""
        if not cur or not os.path.isfile(cur):
            try:
                self.statusBar().showMessage("먼저 사진을 열어주세요.", 3000)
            except Exception:
                pass
            return
        folder = os.path.dirname(cur)
        try:
            from .similar_search_dialog import SimilarSearchDialog
            dlg_ = SimilarSearchDialog(self, cur, folder)
            # 다크 테마 적용을 위해 뷰어 테마러에 참조 전달
            try:
                self.similar_search_dialog = dlg_
                from .theme import apply_ui_theme_and_spacing as _apply_theme
                _apply_theme(self)
            except Exception:
                pass
            dlg_.exec()
        except Exception:
            pass
        finally:
            try:
                self.similar_search_dialog = None
            except Exception:
                pass

    def open_logs_folder(self) -> None:
        try:
            from ..utils.logging_setup import open_logs_folder as _open_logs
            ok, err = _open_logs()
            try:
                if ok:
                    self.statusBar().showMessage("로그 폴더를 열었습니다.", 3000)
                else:
                    self.statusBar().showMessage(f"로그 폴더 열기 실패: {err}", 4000)
            except Exception:
                pass
        except Exception:
            try:
                self.statusBar().showMessage("로그 폴더 열기 실패", 4000)
            except Exception:
                pass

    # ----- 정보 패널 -----
    def toggle_info_panel(self) -> None:
        return info_panel.toggle_info_panel(self)

    def _format_bytes(self, num_bytes: int) -> str:
        return info_panel.format_bytes(num_bytes)

    def _safe_frac_to_float(self, v):
        return info_panel._safe_frac_to_float(v)

    def update_info_panel(self) -> None:
        return info_panel.update_info_panel(self)

    def _dump_exif_all(self, path: str) -> str:
        from .exif_dump import dump_exif_all
        return dump_exif_all(path)

    # EXIF 탭 제거됨

    # ----- 지도 비동기 로딩 유틸 -----
    def _schedule_map_fetch(self, lat: float, lon: float, w: int, h: int, zoom: int):
        return info_panel.schedule_map_fetch(self, lat, lon, w, h, zoom)

    def _kick_map_fetch(self):
        return info_panel.kick_map_fetch(self)

    def _on_map_ready(self, token: int, pm):
        return info_panel.on_map_ready(self, token, pm)

    def _update_info_panel_sizes(self):
        return info_panel.update_info_panel_sizes(self)

    def toggle_privacy_hide_location(self) -> None:
        """주소/지도 표시(위치 정보) 프라이버시 토글.

        - _privacy_hide_location 값을 토글하고 즉시 저장/적용
        - 정보 패널이 열려 있으면 즉시 갱신하여 주소/지도 표시 상태 반영
        """
        try:
            cur = bool(getattr(self, "_privacy_hide_location", False))
        except Exception:
            cur = False
        new_val = not cur
        try:
            self._privacy_hide_location = bool(new_val)
        except Exception:
            pass
        # 저장 및 상태바 피드백
        try:
            self.save_settings()
        except Exception:
            pass
        try:
            msg = "위치 정보 숨김: 켬" if new_val else "위치 정보 숨김: 끔"
            self.statusBar().showMessage(msg, 1500)
        except Exception:
            pass
        # 정보 패널 즉시 갱신
        try:
            if getattr(self, "info_panel", None) is not None and self.info_panel.isVisible():
                self.update_info_panel()
        except Exception:
            pass

    def scan_directory(self, dir_path):
        res = dir_scan_ext.scan_directory(self, dir_path)
        # 디렉터리 진입 예열
        try:
            n = int(getattr(self, "_prefetch_on_dir_enter", 0))
        except Exception:
            n = 0
        if n and n > 0:
            try:
                idx = int(getattr(self, "current_image_index", -1))
                files = getattr(self, "image_files_in_dir", []) or []
                paths: list[str] = []
                for off in range(1, n + 1):
                    j = idx + off
                    if 0 <= j < len(files):
                        paths.append(files[j])
                if paths:
                    prio = int(getattr(self, "_preload_priority", -1))
                    self.image_service.preload(paths, priority=prio)
            except Exception:
                pass
        # 백그라운드 자연어 검색 색인(임베딩) — 폴더 진입 시 비동기 수행
        try:
            files = getattr(self, "image_files_in_dir", []) or []
            if files and bool(getattr(self, "_bg_index_on_dir_enter", True)):
                try:
                    maxn = int(getattr(self, "_bg_index_max", 200))
                except Exception:
                    maxn = 200
                try:
                    from threading import Thread
                except Exception:
                    Thread = None  # type: ignore
                def _bg_index():
                    try:
                        from ..services.online_search_service import OnlineEmbeddingIndex  # type: ignore
                        idx = OnlineEmbeddingIndex()
                        # API 키가 없으면 ensure_index는 건너뜀
                        idx.ensure_index(files[:max(1, int(maxn))], progress_cb=None)
                    except Exception:
                        pass
                if Thread is not None:
                    try:
                        t = Thread(target=_bg_index, daemon=True)
                        t.start()
                    except Exception:
                        pass
        except Exception:
            pass
        # 폴더 진입 시 AI 선분석(캐시 채우기, 다이얼로그 없이) — 옵션
        try:
            files = getattr(self, "image_files_in_dir", []) or []
            if files and bool(getattr(self, "_auto_ai_prefetch_on_dir", False)):
                try:
                    maxn = int(getattr(self, "_auto_ai_prefetch_count", 10))
                except Exception:
                    maxn = 10
                head = files[:max(1, int(maxn))]
                from threading import Thread
                def _prefetch_ai(paths: list[str]):
                    try:
                        from ..services.ai_analysis_service import AIAnalysisService, AnalysisContext  # type: ignore
                        svc = AIAnalysisService()
                        try:
                            cfg = self.open_ai_analysis_dialog.__self__._build_config_from_viewer()  # type: ignore[attr-defined]
                        except Exception:
                            cfg = None
                        if cfg is not None:
                            try:
                                svc.apply_config(cfg)
                            except Exception:
                                pass
                        ctx = AnalysisContext()
                        for p in paths:
                            try:
                                svc.analyze(p, context=ctx, progress_cb=None, is_cancelled=lambda: False)
                            except Exception:
                                pass
                    except Exception:
                        pass
                try:
                    Thread(target=_prefetch_ai, args=(head,), daemon=True).start()
                except Exception:
                    pass
        except Exception:
            pass
        return res

    def _rescan_current_dir(self):
        return dir_scan_ext.rescan_current_dir(self)

    def show_prev_image(self):
        if getattr(self, "_nav", None):
            self._nav.show_prev_image()
        else:
            nav_show_prev_image(self)

    def show_next_image(self):
        if getattr(self, "_nav", None):
            self._nav.show_next_image()
        else:
            nav_show_next_image(self)

    def _on_filmstrip_index_changed(self, row: int):
        try:
            if 0 <= row < len(self.image_files_in_dir):
                # 동일 인덱스면 재로딩 방지
                if int(self.current_image_index) == int(row):
                    return
                self.current_image_index = row
                self.load_image_at_current_index()
                try:
                    # 파일 선택으로 이미지가 로드되었으니 필름스트립/평점바 표시
                    if hasattr(self, 'filmstrip') and self.filmstrip is not None:
                        self.filmstrip.setVisible(True)
                    if hasattr(self, '_rating_flag_bar') and self._rating_flag_bar is not None:
                        self._rating_flag_bar.setVisible(True)
                except Exception:
                    pass
                try:
                    # 자동 스크롤(중앙 정렬) — 설정에 따라 수행
                    self.filmstrip.set_current_index(row)
                    try:
                        if bool(getattr(self, "_filmstrip_auto_center", True)):
                            from PyQt6.QtWidgets import QAbstractItemView  # type: ignore[import]
                            idx = self.filmstrip.model().index(row, 0)
                            self.filmstrip.scrollTo(idx, QAbstractItemView.ScrollHint.PositionAtCenter)
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    rating_bar.refresh(self)
                except Exception:
                    pass
        except Exception:
            pass

    def load_image_at_current_index(self):
        if getattr(self, "_nav", None):
            self._nav.load_image_at_current_index()
        else:
            nav_load_image_at_current_index(self)
        try:
            rating_bar.refresh(self)
        except Exception:
            pass
        try:
            if hasattr(self, 'filmstrip') and self.filmstrip is not None:
                self.filmstrip.setVisible(True)
            if hasattr(self, '_rating_flag_bar') and self._rating_flag_bar is not None:
                self._rating_flag_bar.setVisible(True)
        except Exception:
            pass

    def update_button_states(self):
        if getattr(self, "_nav", None):
            self._nav.update_button_states()
        else:
            nav_update_button_states(self)

    # ----- 내부 유틸: UI 크롬/오버레이/커서 -----
    def _apply_ui_chrome_visibility(self, visible: bool, temporary: bool = False) -> None:
        self._ui_chrome_visible = bool(visible)
        if not self.is_fullscreen:
            try:
                if hasattr(self, 'button_bar') and self.button_bar:
                    self.button_bar.setVisible(bool(visible))
            except Exception:
                pass
            try:
                self.statusBar().setVisible(bool(visible))
            except Exception:
                pass
            try:
                if hasattr(self, 'filmstrip') and self.filmstrip is not None:
                    self.filmstrip.setVisible(bool(visible))
            except Exception:
                pass
            try:
                if hasattr(self, '_rating_flag_bar') and self._rating_flag_bar is not None:
                    self._rating_flag_bar.setVisible(bool(visible))
            except Exception:
                pass
            return
        # 전체화면: 임시 숨김(temporary=True)으로 비활성화할 때는 애니메이션 없이 즉시 적용
        if temporary and not bool(visible):
            try:
                vp = self.image_display_area.viewport()
                vw, vh = vp.width(), vp.height()
            except Exception:
                vw = getattr(self, 'width', lambda: 0)()
                vh = getattr(self, 'height', lambda: 0)()
            # 툴바 즉시 숨김
            try:
                if hasattr(self, 'button_bar') and self.button_bar:
                    try:
                        h = int(getattr(self, '_fs_toolbar_h', None) or self.button_bar.sizeHint().height())
                    except Exception:
                        h = int(self.button_bar.height()) if self.button_bar.height() > 0 else 32
                    self.button_bar.setVisible(False)
                    self.button_bar.move(0, -int(h))
            except Exception:
                pass
            # 필름스트립 즉시 숨김
            try:
                if hasattr(self, 'filmstrip') and self.filmstrip:
                    try:
                        fh = int(max(1, int(getattr(self, '_fs_filmstrip_h', None) or self.filmstrip.sizeHint().height())))
                    except Exception:
                        fh = int(max(1, self.filmstrip.height())) if self.filmstrip.height() > 0 else 64
                    self.filmstrip.setVisible(False)
                    self.filmstrip.move(0, int(vh))
            except Exception:
                pass
        else:
            try:
                self._animate_fs_overlay(visible)
            except Exception:
                pass
        try:
            if hasattr(self, '_rating_flag_bar') and self._rating_flag_bar is not None:
                self._rating_flag_bar.setVisible(bool(visible) and (not self.is_fullscreen))
        except Exception:
            pass

    def _update_info_overlay_text(self) -> None:
        if getattr(self, "_info_overlay", None) is None:
            return
        try:
            path = self.current_image_path or ""
            name = os.path.basename(path) if path else "-"
        except Exception:
            name = "-"
        try:
            w = int(getattr(self.image_display_area, "_natural_width", 0) or 0)
            h = int(getattr(self.image_display_area, "_natural_height", 0) or 0)
        except Exception:
            w = h = 0
        try:
            scale_pct = int(round(float(getattr(self, "_last_scale", 1.0) or 1.0) * 100))
        except Exception:
            scale_pct = 100
        txt = f"{name}\n해상도: {w} x {h}\n배율: {scale_pct}%"
        try:
            self._info_overlay.setText(txt)
        except Exception:
            pass

    def _on_user_activity(self) -> None:
        from .fs_overlays import on_user_activity
        on_user_activity(self)

    def _start_auto_hide_timers(self) -> None:
        from .fs_overlays import start_auto_hide_timers
        start_auto_hide_timers(self)

    def _hide_cursor_if_fullscreen(self) -> None:
        from .fs_overlays import hide_cursor_if_fullscreen
        hide_cursor_if_fullscreen(self)

    def _on_ui_auto_hide_timer(self) -> None:
        try:
            if not getattr(self, "is_fullscreen", False):
                return
            try:
                from .fs_overlays import _is_mouse_over_ui_chrome  # type: ignore
            except Exception:
                _is_mouse_over_ui_chrome = None  # type: ignore
            if _is_mouse_over_ui_chrome is not None and _is_mouse_over_ui_chrome(self):
                # 마우스가 UI 크롬 위에 있으면 숨기지 않음
                return
        except Exception:
            pass
        self._apply_ui_chrome_visibility(False, temporary=True)

    def _restore_cursor(self) -> None:
        from .fs_overlays import restore_cursor
        restore_cursor(self)

    def _ensure_fs_overlays_created(self) -> None:
        from .fs_overlays import ensure_fs_overlays_created
        ensure_fs_overlays_created(self)

    def _position_fullscreen_overlays(self) -> None:
        from .fs_overlays import position_fullscreen_overlays
        position_fullscreen_overlays(self)

    def _animate_fs_overlay(self, show: bool) -> None:
        from .fs_overlays import animate_fs_overlay
        animate_fs_overlay(self, show)

    def _restore_overlays_to_layout(self) -> None:
        from .fs_overlays import restore_overlays_to_layout
        restore_overlays_to_layout(self)