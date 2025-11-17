from __future__ import annotations

import os
from typing import TYPE_CHECKING
from PyQt6.QtGui import QPixmap, QMovie  # type: ignore[import]

from ..storage.mru_store import update_mru, process_mru

if TYPE_CHECKING:
    from .main_window import JusawiViewer


def apply_loaded_image(viewer: "JusawiViewer", path: str, img, source: str) -> None:
    # 확대 상태 유지 정책에 따라 변환/보기 초기화 수준 결정
    zoom_policy = str(getattr(viewer, "_zoom_policy", "mode"))
    # 요구사항: 변환 상태 유지하지 않음 → 항상 초기화(비파괴 표시만)
    try:
        viewer._tf_rotation = 0
        viewer._tf_flip_h = False
        viewer._tf_flip_v = False
    except Exception:
        pass
    pixmap = QPixmap.fromImage(img)
    viewer.image_display_area.setPixmap(pixmap)
    try:
        viewer._fullres_image = img
        try:
            viewer.image_display_area._natural_width = int(img.width())
            viewer.image_display_area._natural_height = int(img.height())
        except Exception:
            pass
    except Exception:
        viewer._fullres_image = None
    # 이전 파일의 애니메이션 여부 저장(탐색 시 상태 유지 정책 판단용)
    try:
        viewer._prev_was_animation = bool(getattr(viewer.image_display_area, "_is_animation", False))
    except Exception:
        viewer._prev_was_animation = False
    try:
        if getattr(viewer, "_movie", None):
            try:
                viewer._movie.stop()
            except Exception:
                pass
            viewer._movie.deleteLater()
    except Exception:
        pass
    viewer._movie = None
    try:
        is_anim, frame_count = viewer.image_service.probe_animation(path)
        try:
            viewer.log.debug(f"gif_probe | file={os.path.basename(path)} | is_anim={is_anim} | frames={frame_count}")
        except Exception:
            pass
        viewer.image_display_area.set_animation_state(is_anim, current_index=0, total_frames=frame_count)
        if is_anim:
            try:
                mv = QMovie(path)
                mv.setCacheMode(QMovie.CacheMode.CacheAll)
                try:
                    mv.jumpToFrame(0)
                except Exception:
                    pass
                mv.frameChanged.connect(viewer._on_movie_frame)
                viewer._movie = mv
                try:
                    is_valid = bool(mv.isValid())
                except Exception:
                    is_valid = True
                try:
                    viewer.log.debug(f"gif_qmovie_create | valid={is_valid} | frameCount={getattr(mv, 'frameCount', lambda: -1)()}")
                except Exception:
                    pass
                try:
                    viewer._anim_timer.stop()
                except Exception:
                    pass
                try:
                    # 루프 설정 반영: True(무한 루프=0), False(1회 재생=1)
                    try:
                        mv.setLoopCount(0 if bool(getattr(viewer, "_anim_loop", True)) else 1)
                    except Exception:
                        pass
                    # 자동 재생 설정 반영
                    # '이동(nav)'에서만 재생 상태 유지 적용, 열기/드롭은 자동재생 우선
                    keep = bool(getattr(viewer, "_anim_keep_state_on_switch", False)) and (source == "nav")
                    desired_play = bool(getattr(viewer, "_anim_autoplay", True))
                    if keep:
                        try:
                            prev_anim = bool(getattr(viewer, "_prev_was_animation", False))
                        except Exception:
                            prev_anim = False
                        if prev_anim:
                            try:
                                desired_play = bool(getattr(viewer, "_anim_is_playing", False))
                            except Exception:
                                desired_play = desired_play
                        else:
                            # 이전 파일이 JPG 등 정지 이미지였으면 자동재생 기준으로 시작
                            desired_play = bool(getattr(viewer, "_anim_autoplay", True))
                    if desired_play:
                        # QMovie 유효성 검사: 실패 시 타이머 기반 폴백 재생
                        try:
                            is_valid = bool(viewer._movie.isValid())
                        except Exception:
                            is_valid = True
                        if is_valid:
                            try:
                                pm0 = viewer._movie.currentPixmap()
                                if pm0 and not pm0.isNull():
                                    viewer.image_display_area.updatePixmapFrame(pm0)
                            except Exception:
                                pass
                            try:
                                viewer.log.debug(f"gif_qmovie_start | source={source}")
                            except Exception:
                                pass
                            viewer._movie.start()
                            viewer._anim_is_playing = True
                        else:
                            viewer._movie = None
                            try:
                                img0, ok0, _ = viewer.image_service.load_frame(path, 0)
                            except Exception:
                                img0, ok0 = None, False
                            if ok0 and img0 and not img0.isNull():
                                try:
                                    viewer.image_display_area.setPixmap(QPixmap.fromImage(img0))
                                except Exception:
                                    pass
                                try:
                                    viewer.image_display_area.set_animation_state(True, 0, frame_count)
                                except Exception:
                                    pass
                            try:
                                try:
                                    viewer.log.debug(f"gif_timer_fallback_start | source={source}")
                                except Exception:
                                    pass
                                viewer._anim_timer.start()
                            except Exception:
                                pass
                            viewer._anim_is_playing = True
                    else:
                        # 정지 상태 유지하되 첫 프레임은 표시
                        try:
                            pm0 = viewer._movie.currentPixmap()
                            if pm0 and not pm0.isNull():
                                viewer.image_display_area.updatePixmapFrame(pm0)
                        except Exception:
                            pass
                        try:
                            try:
                                viewer.log.debug("gif_qmovie_stop_initial")
                            except Exception:
                                pass
                            viewer._movie.stop()
                        except Exception:
                            pass
                        viewer._anim_is_playing = False
                except Exception:
                    viewer._anim_is_playing = False
            except Exception:
                viewer._movie = None
                viewer._anim_is_playing = False
        else:
            viewer._anim_is_playing = False
    except Exception:
        try:
            viewer.image_display_area.set_animation_state(False)
        except Exception:
            pass
    try:
        viewer.log.info("apply_loaded | file=%s | source=%s | anim=%s", os.path.basename(path), source, bool(is_anim))
    except Exception:
        pass
    viewer.load_successful = True
    viewer.current_image_path = path
    viewer.update_window_title(path)
    if os.path.exists(path):
        try:
            dirp = os.path.dirname(path)
            # 같은 폴더이고 목록에 이미 포함되어 있으면 재스캔 생략
            already_listed = False
            try:
                nc = os.path.normcase
                if viewer.image_files_in_dir:
                    listed_set = {nc(p) for p in viewer.image_files_in_dir}
                    already_listed = nc(path) in listed_set and nc(getattr(viewer, "_last_scanned_dir", "")) == nc(dirp)
            except Exception:
                already_listed = False
            # 열기/드롭으로 들어온 경우에는 설정에 따라 자동 스캔 여부 결정
            should_scan = True
            try:
                if source in ('open', 'drop') and not bool(getattr(viewer, "_open_scan_dir_after_open", True)):
                    should_scan = False
            except Exception:
                should_scan = True
            if should_scan and not already_listed:
                viewer.scan_directory(dirp)
            else:
                # 동일 폴더에서 이미 목록이 준비되어 있다면, 현재 경로의 인덱스로 동기화
                try:
                    nc = os.path.normcase
                    paths = getattr(viewer, "image_files_in_dir", []) or []
                    idx = next((i for i, p in enumerate(paths) if nc(p) == nc(path)), -1)
                    if idx >= 0:
                        viewer.current_image_index = idx
                except Exception:
                    pass
        except Exception:
            try:
                if should_scan:
                    viewer.scan_directory(os.path.dirname(path))
            except Exception:
                pass
    viewer.update_button_states()
    viewer.update_status_left()
    viewer.update_status_right()
    # 변환 상태는 정책에 따라 유지/초기화되었으므로, 현재 뷰에 반영
    viewer._apply_transform_to_view()
    # 확대/보기 모드 적용: 정책에 따라 유지
    try:
        if zoom_policy == 'reset':
            pref = getattr(viewer, "_session_preferred_view_mode", None)
            if pref == 'fit':
                viewer.image_display_area.fit_to_window()
            elif pref == 'fit_width':
                viewer.image_display_area.fit_to_width()
            elif pref == 'fit_height':
                viewer.image_display_area.fit_to_height()
            elif pref == 'actual':
                viewer.image_display_area.reset_to_100()
        elif zoom_policy == 'mode':
            # 보기 모드 유지: 직전 모드를 반영
            pref = getattr(viewer, "_last_view_mode", 'fit')
            if pref == 'fit':
                viewer.image_display_area.fit_to_window()
            elif pref == 'fit_width':
                viewer.image_display_area.fit_to_width()
            elif pref == 'fit_height':
                viewer.image_display_area.fit_to_height()
            elif pref == 'actual':
                viewer.image_display_area.reset_to_100()
        elif zoom_policy == 'scale':
            # 배율 유지: 직전 스케일을 그대로 적용
            try:
                prev_scale = float(getattr(viewer, "_last_scale", 1.0))
            except Exception:
                prev_scale = 1.0
            if prev_scale and prev_scale > 0:
                viewer.image_display_area.set_absolute_scale(prev_scale)
        else:
            pass
    except Exception:
        pass
    # 보기 공유 옵션 제거됨
    viewer._mark_dirty(False)
    # 별점/플래그 표시 갱신(최초 로드시에도 즉시 반영)
    try:
        from . import rating_bar
        rating_bar.refresh(viewer)
    except Exception:
        pass
    # 필름 스트립 선택 동기화(내비게이션 등으로 변경 시 표시 반영)
    try:
        if hasattr(viewer, "filmstrip") and viewer.filmstrip is not None:
            if 0 <= viewer.current_image_index < len(viewer.image_files_in_dir):
                viewer.filmstrip.set_current_index(viewer.current_image_index)
    except Exception:
        pass
    # 필름스트립 인덱스 동기화 이후 한 번 더 갱신해 플래그 표시를 확실히 반영
    try:
        from . import rating_bar
        rating_bar.refresh(viewer)
    except Exception:
        pass
    # 정보 패널 갱신(보일 때만)
    try:
        if hasattr(viewer, "update_info_panel") and getattr(viewer, "info_text", None) is not None:
            if viewer.info_text.isVisible():
                viewer.update_info_panel()
    except Exception:
        pass
    try:
        viewer._history_undo.clear()
        viewer._history_redo.clear()
    except Exception:
        pass
    try:
        # 프리로드 정책: 유휴 시에만 설정이 켜져 있으면 즉시 호출 대신 유휴 타이머로 지연
        if bool(getattr(viewer, "_preload_only_when_idle", False)):
            try:
                if hasattr(viewer, "_idle_prefetch_timer") and viewer._idle_prefetch_timer is not None:
                    viewer._idle_prefetch_timer.start()
            except Exception:
                pass
        else:
            preload_neighbors(viewer)
    except Exception:
        pass
    # 첫 이미지 로드 시 보조 UI 표시 (UI 크롬이 표시 상태일 때만)
    try:
        if not viewer.is_fullscreen and bool(getattr(viewer, "_ui_chrome_visible", True)):
            if hasattr(viewer, 'filmstrip') and viewer.filmstrip is not None:
                viewer.filmstrip.setVisible(True)
            if hasattr(viewer, '_rating_flag_bar') and viewer._rating_flag_bar is not None:
                viewer._rating_flag_bar.setVisible(True)
    except Exception:
        pass
    # 자동화: AI 분석 자동 실행 (세분화: 열기/드롭/이동, 캐시 건너뛰기)
    try:
        auto_open = bool(getattr(viewer, "_auto_ai_on_open", False)) and (source == "open")
        auto_drop = bool(getattr(viewer, "_auto_ai_on_drop", False)) and (source == "drop")
        auto_nav  = bool(getattr(viewer, "_auto_ai_on_nav", False)) and (source == "nav")
        if auto_open or auto_drop or auto_nav:
            delay = max(0, int(getattr(viewer, "_auto_ai_delay_ms", 0)))
            from PyQt6.QtCore import QTimer  # type: ignore[import]
            def _run_ai():
                try:
                    # 캐시 건너뛰기: 이미 캐시가 있으면 생략
                    if bool(getattr(viewer, "_ai_skip_if_cached", False)):
                        try:
                            from ..services.ai_analysis_service import AIAnalysisService, AnalysisContext
                            svc = AIAnalysisService()
                            ctx = AnalysisContext()
                            # 내부 캐시 키 경로 접근하여 존재 시 생략
                            key = svc._cache_key(viewer.current_image_path, ctx)
                            cpath = svc._cache_path(key)
                            if os.path.isfile(cpath):
                                return
                        except Exception:
                            pass
                    viewer.open_ai_analysis_dialog()
                except Exception:
                    pass
            if delay > 0:
                QTimer.singleShot(delay, _run_ai)
            else:
                QTimer.singleShot(0, _run_ai)
    except Exception:
        pass
    if source in ('open', 'drop', 'nav'):
        try:
            viewer.recent_files = update_mru(viewer.recent_files, path)
            # MRU 후처리: 최대 개수/제외/핀/자동 정리 적용
            try:
                max_items = int(getattr(viewer, "_recent_max_items", 10))
            except Exception:
                max_items = 10
            viewer.recent_files = process_mru(
                viewer.recent_files,
                max_items=max_items,
                exclude_patterns=str(getattr(viewer, "_recent_exclude_patterns", "")),
                auto_prune_missing=bool(getattr(viewer, "_recent_auto_prune_missing", True)),
                is_folder=False,
            )
            parent_dir = os.path.dirname(path)
            if parent_dir and os.path.isdir(parent_dir):
                try:
                    if bool(getattr(viewer, "_remember_last_open_dir", True)):
                        viewer.last_open_dir = parent_dir
                except Exception:
                    viewer.last_open_dir = parent_dir
            viewer.save_settings()
            viewer.rebuild_recent_menu()
        except Exception:
            pass
    try:
        # 즉시 스케일 적용 트리거 1회만 수행(중복 호출 제거)
        viewer._scale_apply_timer.start(0)
    except Exception:
        pass
    # 객체 탐지: 이미지 적용 후 트리거
    try:
        if hasattr(viewer, "_trigger_object_detection"):
            viewer._trigger_object_detection()
    except Exception:
        pass


def load_image(viewer: "JusawiViewer", file_path: str, source: str = 'other') -> bool:
    # 네트워크 파일 로드 비활성화: HTTP/HTTPS 및 Windows UNC 경로 차단
    try:
        p = str(file_path or "")
        if p.lower().startswith(("http://", "https://")):
            try:
                viewer.statusBar().showMessage("네트워크 URL은 지원하지 않습니다.", 3000)
            except Exception:
                pass
            return False
        import os
        if os.name == 'nt' and p.startswith('\\\\'):
            try:
                viewer.statusBar().showMessage("네트워크 공유 경로는 지원하지 않습니다.", 3000)
            except Exception:
                pass
            return False
    except Exception:
        pass
    if viewer._is_dirty and viewer.current_image_path and os.path.normcase(file_path) != os.path.normcase(viewer.current_image_path):
        if not viewer._handle_dirty_before_action():
            try:
                viewer.log.info("load_image_aborted_dirty | new=%s | cur=%s", os.path.basename(file_path), os.path.basename(viewer.current_image_path or ""))
            except Exception:
                pass
            return False
    try:
        viewer.log.info("load_image_start | src=%s | source=%s", os.path.basename(file_path), source)
    except Exception:
        pass

    # 이전 애니메이션(QMovie) 정리: 프레임 시그널이 남아 이전 파일이 계속 그려지는 문제 방지
    try:
        if getattr(viewer, "_movie", None):
            try:
                try:
                    viewer._movie.frameChanged.disconnect(viewer._on_movie_frame)
                except Exception:
                    pass
                viewer._movie.stop()
            except Exception:
                pass
            try:
                viewer._movie.deleteLater()
            except Exception:
                pass
            viewer._movie = None
        try:
            if getattr(viewer, "_anim_timer", None):
                viewer._anim_timer.stop()
        except Exception:
            pass
        viewer._anim_is_playing = False
        try:
            viewer.image_display_area.set_animation_state(False)
        except Exception:
            pass
    except Exception:
        pass

	# 프리뷰 비활성화: 항상 원본(자동 변환 적용)으로 로드

    # 폴백/애니메이션: 원본 동기 로드 후 적용
    path, img, success, _ = viewer.image_service.load(file_path)
    if success and img is not None:
        apply_loaded_image(viewer, path, img, source)
        try:
            viewer.log.info("load_image_ok | file=%s | w=%d | h=%d", os.path.basename(path), int(img.width()), int(img.height()))
        except Exception:
            pass
        return True
    try:
        viewer.log.error("load_image_fail | file=%s", os.path.basename(file_path))
    except Exception:
        pass
    # 실패 시 기존 목록/상태를 유지하여 다른 이미지 열람이 가능하도록 한다.
    viewer.load_successful = False
    try:
        viewer.statusBar().showMessage("이미지를 불러올 수 없습니다.", 3000)
    except Exception:
        pass
    viewer.update_button_states()
    viewer.update_status_left()
    viewer.update_status_right()
    return False


def preload_neighbors(viewer: "JusawiViewer") -> None:
    """현재 인덱스를 기준으로 다음/이전 이미지를 백그라운드로 프리로드.
    방향/반경/우선순위는 뷰어 설정을 따른다.
    """
    if not viewer.image_files_in_dir:
        return
    idx = viewer.current_image_index
    if not (0 <= idx < len(viewer.image_files_in_dir)):
        return
    paths: list[str] = []
    if not bool(getattr(viewer, "_enable_thumb_prefetch", True)):
        return
    direction = str(getattr(viewer, "_preload_direction", "both"))  # both|forward|backward
    radius = int(getattr(viewer, "_preload_radius", 2))
    for off in range(1, radius + 1):
        if direction in ("both", "forward"):
            n = idx + off
            if 0 <= n < len(viewer.image_files_in_dir):
                paths.append(viewer.image_files_in_dir[n])
        if direction in ("both", "backward"):
            p = idx - off
            if 0 <= p < len(viewer.image_files_in_dir):
                paths.append(viewer.image_files_in_dir[p])
    if paths:
        try:
            prio = int(getattr(viewer, "_preload_priority", -1))
            viewer.image_service.preload(paths, priority=prio)
        except Exception:
            pass


