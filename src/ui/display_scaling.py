from __future__ import annotations

from typing import TYPE_CHECKING
from PyQt6.QtGui import QPixmap  # type: ignore[import]

if TYPE_CHECKING:
    from .main_window import JusawiViewer


def begin_dpr_transition(viewer: "JusawiViewer", guard_ms: int = 160) -> None:
    try:
        viewer._in_dpr_transition = True
        if viewer._dpr_guard_timer.isActive():
            viewer._dpr_guard_timer.stop()
        viewer._dpr_guard_timer.start(int(max(60, guard_ms)))
    except Exception:
        viewer._in_dpr_transition = True


def ensure_screen_signal_connected(viewer: "JusawiViewer") -> None:
    if getattr(viewer, "_screen_signal_connected", False):
        return
    win = None
    try:
        win = viewer.windowHandle() if hasattr(viewer, 'windowHandle') else None
    except Exception:
        win = None
    if win:
        try:
            win.screenChanged.connect(lambda s: on_screen_changed(viewer, s))
            viewer._screen_signal_connected = True
        except Exception:
            viewer._screen_signal_connected = False


def on_screen_changed(viewer: "JusawiViewer", screen) -> None:
    try:
        if screen:
            try:
                screen.logicalDotsPerInchChanged.connect(lambda *a: on_dpi_changed(viewer, *a))
            except Exception:
                pass
    except Exception:
        pass
    begin_dpr_transition(viewer)
    try:
        apply_scaled_pixmap_now(viewer)
    except Exception:
        pass


def on_dpi_changed(viewer: "JusawiViewer", *args) -> None:
    begin_dpr_transition(viewer)
    try:
        apply_scaled_pixmap_now(viewer)
    except Exception:
        pass


def apply_scaled_pixmap_now(viewer: "JusawiViewer") -> None:
    if not viewer.load_successful or not viewer.current_image_path:
        return
    item_anchor_point = None
    try:
        view = viewer.image_display_area
        pix_item = getattr(view, "_pix_item", None)
        if pix_item:
            vp_center = view.viewport().rect().center()
            scene_center = view.mapToScene(vp_center)
            item_anchor_point = pix_item.mapFromScene(scene_center)
    except Exception:
        item_anchor_point = None
    try:
        if viewer._is_current_file_animation():
            return
        if getattr(viewer, "_movie", None):
            return
    except Exception:
        pass
    try:
        cur_scale = float(getattr(viewer, "_last_scale", 1.0) or 1.0)
    except Exception:
        cur_scale = 1.0
    try:
        dpr = float(viewer.image_display_area.viewport().devicePixelRatioF())
    except Exception:
        try:
            dpr = float(viewer.devicePixelRatioF())
        except Exception:
            dpr = 1.0
    prev_dpr = float(getattr(viewer, "_last_dpr", dpr) or dpr)
    dpr_changed = bool(abs(dpr - prev_dpr) > 1e-3)
    view_mode = str(getattr(viewer.image_display_area, "_view_mode", "free") or "free")

    if dpr_changed and view_mode in ("fit", "fit_width", "fit_height"):
        try:
            if getattr(viewer, "_fullres_image", None) is not None and not viewer._fullres_image.isNull():
                pm = QPixmap.fromImage(viewer._fullres_image)
                viewer.image_display_area.updatePixmapFrame(pm)
                viewer.image_display_area.set_source_scale(1.0)
                viewer.image_display_area.apply_current_view_mode()
                try:
                    if item_anchor_point is not None and getattr(viewer.image_display_area, "_pix_item", None):
                        new_scene_point = viewer.image_display_area._pix_item.mapToScene(item_anchor_point)
                        viewer.image_display_area.centerOn(new_scene_point)
                except Exception:
                    pass
                viewer._last_dpr = dpr
                return
        except Exception:
            pass
    if cur_scale >= 1.0:
        try:
            if getattr(viewer, "_fullres_image", None) is not None and not viewer._fullres_image.isNull():
                pm = QPixmap.fromImage(viewer._fullres_image)
                viewer.image_display_area.updatePixmapFrame(pm)
                viewer.image_display_area.set_source_scale(1.0)
        except Exception:
            pass
        if dpr_changed and view_mode in ("fit", "fit_width", "fit_height"):
            try:
                viewer.image_display_area.apply_current_view_mode()
            except Exception:
                pass
        if dpr_changed and view_mode in ("fit", "fit_width", "fit_height"):
            try:
                viewer.image_display_area.apply_current_view_mode()
            except Exception:
                pass
        try:
            if item_anchor_point is not None and getattr(viewer.image_display_area, "_pix_item", None):
                new_scene_point = viewer.image_display_area._pix_item.mapToScene(item_anchor_point)
                viewer.image_display_area.centerOn(new_scene_point)
        except Exception:
            pass
        viewer._last_dpr = dpr
        if getattr(viewer, "_in_dpr_transition", False):
            return
        return
    if getattr(viewer, "_disable_scaled_cache_below_100", False) and cur_scale < 1.0:
        try:
            if getattr(viewer, "_fullres_image", None) is not None and not viewer._fullres_image.isNull():
                pm = QPixmap.fromImage(viewer._fullres_image)
                viewer.image_display_area.updatePixmapFrame(pm)
                viewer.image_display_area.set_source_scale(1.0)
        except Exception:
            pass
        if dpr_changed and view_mode in ("fit", "fit_width", "fit_height"):
            try:
                viewer.image_display_area.apply_current_view_mode()
            except Exception:
                pass
        try:
            if item_anchor_point is not None and getattr(viewer.image_display_area, "_pix_item", None):
                new_scene_point = viewer.image_display_area._pix_item.mapToScene(item_anchor_point)
                viewer.image_display_area.centerOn(new_scene_point)
        except Exception:
            pass
        viewer._last_dpr = dpr
        return
    # 프리뷰/스케일 디코딩 경로 완전 비활성화: 항상 풀해상도 픽스맵만 사용
    try:
        if getattr(viewer, "_fullres_image", None) is not None and not viewer._fullres_image.isNull():
            pm_fb = QPixmap.fromImage(viewer._fullres_image)
            viewer.image_display_area.updatePixmapFrame(pm_fb)
            viewer.image_display_area.set_source_scale(1.0)
    except Exception:
        pass
    if dpr_changed and view_mode in ("fit", "fit_width", "fit_height"):
        try:
            viewer.image_display_area.apply_current_view_mode()
        except Exception:
            pass
    elif dpr_changed and getattr(viewer, "_preserve_visual_size_on_dpr_change", False):
        try:
            last_scale = float(getattr(viewer, "_last_scale", 1.0) or 1.0)
            desired_scale = last_scale * (prev_dpr / dpr)
            viewer.image_display_area.set_absolute_scale(desired_scale)
        except Exception:
            pass
    try:
        if item_anchor_point is not None and getattr(viewer.image_display_area, "_pix_item", None):
            new_scene_point = viewer.image_display_area._pix_item.mapToScene(item_anchor_point)
            viewer.image_display_area.centerOn(new_scene_point)
    except Exception:
        pass
    viewer._last_dpr = dpr
    if getattr(viewer, "_in_dpr_transition", False):
        return
    try:
        ss = float(getattr(viewer.image_display_area, "_source_scale", 1.0) or 1.0)
        need_upgrade = (ss < 1.0) or getattr(viewer, "_fullres_image", None) is None or getattr(viewer, "_is_scaled_preview", False)
        if need_upgrade and not bool(getattr(viewer, "_pause_auto_upgrade", False)):
            if viewer._fullres_upgrade_timer.isActive():
                viewer._fullres_upgrade_timer.stop()
            delay = int(getattr(viewer, "_fullres_upgrade_delay_ms", 300))
            viewer._fullres_upgrade_timer.start(max(0, delay))
            # 이벤트 루프 다음 틱에 업그레이드 시도(지연 반영)
            try:
                from PyQt6.QtCore import QTimer  # type: ignore[import]
                QTimer.singleShot(max(0, delay), getattr(viewer, "_upgrade_to_fullres_if_needed", lambda: None))
            except Exception:
                pass
    except Exception:
        pass


def upgrade_to_fullres_if_needed(viewer: "JusawiViewer") -> None:
    try:
        if not viewer.load_successful or not viewer.current_image_path:
            return
        if viewer._is_current_file_animation() or getattr(viewer, "_movie", None):
            return
        # fit 계열에서 너무 작은 유효 배율이면 풀해상도 업그레이드를 보류(스무딩 유지)
        try:
            vm = str(getattr(viewer.image_display_area, "_view_mode", "free") or "free")
        except Exception:
            vm = "free"
        try:
            cur_scale = float(getattr(viewer, "_last_scale", 1.0) or 1.0)
        except Exception:
            cur_scale = 1.0
        try:
            src_scale = float(getattr(viewer.image_display_area, "_source_scale", 1.0) or 1.0)
        except Exception:
            src_scale = 1.0
        try:
            min_scale = float(getattr(viewer, "_fullres_upgrade_min_scale", 0.5) or 0.5)
        except Exception:
            min_scale = 0.5
        # 유효 배율 = 보기 배율 * 소스(프리뷰) 배율
        effective_scale = cur_scale * src_scale
        if vm in ("fit", "fit_width", "fit_height") and effective_scale <= min_scale:
            return
        
        # 풀해상도가 없으면 지금 디코드하여 업그레이드 준비
        if getattr(viewer, "_fullres_image", None) is None or viewer._fullres_image.isNull():
            try:
                path, img, ok, _ = viewer.image_service.load(viewer.current_image_path)
                if ok and img is not None and not img.isNull():
                    viewer._fullres_image = img
                else:
                    return
            except Exception:
                return
        # 현재 픽스맵이 풀해상도보다 작으면 업그레이드 필요(소스 스케일 값에 의존하지 않음)
        cur_pix = None
        try:
            cur_pix = viewer.image_display_area.originalPixmap()
        except Exception:
            cur_pix = None
        if cur_pix and not cur_pix.isNull():
            try:
                if cur_pix.width() >= viewer._fullres_image.width() and cur_pix.height() >= viewer._fullres_image.height():
                    return
            except Exception:
                pass
        item_anchor_point = None
        try:
            # Ctrl+U 등 프리뷰 단계에서 저장한 앵커 우선 사용
            saved_anchor = getattr(viewer, "_pending_anchor_point", None)
            if saved_anchor is not None:
                item_anchor_point = saved_anchor
            else:
                view = viewer.image_display_area
                pix_item = getattr(view, "_pix_item", None)
                if pix_item:
                    vp_center = view.viewport().rect().center()
                    scene_center = view.mapToScene(vp_center)
                    item_anchor_point = pix_item.mapFromScene(scene_center)
        except Exception:
            item_anchor_point = None
        pm = QPixmap.fromImage(viewer._fullres_image)
        viewer.image_display_area.updatePixmapFrame(pm)
        # 좌표계를 풀해상도 기준으로 갱신
        try:
            viewer.image_display_area._natural_width = int(viewer._fullres_image.width())
            viewer.image_display_area._natural_height = int(viewer._fullres_image.height())
        except Exception:
            pass
        viewer.image_display_area.set_source_scale(1.0)
        # 맞춤 모드에서는 즉시 재맞춤 적용하여 가시 결과를 보장
        try:
            vm = str(getattr(viewer.image_display_area, "_view_mode", "free") or "free")
            if vm in ("fit", "fit_width", "fit_height"):
                viewer.image_display_area.apply_current_view_mode()
        except Exception:
            pass
        try:
            if item_anchor_point is not None and getattr(viewer.image_display_area, "_pix_item", None):
                new_scene_point = viewer.image_display_area._pix_item.mapToScene(item_anchor_point)
                viewer.image_display_area.centerOn(new_scene_point)
        except Exception:
            pass
        # 사용한 앵커는 일회성으로 소거
        try:
            if getattr(viewer, "_pending_anchor_point", None) is not None:
                viewer._pending_anchor_point = None
        except Exception:
            pass
        # 좌표/상태 갱신: 현재 커서 위치 기준으로 동기화
        try:
            from PyQt6.QtCore import QPointF  # type: ignore[import]
            from PyQt6.QtGui import QCursor  # type: ignore[import]
            vp_point = viewer.image_display_area.viewport().mapFromGlobal(QCursor.pos())
            viewer.image_display_area._emit_cursor_pos_at_viewport_point(QPointF(vp_point))
        except Exception:
            pass
        try:
            viewer.update_status_right()
        except Exception:
            pass
        try:
            viewer.update_status_left()
        except Exception:
            pass
        # 한 틱 동안 깜빡임 방지 플래그
        try:
            from PyQt6.QtCore import QTimer  # type: ignore[import]
            viewer._just_upgraded_fullres = True
            QTimer.singleShot(0, lambda: setattr(viewer, "_just_upgraded_fullres", False))
        except Exception:
            pass
        # 업그레이드 완료: 플래그 해제
        try:
            viewer._is_scaled_preview = False
        except Exception:
            pass
    except Exception:
        pass


