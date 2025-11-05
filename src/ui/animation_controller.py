from __future__ import annotations

from typing import TYPE_CHECKING
from PyQt6.QtGui import QMovie, QPixmap  # type: ignore[import]
from PyQt6.QtGui import QColorSpace  # type: ignore[import]

if TYPE_CHECKING:
    from .main_window import JusawiViewer


def is_current_file_animation(viewer: "JusawiViewer") -> bool:
    try:
        if not viewer.current_image_path:
            return False
        is_anim, _ = viewer.image_service.probe_animation(viewer.current_image_path)
        return bool(is_anim)
    except Exception:
        return False


def prev_frame(viewer: "JusawiViewer") -> None:
    jump_frames(viewer, -1)


def next_frame(viewer: "JusawiViewer") -> None:
    jump_frames(viewer, +1)


def _compute_target_index(viewer: "JusawiViewer", delta: int) -> tuple[int, int]:
    cur = getattr(viewer.image_display_area, "_current_frame_index", 0)
    total = getattr(viewer.image_display_area, "_total_frames", -1)
    if isinstance(total, int) and total > 0:
        loop = bool(getattr(viewer, "_anim_loop", True))
        tgt = cur + int(delta)
        if loop:
            try:
                tgt = tgt % total
            except Exception:
                tgt = max(0, min(total - 1, tgt))
        else:
            tgt = max(0, min(total - 1, tgt))
    else:
        tgt = max(0, cur + int(delta))
    return int(tgt), int(total)


def jump_frames(viewer: "JusawiViewer", delta: int) -> None:
    if not is_current_file_animation(viewer):
        return
    try:
        new_index, total = _compute_target_index(viewer, delta)
        # QMovie 사용 중이면 우선 일시정지 후 점프 시도
        mv = getattr(viewer, "_movie", None)
        if mv is not None:
            try:
                mv.setPaused(True)
                viewer._anim_is_playing = False
            except Exception:
                pass
            jumped = False
            try:
                jumped = bool(mv.jumpToFrame(int(new_index)))
            except Exception:
                jumped = False
            if jumped:
                # frameChanged 시그널에서 화면 갱신이 이루어지므로 상태만 반영
                try:
                    viewer.image_display_area.set_animation_state(True, new_index, total)
                except Exception:
                    pass
                return
        # 폴백: 직접 프레임 로드
        img, ok, err = viewer.image_service.load_frame(viewer.current_image_path, new_index)
        if ok and img and not img.isNull():
            try:
                viewer.image_display_area.setPixmap(QPixmap.fromImage(img))
            except Exception:
                pass
            viewer.image_display_area.set_animation_state(True, new_index, total)
    except Exception:
        pass


def toggle_play(viewer: "JusawiViewer") -> None:
    if not is_current_file_animation(viewer):
        return
    try:
        try:
            st = (getattr(viewer._movie, 'state', lambda: None)() if getattr(viewer, '_movie', None) else None)
            viewer.log.debug(f"gif_toggle | movie_state={st} | playing={bool(getattr(viewer, '_anim_is_playing', False))}")
        except Exception:
            pass
        if viewer._movie:
            # 루프 설정이 바뀌었을 수 있으므로 매 토글 시 반영
            try:
                viewer._movie.setLoopCount(0 if bool(getattr(viewer, "_anim_loop", True)) else 1)
            except Exception:
                pass
            if viewer._movie.state() == QMovie.MovieState.Running:
                viewer._movie.setPaused(True)
                viewer._anim_is_playing = False
            elif viewer._movie.state() == QMovie.MovieState.Paused:
                viewer._movie.setPaused(False)
                viewer._anim_is_playing = True
            else:
                viewer._movie.start()
                viewer._anim_is_playing = True
        else:
            viewer._anim_is_playing = not viewer._anim_is_playing
            if viewer._anim_is_playing:
                viewer._anim_timer.start()
            else:
                viewer._anim_timer.stop()
        try:
            st2 = (getattr(viewer._movie, 'state', lambda: None)() if getattr(viewer, '_movie', None) else None)
            viewer.log.debug(f"gif_toggle_done | movie_state={st2} | playing={bool(getattr(viewer, '_anim_is_playing', False))}")
        except Exception:
            pass
    except Exception:
        pass


def on_tick(viewer: "JusawiViewer") -> None:
    if getattr(viewer, "_movie", None):
        return
    if not is_current_file_animation(viewer):
        viewer._anim_timer.stop()
        viewer._anim_is_playing = False
        return
    try:
        cur = getattr(viewer.image_display_area, "_current_frame_index", 0)
        total = getattr(viewer.image_display_area, "_total_frames", -1)
        loop = bool(getattr(viewer, "_anim_loop", True))
        if isinstance(total, int) and total > 1:
            next_index = (cur + 1) % total if loop else min(total - 1, cur + 1)
        else:
            next_index = cur + 1
        img, ok, err = viewer.image_service.load_frame(viewer.current_image_path, next_index)
        if ok and img and not img.isNull():
            viewer.image_display_area.updatePixmapFrame(QPixmap.fromImage(img))
            if not (isinstance(total, int) and total > 0):
                try:
                    is_anim, fc = viewer.image_service.probe_animation(viewer.current_image_path)
                    total = fc if (is_anim and isinstance(fc, int)) else -1
                except Exception:
                    total = -1
            viewer.image_display_area.set_animation_state(True, next_index, total)
    except Exception:
        pass


def on_movie_frame(viewer: "JusawiViewer", frame_index: int) -> None:
    try:
        if not viewer._movie:
            return
        pm = viewer._movie.currentPixmap()
        if pm and not pm.isNull():
            if getattr(viewer, "_convert_movie_frames_to_srgb", False):
                try:
                    img = pm.toImage()
                    cs = img.colorSpace()
                    srgb = QColorSpace(QColorSpace.NamedColorSpace.SRgb)
                    if cs.isValid() and cs != srgb:
                        img.convertToColorSpace(srgb)
                    pm = QPixmap.fromImage(img)
                except Exception:
                    pass
            viewer.image_display_area.updatePixmapFrame(pm)
            total = viewer._movie.frameCount()
            viewer.image_display_area.set_animation_state(True, frame_index, total)
            # 루프 해제 시 마지막 프레임에서 정지
            try:
                try:
                    if int(frame_index) == 0:
                        viewer.log.debug("gif_frame0")
                except Exception:
                    pass
                if not bool(getattr(viewer, "_anim_loop", True)):
                    # 우선 QMovie가 총 프레임을 알려줄 때
                    if isinstance(total, int) and total > 0:
                        if int(frame_index) >= int(total) - 1:
                            try:
                                viewer._movie.setPaused(True)
                            except Exception:
                                pass
                            viewer._anim_is_playing = False
                            try:
                                viewer.log.debug("gif_reached_last_frame_stop")
                            except Exception:
                                pass
                    else:
                        # 총 프레임이 미상인 경우: 래핑 감지(이전 인덱스보다 작아지면 루프)
                        prev = getattr(viewer.image_display_area, "_current_frame_index", 0)
                        if int(frame_index) < int(prev):
                            try:
                                viewer._movie.setPaused(True)
                            except Exception:
                                pass
                            viewer._anim_is_playing = False
                            try:
                                viewer.log.debug("gif_wrapped_stop")
                            except Exception:
                                pass
            except Exception:
                pass
    except Exception:
        pass


