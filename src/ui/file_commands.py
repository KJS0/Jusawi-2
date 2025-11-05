from __future__ import annotations

import os
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMessageBox  # type: ignore[import]

if TYPE_CHECKING:
    from .main_window import JusawiViewer


def delete_current_image(viewer: "JusawiViewer") -> None:
    if not viewer.current_image_path or not os.path.exists(viewer.current_image_path):
        return
    reply = QMessageBox.question(
        viewer,
        "파일 삭제",
        f"'{os.path.basename(viewer.current_image_path)}'을(를) 휴지통으로 이동하시겠습니까?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    if reply != QMessageBox.StandardButton.Yes:
        return
    try:
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
                viewer.image_display_area.updatePixmapFrame(None)
            except Exception:
                pass
            try:
                viewer.image_display_area.setPixmap(None)
            except Exception:
                pass
            try:
                viewer.image_service.invalidate_path(viewer.current_image_path)
            except Exception:
                pass
            try:
                import gc
                gc.collect()
            except Exception:
                pass
        except Exception:
            pass

        from ..utils.delete_utils import move_to_trash_windows
        move_to_trash_windows(viewer.current_image_path)

        original_index = viewer.current_image_index
        try:
            if viewer.current_image_path in viewer.image_files_in_dir:
                viewer.image_files_in_dir.remove(viewer.current_image_path)
        except Exception:
            pass
        if viewer.image_files_in_dir:
            target_index = original_index - 1 if (isinstance(original_index, int) and original_index > 0) else 0
            target_index = max(0, min(target_index, len(viewer.image_files_in_dir) - 1))
            viewer.current_image_index = target_index
            viewer.load_image(viewer.image_files_in_dir[viewer.current_image_index], source='nav')
        else:
            viewer.clear_display()
        viewer.update_button_states()
        try:
            viewer._rescan_current_dir()
        except Exception:
            pass
        try:
            viewer.statusBar().showMessage("삭제됨 — 실행 취소는 휴지통에서 가능", 3000)
        except Exception:
            pass
    except Exception as e:
        try:
            QMessageBox.critical(viewer, "삭제 오류", f"파일을 삭제할 수 없습니다:\n{str(e)}")
        except Exception:
            pass


def open_file(viewer: "JusawiViewer") -> None:
    from ..utils.file_utils import open_file_dialog_util
    file_path = open_file_dialog_util(viewer, getattr(viewer, "last_open_dir", ""))
    if not file_path:
        return
    try:
        viewer.log.info("open_dialog_selected | file=%s", os.path.basename(file_path))
    except Exception:
        pass
    success = viewer.load_image(file_path, source='open')
    if not success:
        return
    try:
        parent_dir = os.path.dirname(file_path)
        if parent_dir and os.path.isdir(parent_dir):
            # 마지막 폴더 저장 정책 반영
            try:
                if bool(getattr(viewer, "_remember_last_open_dir", True)):
                    viewer.last_open_dir = parent_dir
            except Exception:
                viewer.last_open_dir = parent_dir
            viewer.save_settings()
            try:
                viewer.log.info("open_dialog_applied_last_dir | dir=%s", os.path.basename(parent_dir))
            except Exception:
                pass
            # 옵션: 파일 열기 후 해당 폴더를 스캔하여 탐색 가능하도록 구성
            try:
                if bool(getattr(viewer, "_open_scan_dir_after_open", True)):
                    viewer.scan_directory(parent_dir)
                    # 현재 파일 인덱스로 이동
                    try:
                        if file_path in (viewer.image_files_in_dir or []):
                            viewer.current_image_index = (viewer.image_files_in_dir or []).index(file_path)
                            viewer.load_image_at_current_index()
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass


def save_current_image(viewer: "JusawiViewer") -> bool:
    if not viewer.load_successful or not viewer.current_image_path:
        return False
    if viewer._tf_rotation == 0 and not viewer._tf_flip_h and not viewer._tf_flip_v:
        return True
    try:
        img = getattr(viewer, "_fullres_image", None)
        if img is None or img.isNull():
            pix = viewer.image_display_area.originalPixmap()
            if not pix or pix.isNull():
                return False
            img = pix.toImage()
        try:
            if viewer._progress:
                viewer._progress.setVisible(True)
                viewer._progress.setValue(0)
            if viewer._cancel_btn:
                viewer._cancel_btn.setVisible(True)
        except Exception:
            pass

        def on_progress(p: int):
            try:
                if viewer._progress:
                    viewer._progress.setValue(max(0, min(100, int(p))))
            except Exception:
                pass

        def on_done(ok: bool, err: str):
            try:
                if viewer._progress:
                    viewer._progress.setVisible(False)
                if viewer._cancel_btn:
                    viewer._cancel_btn.setVisible(False)
            except Exception:
                pass
            if ok:
                viewer.load_image(viewer.current_image_path, source='save')
                viewer._mark_dirty(False)
                try:
                    viewer.statusBar().showMessage("저장됨", 1800)
                except Exception:
                    pass
            else:
                try:
                    QMessageBox.critical(viewer, "저장 오류", err or "파일 저장에 실패했습니다.")
                except Exception:
                    pass

        viewer.image_service.save_async(
            img,
            viewer.current_image_path,
            viewer.current_image_path,
            viewer._tf_rotation,
            viewer._tf_flip_h,
            viewer._tf_flip_v,
            quality=viewer._jpeg_quality,
            on_progress=on_progress,
            on_done=on_done,
        )
        return True
    except Exception as e:
        try:
            QMessageBox.critical(viewer, "저장 오류", str(e))
        except Exception:
            pass
        return False


def save_current_image_as(viewer: "JusawiViewer") -> bool:
    if not viewer.load_successful or not viewer.current_image_path:
        return False
    try:
        from PyQt6.QtWidgets import QFileDialog  # type: ignore[import]
        start_dir = os.path.dirname(viewer.current_image_path) if viewer.current_image_path else ""
        dest_path, _ = QFileDialog.getSaveFileName(viewer, "다른 이름으로 저장", start_dir)
        if not dest_path:
            return False
        img = getattr(viewer, "_fullres_image", None)
        if img is None or img.isNull():
            pix = viewer.image_display_area.originalPixmap()
            if not pix or pix.isNull():
                return False
            img = pix.toImage()
        ok, err = viewer.image_service.save_with_transform(
            img,
            viewer.current_image_path,
            dest_path,
            viewer._tf_rotation,
            viewer._tf_flip_h,
            viewer._tf_flip_v,
            quality=viewer._jpeg_quality,
        )
        if not ok:
            QMessageBox.critical(viewer, "저장 오류", err or "파일 저장에 실패했습니다.")
            return False
        viewer.load_image(dest_path, source='saveas')
        viewer._mark_dirty(False)
        return True
    except Exception as e:
        QMessageBox.critical(viewer, "저장 오류", str(e))
        return False



def reload_current_image(viewer: "JusawiViewer") -> None:
    try:
        path = viewer.current_image_path or ""
        if not path or not os.path.isfile(path):
            try:
                viewer.statusBar().showMessage("다시 읽을 이미지가 없습니다.", 2000)
            except Exception:
                pass
            return
        # 파일만 다시 로드 + 폴더 재스캔(리셋)
        try:
            # 모든 캐시를 초기화하여 처음 로드처럼 동작
            try:
                viewer.image_service.clear_all_caches()
            except Exception:
                pass
            viewer.image_service.invalidate_path(path)
        except Exception:
            pass
        # 폴더 재스캔: 현재 파일이 속한 디렉터리 기준
        try:
            dirp = os.path.dirname(path)
            if dirp and os.path.isdir(dirp):
                try:
                    # 썸네일 메모리 캐시도 초기화
                    viewer._clear_filmstrip_cache()
                except Exception:
                    pass
                # 인덱스 보존을 위해 현재 파일 경로를 기준으로 재스캔 후 현재 인덱스 복원
                viewer.scan_directory(dirp)
                try:
                    nc = os.path.normcase
                    if viewer.image_files_in_dir:
                        idx = [nc(p) for p in viewer.image_files_in_dir].index(nc(path))
                        viewer.current_image_index = idx
                except Exception:
                    pass
        except Exception:
            pass
        viewer.load_image(path, source='reload')
    except Exception:
        pass


def open_folder(viewer: "JusawiViewer") -> None:
    try:
        from PyQt6.QtWidgets import QFileDialog  # type: ignore[import]
        start_dir = getattr(viewer, "last_open_dir", "") if (viewer.last_open_dir and os.path.isdir(viewer.last_open_dir)) else ""
        dir_path = QFileDialog.getExistingDirectory(viewer, "폴더 선택", start_dir)
    except Exception:
        dir_path = ""
    if not dir_path:
        return
    viewer.scan_directory(dir_path)
    if 0 <= viewer.current_image_index < len(viewer.image_files_in_dir):
        viewer.load_image(viewer.image_files_in_dir[viewer.current_image_index], source='open_folder')
    else:
        try:
            viewer.clear_display()
        except Exception:
            pass
        viewer.statusBar().showMessage("폴더에 표시할 이미지가 없습니다.", 3000)
    try:
        if os.path.isdir(dir_path):
            if bool(getattr(viewer, "_remember_last_open_dir", True)):
                viewer.last_open_dir = dir_path
            viewer.save_settings()
    except Exception:
        pass
