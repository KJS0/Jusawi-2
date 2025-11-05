import os


def save_last_session(viewer) -> None:
    try:
        last = {
            "file_path": viewer.current_image_path or "",
            "dir_path": os.path.dirname(viewer.current_image_path) if viewer.current_image_path else "",
            "dir_index": int(getattr(viewer, "current_image_index", -1) or -1),
            "view_mode": getattr(viewer, "_session_preferred_view_mode", getattr(viewer.image_display_area, "_view_mode", 'fit')),
            "scale": float(getattr(viewer, "_last_scale", 1.0) or 1.0),
            "fullscreen": bool(viewer.is_fullscreen),
            "window_geometry": viewer.saveGeometry(),
        }
        viewer.settings.setValue("session/last", last)
        viewer.save_settings()
    except Exception:
        pass


def restore_last_session(viewer) -> None:
    try:
        # 시작 시 복원 정책: always/ask/never
        policy = str(getattr(viewer, "_startup_restore_policy", "always"))
        if policy not in ("always", "ask", "never"):
            policy = "always"
        if policy == "never":
            return
        last = viewer.settings.value("session/last", {}, dict)
        if not isinstance(last, dict):
            return
        fpath = last.get("file_path") or ""
        dpath = last.get("dir_path") or ""
        dindex = int(last.get("dir_index") or -1)
        vmode = last.get("view_mode") or 'fit'
        scale = float(last.get("scale") or 1.0)
        # 디렉터리 컨텍스트가 있으면 이를 우선 복원하고 인덱스를 반영
        if dpath and os.path.isdir(dpath):
            viewer.scan_directory(dpath)
            # 우선순위: 저장된 파일 경로(fpath)가 해당 폴더 목록에 있으면 그 인덱스를 사용
            try:
                idx = -1
                if fpath and os.path.isfile(fpath) and (viewer.image_files_in_dir or []):
                    try:
                        from os.path import normcase
                        lst = [normcase(p) for p in (viewer.image_files_in_dir or [])]
                        fp = normcase(fpath)
                        if fp in lst:
                            idx = lst.index(fp)
                    except Exception:
                        pass
                # fpath로 못 찾으면 저장된 dindex를 사용
                if idx < 0 and 0 <= dindex < len(viewer.image_files_in_dir):
                    idx = dindex
                if idx >= 0:
                    viewer.current_image_index = idx
            except Exception:
                pass
            if 0 <= viewer.current_image_index < len(viewer.image_files_in_dir):
                viewer.load_image(viewer.image_files_in_dir[viewer.current_image_index], source='restore')
                try:
                    # 로드시 플래그/별점 즉시 반영 보조
                    from ..ui import rating_bar  # type: ignore
                    rating_bar.refresh(viewer)
                except Exception:
                    pass
        elif fpath and os.path.isfile(fpath):
            # 폴더 정보가 없을 때만 단일 파일 복원
            # 묻기 정책이면 확인
            if policy == "ask":
                try:
                    from PyQt6.QtWidgets import QMessageBox  # type: ignore[import]
                    r = QMessageBox.question(viewer, "세션 복원", f"마지막 파일을 다시 열까요?\n{fpath}")
                    if r != QMessageBox.StandardButton.Yes:
                        return
                except Exception:
                    pass
            viewer.load_image(fpath, source='restore')
            try:
                from ..ui import rating_bar  # type: ignore
                rating_bar.refresh(viewer)
            except Exception:
                pass
        else:
            # 보조 복원: 최근 파일 목록에서 첫 항목 시도
            try:
                recent = viewer.recent_files or []
                if recent:
                    first_path = recent[0].get("path") if isinstance(recent[0], dict) else str(recent[0])
                    if first_path and os.path.isfile(first_path):
                        if policy == "ask":
                            try:
                                from PyQt6.QtWidgets import QMessageBox  # type: ignore[import]
                                r = QMessageBox.question(viewer, "세션 복원", f"최근 파일을 다시 열까요?\n{first_path}")
                                if r != QMessageBox.StandardButton.Yes:
                                    return
                            except Exception:
                                pass
                        viewer.load_image(first_path, source='restore')
            except Exception:
                pass
        # 보기 모드/배율 적용
        mode_to_apply = vmode or 'fit'
        if mode_to_apply == 'fit':
            viewer.image_display_area.fit_to_window()
        elif mode_to_apply == 'fit_width':
            viewer.image_display_area.fit_to_width()
        elif mode_to_apply == 'fit_height':
            viewer.image_display_area.fit_to_height()
        elif mode_to_apply == 'actual':
            viewer.image_display_area.reset_to_100()
        if vmode == 'free' and viewer.image_display_area:
            viewer.image_display_area.set_absolute_scale(scale)
        # 한 틱 뒤 한 번 더 반영
        try:
            from PyQt6.QtCore import QTimer  # type: ignore
            QTimer.singleShot(0, lambda: __import__('importlib').import_module('src.ui.rating_bar').refresh(viewer))
        except Exception:
            pass
        # 창 위치/크기 복원은 기본 비활성화 (_restore_window_geometry 가 True일 때만)
        try:
            if bool(getattr(viewer, "_restore_window_geometry", False)):
                geom = last.get("window_geometry")
                if geom:
                    viewer.restoreGeometry(geom)
        except Exception:
            pass
    except Exception:
        pass


