from __future__ import annotations

from PyQt6.QtCore import QUrl  # type: ignore[import]
from PyQt6.QtGui import QDesktopServices  # type: ignore[import]


def handle_mouse_press(owner, event) -> bool:
    try:
        if event is not None and getattr(owner, "info_map_label", None) is not None:
            if owner.info_map_label.isVisible() and owner.info_panel.isVisible():
                if owner.info_map_label.rect().contains(owner.info_map_label.mapFrom(owner, event.position().toPoint())):
                    # 외부 링크 오픈 대신 폴더 지도 다이얼로그 표시
                    try:
                        from .folder_map_dialog import FolderMapDialog  # type: ignore
                        paths = getattr(owner, "image_files_in_dir", []) or []
                        if not paths:
                            # 폴더 파일 목록이 없으면 툴팁 링크로 폴백
                            raise RuntimeError("no_paths")
                        cur = getattr(owner, "current_image_path", None)
                        dlg_ = FolderMapDialog(owner, paths, cur)
                        # 테마 적용 시도
                        try:
                            setattr(owner, "folder_map_dialog", dlg_)
                            from .theme import apply_ui_theme_and_spacing as _apply_theme  # type: ignore
                            _apply_theme(owner)
                        except Exception:
                            pass
                        dlg_.exec()
                        return True
                    except Exception:
                        # 폴백: 기존 외부 링크 열기
                        link = owner.info_map_label.toolTip() if hasattr(owner.info_map_label, 'toolTip') else ""
                        if link and isinstance(link, str) and link.startswith("http"):
                            QDesktopServices.openUrl(QUrl(link))
                            return True
    except Exception:
        return False
    return False


