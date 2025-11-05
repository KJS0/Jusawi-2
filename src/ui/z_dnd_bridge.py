from __future__ import annotations

from ..dnd.dnd_handlers import handle_dropped_files as _handle_files  # type: ignore
from . import event_handlers as ev


def handle_dropped_files(owner, files):
    return _handle_files(owner, files)


def handle_dropped_folders(owner, folders):
    if not folders:
        owner.statusBar().showMessage("폴더가 없습니다.", 3000)
        return
    # 설정: 폴더 드롭 허용 여부
    if not bool(getattr(owner, "_drop_allow_folder", False)):
        owner.statusBar().showMessage("설정에서 폴더 드롭을 허용하지 않았습니다.", 3000)
        return
    dir_path = folders[0]
    # 재귀 스캔 옵션은 services.image_service에서 지원 범위를 넘으므로 우선 단일 폴더 스캔만
    owner.scan_directory(dir_path)
    if 0 <= owner.current_image_index < len(owner.image_files_in_dir):
        owner.load_image(owner.image_files_in_dir[owner.current_image_index])
    else:
        try:
            owner.clear_display()
        except Exception:
            pass
        owner.statusBar().showMessage("폴더에 표시할 이미지가 없습니다.", 3000)

def drag_enter(owner, event):
    ev.drag_enter(owner, event)


def drag_move(owner, event):
    ev.drag_move(owner, event)


def drop(owner, event):
    ev.drop(owner, event)


