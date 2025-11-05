from __future__ import annotations


def set_global_shortcuts_enabled(viewer, enabled: bool) -> None:
    viewer._global_shortcuts_enabled = bool(enabled)
    try:
        for sc in getattr(viewer, "_shortcuts", []) or []:
            try:
                sc.setEnabled(viewer._global_shortcuts_enabled)
            except Exception:
                pass
    except Exception:
        pass
    try:
        if hasattr(viewer, "anim_space_shortcut") and viewer.anim_space_shortcut:
            viewer.anim_space_shortcut.setEnabled(viewer._global_shortcuts_enabled)
    except Exception:
        pass
    # 레이팅/플래그 단축키(ApplicationShortcut)도 함께 토글
    try:
        for sc in getattr(viewer, "_rating_shortcuts", []) or []:
            try:
                sc.setEnabled(viewer._global_shortcuts_enabled)
            except Exception:
                pass
    except Exception:
        pass
    try:
        for name in ("_flag_sc_z", "_flag_sc_x", "_flag_sc_c"):
            try:
                sc = getattr(viewer, name, None)
                if sc is not None:
                    sc.setEnabled(viewer._global_shortcuts_enabled)
            except Exception:
                pass
    except Exception:
        pass


