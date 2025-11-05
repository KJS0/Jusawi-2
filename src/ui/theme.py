from __future__ import annotations

def style_rating_bar(viewer) -> None:
    try:
        from . import rating_bar as _rating_bar
        _rating_bar.apply_theme(viewer, False)
    except Exception:
        pass


def apply_ui_theme_and_spacing(viewer) -> None:
    # 여백/간격은 고정 기본값 적용
    try:
        viewer.main_layout.setContentsMargins(5, 5, 5, 5)
        viewer.main_layout.setSpacing(6)
    except Exception:
        pass
    resolved = 'dark'
    try:
        viewer._resolved_theme = resolved
    except Exception:
        pass
    bg = "#373737"
    fg = "#EAEAEA"
    bar_bg = "#373737"
    try:
        viewer.centralWidget().setStyleSheet(f"background-color: {bg};")
    except Exception:
        pass
    try:
        button_style = f"color: {fg}; background-color: transparent;"
        for btn in [
            getattr(viewer, 'open_button', None),
            getattr(viewer, 'recent_button', None),
            getattr(viewer, 'info_button', None),
            getattr(viewer, 'fullscreen_button', None),
            getattr(viewer, 'prev_button', None),
            getattr(viewer, 'next_button', None),
            getattr(viewer, 'zoom_out_button', None),
            getattr(viewer, 'fit_button', None),
            getattr(viewer, 'zoom_in_button', None),
            getattr(viewer, 'rotate_left_button', None),
            getattr(viewer, 'rotate_right_button', None),
            getattr(viewer, 'settings_button', None),
            getattr(viewer, 'similar_button', None),
        ]:
            if btn:
                btn.setStyleSheet(button_style)
        # 다크 테마에서 AI 분석/검색은 흰색을 명시적으로 강제
        if resolved == 'dark':
            try:
                viewer.ai_button.setStyleSheet("color: #FFFFFF; background-color: transparent;")
            except Exception:
                pass
            try:
                viewer.search_button.setStyleSheet("color: #FFFFFF; background-color: transparent;")
            except Exception:
                pass
            try:
                if getattr(viewer, 'similar_button', None):
                    viewer.similar_button.setStyleSheet("color: #FFFFFF; background-color: transparent;")
            except Exception:
                pass
    except Exception:
        pass
    try:
        viewer.statusBar().setStyleSheet(
            f"QStatusBar {{ background-color: {bar_bg}; border-top: 1px solid {bar_bg}; color: {fg}; }} "
            f"QStatusBar QLabel {{ color: {fg}; }} "
            "QStatusBar::item { border: 0px; }"
        )
        viewer.status_left_label.setStyleSheet(f"color: {fg};")
        viewer.status_right_label.setStyleSheet(f"color: {fg};")
    except Exception:
        pass
    try:
        from PyQt6.QtGui import QColor, QBrush  # type: ignore[import]
        viewer.image_display_area.setBackgroundBrush(QBrush(QColor("#373737")))
    except Exception:
        pass
    try:
        viewer.button_bar.setStyleSheet(f"background-color: transparent; QPushButton {{ color: {fg}; }}")
    except Exception:
        pass

    # Info panel theming (dark only) + 동적 폰트 크기
    try:
        if getattr(viewer, 'info_text', None) is not None:
            try:
                total_w = max(640, int(viewer.width()))
                scaled = max(16, min(24, int(total_w / 80)))
            except Exception:
                scaled = 20
            viewer.info_text.setStyleSheet(
                f"QTextEdit {{ color: #EAEAEA; background-color: #2B2B2B; border: 1px solid #444; font-size: {scaled}px; line-height: 140%; }} QTextEdit:disabled {{ color: #777777; }}"
            )
    except Exception:
        pass
    try:
        if getattr(viewer, 'info_map_label', None) is not None:
            viewer.info_map_label.setStyleSheet("QLabel { background-color: #2B2B2B; color: #AAAAAA; border: 1px solid #444; }")
    except Exception:
        pass

    # Filmstrip theming (dark only, including scrollbar)
    try:
        fs = getattr(viewer, 'filmstrip', None)
        if fs is not None:
            fs.setStyleSheet(
                "QListView, QListView::viewport { background-color: #1F1F1F; }"
                " QScrollBar:horizontal { background: #2B2B2B; height: 12px; }"
                " QScrollBar::handle:horizontal { background: #555; min-width: 24px; border-radius: 6px; }"
                " QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { background: transparent; width: 0px; }"
                " QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: #2B2B2B; }"
            )
            try:
                from PyQt6.QtGui import QPalette, QColor  # type: ignore
                pal = fs.viewport().palette()
                pal.setColor(QPalette.ColorRole.Base, QColor("#1F1F1F"))
                pal.setColor(QPalette.ColorRole.Window, QColor("#1F1F1F"))
                fs.viewport().setPalette(pal)
                fs.viewport().setAutoFillBackground(True)
            except Exception:
                pass
    except Exception:
        pass

    # Rating bar theming centralized (base)
    try:
        style_rating_bar(viewer)
    except Exception:
        pass

    # Dialogs and common widgets theming (extend coverage)
    try:
        fg = fg  # already defined
        bg = bg
        # Apply to known dialogs if present
        for name in [
            'settings_dialog',
            'shortcuts_help_dialog',
            'ai_analysis_dialog',
            'natural_search_dialog',
            'similar_search_dialog',
        ]:
            dlg = getattr(viewer, name, None)
            if dlg is None:
                continue
            try:
                dlg.setStyleSheet(
                    f"QDialog {{ background-color: {bg}; color: {fg}; }}"
                    f" QLabel {{ color: {fg}; }}"
                    f" QLineEdit, QComboBox, QTextEdit {{ background-color: #2B2B2B; color: {fg}; border: 1px solid #444; }}"
                    f" QPushButton {{ color: {fg}; background-color: transparent; border: 1px solid {'#9E9E9E' if resolved=='light' else '#555'}; padding: 4px 8px; border-radius: 4px; }}"
                )
            except Exception:
                pass
    except Exception:
        pass

    # QToolTip 전역 스타일(어두운 배경 위 가독성 확보: 흰 글자)
    try:
        from PyQt6.QtWidgets import QApplication  # type: ignore
        app = QApplication.instance()
        if app is not None:
            prev = app.styleSheet() or ""
            tooltip_css = (
                " QToolTip {"
                " color: #FFFFFF;"
                " background-color: rgba(0,0,0,200);"
                " border: 1px solid #777777;"
                " padding: 4px 6px;"
                " }"
            )
            # 중복 추가 방지
            if "QToolTip" not in prev:
                app.setStyleSheet(prev + tooltip_css)
    except Exception:
        pass


