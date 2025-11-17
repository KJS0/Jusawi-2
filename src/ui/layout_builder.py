from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget, QLabel, QSizePolicy  # type: ignore[import]
from PyQt6.QtGui import QKeySequence  # type: ignore[import]
from PyQt6.QtCore import Qt  # type: ignore[import]


def build_top_and_status_bars(viewer) -> None:
    # Menus
    from PyQt6.QtWidgets import QMenu  # type: ignore[import]
    viewer.recent_menu = QMenu(viewer)
    # 로그 버튼/메뉴 제거

    # Buttons and layout
    viewer.button_layout = QHBoxLayout()
    viewer.open_button = QPushButton("열기")
    viewer.open_button.clicked.connect(viewer.open_file)

    viewer.recent_button = QPushButton("최근 파일")
    viewer.recent_button.setMenu(viewer.recent_menu)

    # 정보 패널 토글 버튼
    viewer.info_button = QPushButton("정보")
    try:
        viewer.info_button.clicked.connect(viewer.toggle_info_panel)
    except Exception:
        pass

    # 폴더 지도 버튼
    viewer.map_button = QPushButton("지도")
    try:
        viewer.map_button.clicked.connect(viewer.open_folder_map_dialog)
    except Exception:
        pass

    # 로그 버튼 제거

    # EXIF 정보 버튼 제거됨

    # NEW: AI Analysis button
    viewer.ai_button = QPushButton("AI 분석")
    viewer.ai_button.clicked.connect(viewer.open_ai_analysis_dialog)

    # 배치 분석 버튼 제거
    viewer.batch_ai_button = None

    # NEW: Natural language search button
    viewer.search_button = QPushButton("검색")
    viewer.search_button.clicked.connect(viewer.open_natural_search_dialog)

    viewer.settings_button = QPushButton("설정")
    viewer.settings_button.clicked.connect(viewer.open_settings_dialog)

    viewer.fullscreen_button = QPushButton("전체화면")
    viewer.fullscreen_button.clicked.connect(viewer.toggle_fullscreen)
    # 이미지 더블클릭은 ImageView의 기본 동작(화면 맞춤 ↔ 100%)을 사용합니다.

    viewer.prev_button = QPushButton("이전")
    viewer.next_button = QPushButton("다음")
    viewer.prev_button.clicked.connect(viewer.show_prev_image)
    viewer.next_button.clicked.connect(viewer.show_next_image)

    viewer.zoom_out_button = QPushButton("축소")
    viewer.zoom_out_button.clicked.connect(viewer.zoom_out)
    viewer.fit_button = QPushButton("100%")
    # 일부 환경에서 viewer.reset_to_100 바인딩 시점 문제 회피: 직접 ImageView 메서드로 연결
    try:
        viewer.fit_button.clicked.connect(viewer.image_display_area.reset_to_100)
    except Exception:
        # 폴백: 뷰어 메서드가 있으면 사용
        try:
            viewer.fit_button.clicked.connect(viewer.reset_to_100)
        except Exception:
            pass
    viewer.zoom_in_button = QPushButton("확대")
    viewer.zoom_in_button.clicked.connect(viewer.zoom_in)

    viewer.rotate_left_button = QPushButton("↶90°")
    viewer.rotate_right_button = QPushButton("↷90°")
    viewer.rotate_left_button.clicked.connect(viewer.rotate_ccw_90)
    viewer.rotate_right_button.clicked.connect(viewer.rotate_cw_90)

    viewer.comparsion_button = QPushButton("사진 비교")
    viewer.comparsion_button.clicked.connect(viewer.open_comparsion_dialog)

    button_style = "color: #EAEAEA;"
    for btn in [
        viewer.open_button,
        viewer.recent_button,
        viewer.info_button,
        viewer.map_button,
        viewer.ai_button,
        viewer.settings_button,
        viewer.fullscreen_button,
        viewer.prev_button,
        viewer.next_button,
        viewer.zoom_out_button,
        viewer.fit_button,
        viewer.zoom_in_button,
        viewer.rotate_left_button,
        viewer.rotate_right_button,
        viewer.comparsion_button,
    ]:
        btn.setStyleSheet(button_style)
        btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    viewer.button_layout.addWidget(viewer.open_button)
    viewer.button_layout.addWidget(viewer.recent_button)
    # 정보 버튼
    viewer.button_layout.addWidget(viewer.info_button)
    # 지도 버튼
    viewer.button_layout.addWidget(viewer.map_button)
    # EXIF 버튼 제거됨
    viewer.button_layout.addWidget(viewer.ai_button)
    # 배치 버튼 제거됨
    viewer.button_layout.addWidget(viewer.comparsion_button)
    viewer.button_layout.addWidget(viewer.search_button)
    # Similar search button
    try:
        viewer.similar_button = QPushButton("유사 검색")
        viewer.similar_button.clicked.connect(viewer.open_similar_search_dialog)
        viewer.button_layout.addWidget(viewer.similar_button)
    except Exception:
        pass
    viewer.button_layout.addWidget(viewer.fullscreen_button)
    viewer.button_layout.addWidget(viewer.prev_button)
    viewer.button_layout.addWidget(viewer.next_button)
    viewer.button_layout.addWidget(viewer.zoom_out_button)
    viewer.button_layout.addWidget(viewer.fit_button)
    viewer.button_layout.addWidget(viewer.zoom_in_button)
    viewer.button_layout.addWidget(viewer.rotate_left_button)
    viewer.button_layout.addWidget(viewer.rotate_right_button)
    viewer.button_layout.addStretch(1)
    viewer.button_layout.addWidget(viewer.settings_button)

    viewer.button_bar = QWidget()
    # 테마 스타일은 theme.apply_ui_theme_and_spacing에서 일괄 적용
    viewer.button_bar.setStyleSheet("")
    viewer.button_bar.setLayout(viewer.button_layout)
    viewer.main_layout.insertWidget(0, viewer.button_bar)

    # Status bar
    viewer.status_left_label = QLabel("", viewer)
    viewer.status_right_label = QLabel("", viewer)
    viewer.status_left_label.setStyleSheet("")
    viewer.status_right_label.setStyleSheet("")
    viewer.statusBar().addWidget(viewer.status_left_label, 1)
    viewer.statusBar().addPermanentWidget(viewer.status_right_label)
    # 상태바 스타일은 theme.apply_ui_theme_and_spacing에서 설정
    viewer.statusBar().setStyleSheet("")

    # Progress and cancel button on status bar
    try:
        from PyQt6.QtWidgets import QProgressBar  # type: ignore[import]
        viewer._progress = QProgressBar(viewer)
        viewer._progress.setFixedWidth(160)
        viewer._progress.setRange(0, 100)
        viewer._progress.setVisible(False)
        viewer._cancel_btn = QPushButton("취소", viewer)
        viewer._cancel_btn.setVisible(False)
        viewer._cancel_btn.clicked.connect(lambda: getattr(viewer.image_service, "cancel_save", lambda: None)())
        viewer.statusBar().addPermanentWidget(viewer._progress)
        viewer.statusBar().addPermanentWidget(viewer._cancel_btn)
    except Exception:
        viewer._progress = None
        viewer._cancel_btn = None

    # DnD enable on status bar and labels
    try:
        viewer.statusBar().setAcceptDrops(True)
        viewer.statusBar().installEventFilter(viewer)
        viewer.status_left_label.setAcceptDrops(True)
        viewer.status_left_label.installEventFilter(viewer)
        viewer.status_right_label.setAcceptDrops(True)
        viewer.status_right_label.installEventFilter(viewer)
    except Exception:
        pass

    # Shortcuts setup (space to toggle animation)
    viewer.setup_shortcuts()
    try:
        from PyQt6.QtWidgets import QShortcut  # type: ignore[import]
        viewer.anim_space_shortcut = QShortcut(QKeySequence("Space"), viewer)
        viewer.anim_space_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        viewer.anim_space_shortcut.activated.connect(viewer.anim_toggle_play)
    except Exception:
        pass

