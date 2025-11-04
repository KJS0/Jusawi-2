class NavigationController:
    def __init__(self, viewer):
        self._viewer = viewer
        # 연속 탐색 스로틀을 위한 마지막 전환 시각(ms)
        try:
            from PyQt6.QtCore import QDateTime  # type: ignore[import]
            self._now_ms = lambda: int(QDateTime.currentMSecsSinceEpoch())
        except Exception:
            import time
            self._now_ms = lambda: int(time.time() * 1000)
        self._last_nav_ms = 0

    def _throttle_ok(self) -> bool:
        try:
            min_interval = int(getattr(self._viewer, "_nav_min_interval_ms", 100))
        except Exception:
            min_interval = 100
        now = self._now_ms()
        if now - self._last_nav_ms < max(0, min_interval):
            return False
        self._last_nav_ms = now
        return True

    def can_prev(self) -> bool:
        return self._viewer.current_image_index > 0

    def can_next(self) -> bool:
        return self._viewer.current_image_index < len(self._viewer.image_files_in_dir) - 1

    def show_prev_image(self) -> None:
        viewer = self._viewer
        if not self._throttle_ok():
            return
        idx = int(getattr(viewer, "current_image_index", -1))
        count = int(len(getattr(viewer, "image_files_in_dir", []) or []))
        if idx <= 0:
            if bool(getattr(viewer, "_nav_wrap_ends", False)) and count > 0:
                if getattr(viewer, "_is_dirty", False):
                    if not viewer._handle_dirty_before_action():
                        return
                viewer.current_image_index = count - 1
                viewer.load_image_at_current_index()
                # 이동 후 필름스트립 중앙 정렬
                try:
                    if hasattr(viewer, 'filmstrip') and viewer.filmstrip is not None and bool(getattr(viewer, '_filmstrip_auto_center', True)):
                        viewer.filmstrip.set_current_index(viewer.current_image_index)
                        try:
                            from PyQt6.QtWidgets import QAbstractItemView  # type: ignore[import]
                            idx_q = viewer.filmstrip.model().index(viewer.current_image_index, 0)
                            viewer.filmstrip.scrollTo(idx_q, QAbstractItemView.ScrollHint.PositionAtCenter)
                        except Exception:
                            pass
                except Exception:
                    pass
            return
        if getattr(viewer, "_is_dirty", False):
            if not viewer._handle_dirty_before_action():
                return
        viewer.current_image_index -= 1
        viewer.load_image_at_current_index()
        # 이동 후 필름스트립 중앙 정렬
        try:
            if hasattr(viewer, 'filmstrip') and viewer.filmstrip is not None and bool(getattr(viewer, '_filmstrip_auto_center', True)):
                viewer.filmstrip.set_current_index(viewer.current_image_index)
                try:
                    from PyQt6.QtWidgets import QAbstractItemView  # type: ignore[import]
                    idx_q = viewer.filmstrip.model().index(viewer.current_image_index, 0)
                    viewer.filmstrip.scrollTo(idx_q, QAbstractItemView.ScrollHint.PositionAtCenter)
                except Exception:
                    pass
        except Exception:
            pass

    def show_next_image(self) -> None:
        viewer = self._viewer
        if not self._throttle_ok():
            return
        idx = int(getattr(viewer, "current_image_index", -1))
        count = int(len(getattr(viewer, "image_files_in_dir", []) or []))
        if idx >= count - 1:
            if bool(getattr(viewer, "_nav_wrap_ends", False)) and count > 0:
                if getattr(viewer, "_is_dirty", False):
                    if not viewer._handle_dirty_before_action():
                        return
                viewer.current_image_index = 0
                viewer.load_image_at_current_index()
                # 이동 후 필름스트립 중앙 정렬
                try:
                    if hasattr(viewer, 'filmstrip') and viewer.filmstrip is not None and bool(getattr(viewer, '_filmstrip_auto_center', True)):
                        viewer.filmstrip.set_current_index(viewer.current_image_index)
                        try:
                            from PyQt6.QtWidgets import QAbstractItemView  # type: ignore[import]
                            idx_q = viewer.filmstrip.model().index(viewer.current_image_index, 0)
                            viewer.filmstrip.scrollTo(idx_q, QAbstractItemView.ScrollHint.PositionAtCenter)
                        except Exception:
                            pass
                except Exception:
                    pass
            return
        if getattr(viewer, "_is_dirty", False):
            if not viewer._handle_dirty_before_action():
                return
        viewer.current_image_index += 1
        viewer.load_image_at_current_index()
        # 이동 후 필름스트립 중앙 정렬
        try:
            if hasattr(viewer, 'filmstrip') and viewer.filmstrip is not None and bool(getattr(viewer, '_filmstrip_auto_center', True)):
                viewer.filmstrip.set_current_index(viewer.current_image_index)
                try:
                    from PyQt6.QtWidgets import QAbstractItemView  # type: ignore[import]
                    idx_q = viewer.filmstrip.model().index(viewer.current_image_index, 0)
                    viewer.filmstrip.scrollTo(idx_q, QAbstractItemView.ScrollHint.PositionAtCenter)
                except Exception:
                    pass
        except Exception:
            pass

    def load_image_at_current_index(self) -> None:
        viewer = self._viewer
        if 0 <= viewer.current_image_index < len(viewer.image_files_in_dir):
            viewer.load_image(viewer.image_files_in_dir[viewer.current_image_index], source='nav')
            # 내비게이션으로 전환된 경우에도 원본 업그레이드를 확실히 예약
            try:
                if viewer.load_successful and not viewer._is_current_file_animation() and not getattr(viewer, "_movie", None):
                    if viewer._fullres_upgrade_timer.isActive():
                        viewer._fullres_upgrade_timer.stop()
                    # 빠른 전환 중 과도한 디코드 폭주 방지를 위해 약간의 텀
                    viewer._fullres_upgrade_timer.start(100)
                    # 다음 틱이 아닌, 약간의 텀 뒤 업그레이드 시도(키 꾹 누름 대비)
                    from PyQt6.QtCore import QTimer  # type: ignore[import]
                    QTimer.singleShot(100, getattr(viewer, "_upgrade_to_fullres_if_needed", lambda: None))
            except Exception:
                pass

    def update_button_states(self) -> None:
        viewer = self._viewer
        num_images = len(viewer.image_files_in_dir)
        is_valid_index = 0 <= viewer.current_image_index < num_images

        if bool(getattr(viewer, "_nav_wrap_ends", False)):
            # 래핑일 때는 항상 활성화(목록이 1개 이상)
            viewer.prev_button.setEnabled(num_images > 0)
            viewer.next_button.setEnabled(num_images > 0)
        else:
            viewer.prev_button.setEnabled(is_valid_index and viewer.current_image_index > 0)
            viewer.next_button.setEnabled(is_valid_index and viewer.current_image_index < num_images - 1)
        has_image = bool(viewer.load_successful)
        viewer.zoom_in_button.setEnabled(has_image)
        viewer.zoom_out_button.setEnabled(has_image)
        viewer.fit_button.setEnabled(has_image)


def show_prev_image(viewer) -> None:
    if viewer.current_image_index > 0:
        # Dirty 확인은 viewer.load_image에서도 수행되나, 인덱스 롤백을 방지하려면 사전 확인이 더 안전
        if getattr(viewer, "_is_dirty", False):
            if not viewer._handle_dirty_before_action():
                return
        viewer.current_image_index -= 1
        viewer.load_image_at_current_index()


def show_next_image(viewer) -> None:
    if viewer.current_image_index < len(viewer.image_files_in_dir) - 1:
        if getattr(viewer, "_is_dirty", False):
            if not viewer._handle_dirty_before_action():
                return
        viewer.current_image_index += 1
        viewer.load_image_at_current_index()


def load_image_at_current_index(viewer) -> None:
    if 0 <= viewer.current_image_index < len(viewer.image_files_in_dir):
        viewer.load_image(viewer.image_files_in_dir[viewer.current_image_index], source='nav')
        # 함수형 경로에서도 동일하게 업그레이드 예약
        try:
            if viewer.load_successful and not viewer._is_current_file_animation() and not getattr(viewer, "_movie", None):
                if not bool(getattr(viewer, "_pause_auto_upgrade", False)):
                    if viewer._fullres_upgrade_timer.isActive():
                        viewer._fullres_upgrade_timer.stop()
                    delay = int(getattr(viewer, "_fullres_upgrade_delay_ms", 120))
                    viewer._fullres_upgrade_timer.start(max(0, delay))
                    from PyQt6.QtCore import QTimer  # type: ignore[import]
                    QTimer.singleShot(max(0, delay), getattr(viewer, "_upgrade_to_fullres_if_needed", lambda: None))
        except Exception:
            pass


def update_button_states(viewer) -> None:
    num_images = len(viewer.image_files_in_dir)
    is_valid_index = 0 <= viewer.current_image_index < num_images

    viewer.prev_button.setEnabled(is_valid_index and viewer.current_image_index > 0)
    viewer.next_button.setEnabled(is_valid_index and viewer.current_image_index < num_images - 1)
    has_image = bool(viewer.load_successful)
    viewer.zoom_in_button.setEnabled(has_image)
    viewer.zoom_out_button.setEnabled(has_image)
    viewer.fit_button.setEnabled(has_image)


