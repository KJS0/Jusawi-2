from __future__ import annotations

import os
import hashlib
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from PyQt6.QtCore import (
    Qt, QSize, QRect, QModelIndex, QAbstractListModel, pyqtSignal, pyqtSlot, QRunnable, QThreadPool, QStandardPaths
)
from PyQt6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QFont, QImageReader
)
from PyQt6.QtWidgets import (
    QListView, QStyledItemDelegate, QStyleOptionViewItem, QWidget, QApplication, QStyle
)


# 썸네일 크기 단계(짧은 변 기준)
THUMB_STEPS = [72, 96, 128, 160]


def _cache_root() -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".cache", "Jusawi")
    path = os.path.join(base, "thumbs")
    os.makedirs(path, exist_ok=True)
    return path


def cache_key(path: str, mtime: float, size: int) -> str:
    h = hashlib.sha1()
    h.update(path.encode("utf-8", errors="ignore"))
    h.update(str(int(mtime)).encode("ascii"))
    h.update(str(int(size)).encode("ascii"))
    return h.hexdigest()


class Roles:
    PathRole = Qt.ItemDataRole.UserRole + 1
    PixmapRole = Qt.ItemDataRole.UserRole + 2
    MetaRole = Qt.ItemDataRole.UserRole + 3
    RatingRole = Qt.ItemDataRole.UserRole + 4
    FlagRole = Qt.ItemDataRole.UserRole + 5
    LabelRole = Qt.ItemDataRole.UserRole + 6


@dataclass
class FilmItem:
    path: str
    mtime: float
    meta: Dict


class ThumbCache:
    def __init__(self, quality: int = 85, root_dir: str | None = None):
        self.root = root_dir or _cache_root()
        self.quality = int(quality)
        self._mem: Dict[str, QPixmap] = {}
        self._lock = threading.Lock()

    def _disk_path(self, key: str, size: int) -> str:
        return os.path.join(self.root, f"{key}_{size}.jpg")

    def get_pixmap(self, key: str, size: int) -> Optional[QPixmap]:
        with self._lock:
            pm = self._mem.get(f"{key}|{size}")
            if pm is not None and not pm.isNull():
                return pm
        p = self._disk_path(key, size)
        if os.path.exists(p):
            pm = QPixmap(p)
            if not pm.isNull():
                with self._lock:
                    self._mem[f"{key}|{size}"] = pm
                return pm
        return None

    def put_pixmap(self, key: str, size: int, pm: QPixmap):
        if pm is None or pm.isNull():
            return
        with self._lock:
            self._mem[f"{key}|{size}"] = pm
        p = self._disk_path(key, size)
        try:
            pm.save(p, "JPG", self.quality)
        except Exception:
            pass

    def clear(self) -> None:
        with self._lock:
            self._mem.clear()


## EXIF 메타는 필름 스트립에서 사용하지 않음(썸네일/경로만)

class ThumbTask(QRunnable):
    def __init__(self, row: int, path: str, mtime: float, size: int, cache: ThumbCache, signal, dpr: float = 1.0):
        super().__init__()
        self.row = row
        self.path = path
        self.mtime = mtime
        self.size = int(size)
        try:
            self._dpr = float(dpr)
        except Exception:
            self._dpr = 1.0
        self.cache = cache
        self.signal = signal

    def run(self):
        eff_px = max(1, int(round(self.size * max(1.0, self._dpr))))
        key = cache_key(self.path, self.mtime, eff_px)
        pm = self.cache.get_pixmap(key, eff_px)
        if pm is not None:
            self.signal.emit(self.row, pm)
            return
        reader = QImageReader(self.path)
        reader.setAutoTransform(True)
        try:
            orig = reader.size()
            ow, oh = int(orig.width()), int(orig.height())
            if ow > 0 and oh > 0:
                if ow < oh:
                    target_w = eff_px
                    target_h = int(eff_px * oh / max(1, ow))
                else:
                    target_w = int(eff_px * ow / max(1, oh))
                    target_h = eff_px
                reader.setScaledSize(QSize(target_w, target_h))
        except Exception:
            pass
        img = reader.read()
        if img.isNull():
            self.signal.emit(self.row, QPixmap())
            return
        # 썸네일 색 변환(옵션)
        try:
            parent = getattr(self, 'parent', lambda: None)()
        except Exception:
            parent = None
        try:
            win = parent if parent is not None else None
            if win is not None and bool(getattr(win, "_thumb_convert_to_srgb", True)):
                try:
                    from ..services.image_service import _convert_to_srgb  # type: ignore
                    img = _convert_to_srgb(img)
                except Exception:
                    pass
        except Exception:
            pass
        if img.width() > eff_px or img.height() > eff_px:
            img = img.scaled(eff_px, eff_px, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        pm = QPixmap.fromImage(img)
        try:
            if hasattr(pm, 'setDevicePixelRatio'):
                pm.setDevicePixelRatio(max(1.0, self._dpr))
        except Exception:
            pass
        if not pm.isNull():
            self.cache.put_pixmap(key, eff_px, pm)
        self.signal.emit(self.row, pm)


class FilmstripModel(QAbstractListModel):
    thumbReady = pyqtSignal(int, QPixmap)

    def __init__(self, cache: ThumbCache, pool: QThreadPool, get_size_callable, get_dpr_callable):
        super().__init__()
        self._items: List[FilmItem] = []
        self._pix: Dict[int, QPixmap] = {}
        self._cache = cache
        self._pool = pool
        self._get_size = get_size_callable
        self._get_dpr = get_dpr_callable
        self.thumbReady.connect(self._on_thumb_ready)
        self._paths_sig: Tuple[str, ...] = tuple()

    def set_items(self, paths: List[str], current_index: int = -1):
        new_sig = tuple(paths)
        if new_sig == self._paths_sig:
            return
        items: List[FilmItem] = []
        for p in paths:
            try:
                st = os.stat(p)
                mt = float(st.st_mtime)
            except Exception:
                mt = 0.0
            # ratings_store에서 기존 값이 있으면 로드
            meta: Dict = {}
            try:
                from ..services.ratings_store import get_image  # type: ignore
                row = get_image(p)
                if row:
                    meta["rating"] = int(row.get("rating", 0))
                    meta["label"] = row.get("label")
                    meta["flag"] = row.get("flag") or "unflagged"
            except Exception:
                pass
            items.append(FilmItem(path=p, mtime=mt, meta=meta))
        self.beginResetModel()
        self._items = items
        self._pix.clear()
        self._paths_sig = new_sig
        self.endResetModel()
        if 0 <= current_index < len(self._items):
            self.dataChanged.emit(self.index(current_index, 0), self.index(current_index, 0), [])
            try:
                # 현재 인덱스 선택 반영 후 상단 바(별점/플래그) 갱신
                from . import rating_bar  # type: ignore
                parent = getattr(self, 'parent', lambda: None)()
                if parent is None:
                    # 모델이 뷰어에 바인딩되어 있으므로 상위 위젯 탐색
                    parent = getattr(self, 'window', lambda: None)()
                if parent is not None:
                    rating_bar.refresh(parent)
            except Exception:
                pass

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._items)

    def data(self, idx: QModelIndex, role: int):
        if not idx.isValid():
            return None
        it = self._items[idx.row()]
        if role == Roles.PathRole:
            return it.path
        if role == Roles.MetaRole:
            return it.meta
        if role == Roles.RatingRole:
            return (it.meta or {}).get("rating", 0)
        if role == Roles.FlagRole:
            return (it.meta or {}).get("flag", "unflagged")
        if role == Roles.LabelRole:
            return (it.meta or {}).get("label")
        if role == Roles.PixmapRole:
            pm = self._pix.get(idx.row())
            if pm is None or pm.isNull():
                self._maybe_request_thumb(idx.row())
            return pm
        if role == Qt.ItemDataRole.ToolTipRole:
            # 파일명만 툴팁으로 표시
            return os.path.basename(it.path)
        if role == Qt.ItemDataRole.AccessibleTextRole:
            return os.path.basename(it.path)
        return None

    def _maybe_request_thumb(self, row: int):
        if not (0 <= row < len(self._items)):
            return
        it = self._items[row]
        size = int(self._get_size())
        try:
            dpr = float(self._get_dpr())
        except Exception:
            dpr = 1.0
        eff_px = max(1, int(round(size * max(1.0, dpr))))
        key = cache_key(it.path, it.mtime, eff_px)
        pm = self._cache.get_pixmap(key, eff_px)
        if pm is not None:
            try:
                if hasattr(pm, 'setDevicePixelRatio'):
                    pm.setDevicePixelRatio(max(1.0, dpr))
            except Exception:
                pass
            self._pix[row] = pm
            self.dataChanged.emit(self.index(row, 0), self.index(row, 0), [Roles.PixmapRole])
            return
        task = ThumbTask(row, it.path, it.mtime, size, self._cache, self.thumbReady, dpr)
        self._pool.start(task)

    @pyqtSlot(int, QPixmap)
    def _on_thumb_ready(self, row: int, pm: QPixmap):
        if not (0 <= row < len(self._items)):
            return
        self._pix[row] = pm
        self.dataChanged.emit(self.index(row, 0), self.index(row, 0), [Roles.PixmapRole])

    # 메타 업데이트는 사용하지 않음

    # --- 외부 갱신용 메타 업데이트 ---
    def update_item_meta_by_path(self, path: str, rating=None, label=None, flag=None) -> None:
        for i, it in enumerate(self._items):
            if os.path.normcase(it.path) == os.path.normcase(path):
                if rating is not None:
                    it.meta["rating"] = int(rating)
                if label is not None or label is None:
                    it.meta["label"] = label
                if flag is not None:
                    it.meta["flag"] = str(flag)
                self.dataChanged.emit(self.index(i, 0), self.index(i, 0), [Roles.MetaRole, Roles.RatingRole, Roles.LabelRole, Roles.FlagRole])
                return


class FilmstripDelegate(QStyledItemDelegate):
    def __init__(self, get_size_callable, parent=None):
        super().__init__(parent)
        self._get_size = get_size_callable
        self._label_font = QFont()
        self._label_font.setPointSize(8)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        r = option.rect
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # 필름 스트립은 라이트 모드에서도 다크 테마와 동일 스타일(요청사항)
        bg_col = QColor("#2B2B2B")
        sel_col = QColor("#4DA3FF")
        painter.fillRect(r, bg_col)
        # 상태값 미리 추출
        try:
            rating = int(index.data(Roles.RatingRole) or 0)
        except Exception:
            rating = 0
        try:
            flag = str(index.data(Roles.FlagRole) or "unflagged")
        except Exception:
            flag = "unflagged"
        label = index.data(Roles.LabelRole)
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)

        # 썸네일 중앙 배치(파일명은 숨김). DPR을 고려해 DIP 기준으로 정렬
        pm = index.data(Roles.PixmapRole)
        if isinstance(pm, QPixmap) and not pm.isNull():
            try:
                dpr = float(getattr(pm, 'devicePixelRatio', lambda: 1.0)())
            except Exception:
                dpr = 1.0
            pw, ph = pm.width(), pm.height()
            dpw = max(1, int(round(pw / max(1.0, dpr))))
            dph = max(1, int(round(ph / max(1.0, dpr))))
            x = r.x() + (r.width() - dpw) // 2
            y = r.y() + (r.height() - dph) // 2
            if flag == "rejected" and not is_selected:
                painter.save()
                painter.setOpacity(0.35)
                painter.drawPixmap(x, y, pm)
                painter.restore()
            else:
                painter.drawPixmap(x, y, pm)
        else:
            # 픽스맵이 아직 없을 때(로딩 전)도 테마 배경이 보이도록 칠해 둠
            painter.fillRect(r.adjusted(6, 6, -6, -6), bg_col)
        # 썸네일 사이 경계선(미세 구분선)
        try:
            sep_col = QColor("#3A3A3A")
            painter.setPen(QPen(sep_col, 1))
            painter.drawLine(r.topRight(), r.bottomRight())
        except Exception:
            pass
        if option.state & QStyle.StateFlag.State_Selected:
            pen = QPen(sel_col, 3)
            painter.setPen(pen)
            painter.drawRoundedRect(r.adjusted(2, 2, -2, -2), 4, 4)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        # 셀 크기는 현재 썸네일 단계 + 여백(좌우 12px, 상하 14px)
        size = int(self._get_size())
        return QSize(size + 24, size + 28)


class FilmstripView(QListView):
    currentIndexChanged = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # 기본: 최대 썸네일 단계 사용 (가장 크게 보이도록)
        self._size_idx = len(THUMB_STEPS) - 1
        # 품질/경로는 부모 뷰어 설정을 반영
        try:
            parent_q = int(getattr(parent, "_thumb_cache_quality", 85)) if parent is not None else 85
        except Exception:
            parent_q = 85
        try:
            parent_root = str(getattr(parent, "_thumb_cache_dir", "")) if parent is not None else ""
        except Exception:
            parent_root = ""
        self._cache = ThumbCache(quality=parent_q, root_dir=(parent_root or None))
        self._pool = QThreadPool.globalInstance()
        self._model = FilmstripModel(self._cache, self._pool, self._target_size, self._current_dpr)
        self._delegate = FilmstripDelegate(self._target_size, self)
        self.setModel(self._model)
        self.setItemDelegate(self._delegate)

        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setMovement(QListView.Movement.Static)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setWrapping(False)
        # 라이트 모드에서도 다크 배경 경계처럼 보이도록 간격을 0으로(요청) 
        self.setSpacing(0)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        # 초기 테마 적용은 부모 뷰어의 _resolved_theme를 참조하여 동적으로 설정됨
        try:
            self.setStyleSheet("QListView, QListView::viewport { background-color: #1F1F1F; }")
        except Exception:
            self.setStyleSheet("QListView, QListView::viewport { background-color: #1F1F1F; }")

        try:
            self.selectionModel().selectionChanged.connect(self._on_selection_changed)
        except Exception:
            pass

        self._update_fixed_height()

        # 접근성: 포커스 가능 및 스크린리더 라벨
        try:
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setAccessibleName("필름 스트립")
            self.setAccessibleDescription("현재 폴더의 이미지 썸네일을 가로로 나열합니다.")
        except Exception:
            pass

        # 내부 선택 변경 신호 억제 가드 (프로그램적 갱신 루프 방지)
        self._suppress_signal = False

    def _target_size(self) -> int:
        return THUMB_STEPS[self._size_idx]

    def _current_dpr(self) -> float:
        try:
            return float(self.viewport().devicePixelRatioF())
        except Exception:
            return 1.0

    # 외부에서 메타 변경 시 갱신
    def update_item_meta_by_path(self, path: str, rating=None, label=None, flag=None) -> None:
        try:
            self._model.update_item_meta_by_path(path, rating=rating, label=label, flag=flag)
        except Exception:
            pass

    def _update_fixed_height(self):
        h = self._target_size() + 28 + 4
        try:
            # 자유 조절 허용: 최대 높이만 제한 (썸네일 최대 높이 초과 금지)
            self.setMaximumHeight(h)
            # 하한은 최소 UI 가독성 유지 수준으로 설정
            self.setMinimumHeight(48)
        except Exception:
            self.setFixedHeight(h)

    def adjust_thumbnail_step_for_height(self, container_height: int) -> None:
        try:
            avail = max(1, int(container_height) - (28 + 4))
            best_idx = 0
            for i in range(len(THUMB_STEPS)):
                if THUMB_STEPS[i] + 2 <= avail:
                    best_idx = i
            if best_idx != self._size_idx:
                self._size_idx = best_idx
                self._update_fixed_height()
                try:
                    self.viewport().update()
                except Exception:
                    pass
        except Exception:
            pass

    def apply_theme(self, is_light: bool) -> None:
        try:
            # 요청: 라이트 테마에서도 스크롤/배경은 다크 테마와 동일 계열 유지
            bg = "#1F1F1F"
            self.setStyleSheet(
                f"QListView, QListView::viewport {{ background-color: {bg}; }}"
                f" QScrollBar:horizontal {{ background: #2B2B2B; height: 12px; }}"
                f" QScrollBar::handle:horizontal {{ background: #555; min-width: 24px; border-radius: 6px; }}"
                f" QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ background: transparent; width: 0px; }}"
                f" QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: #2B2B2B; }}"
            )
            try:
                from PyQt6.QtGui import QPalette, QColor  # type: ignore
                pal = self.viewport().palette()
                pal.setColor(QPalette.ColorRole.Base, QColor(bg))
                pal.setColor(QPalette.ColorRole.Window, QColor(bg))
                self.viewport().setPalette(pal)
                self.viewport().setAutoFillBackground(True)
            except Exception:
                pass
            self.viewport().update()
        except Exception:
            pass

    def set_items(self, paths: List[str], current_index: int = -1):
        self._model.set_items(paths, current_index)
        try:
            if 0 <= current_index < self._model.rowCount():
                self._suppress_signal = True
                try:
                    self.setCurrentIndex(self._model.index(current_index, 0))
                    # 기본적으로 가운데 정렬로 스크롤하지 않음
                finally:
                    self._suppress_signal = False
        except Exception:
            pass
        # 현재 인덱스 주변(±50)만 메타/썸네일 미스시 빠르게 보완되도록 뷰를 강제 업데이트
        try:
            self.viewport().update()
        except Exception:
            pass

    def set_current_index(self, row: int):
        if 0 <= row < self._model.rowCount():
            self._suppress_signal = True
            try:
                self.setCurrentIndex(self._model.index(row, 0))
                # 기본적으로 가운데 정렬 스크롤을 하지 않음
            finally:
                self._suppress_signal = False

    def _on_selection_changed(self, selected, deselected):
        if self._suppress_signal:
            return
        idx = self.currentIndex()
        if idx.isValid():
            self.currentIndexChanged.emit(idx.row())

    def wheelEvent(self, event):
        mods = QApplication.keyboardModifiers()
        dy = int(event.angleDelta().y())
        dx = int(event.angleDelta().x())
        if mods & Qt.KeyboardModifier.ControlModifier:
            if dy > 0 or dx > 0:
                self._size_idx = min(self._size_idx + 1, len(THUMB_STEPS) - 1)
            elif dy < 0 or dx < 0:
                self._size_idx = max(self._size_idx - 1, 0)
            self._update_fixed_height()
            self.viewport().update()
            event.accept()
            return
        # 기본: 수평 스크롤(Shift 없이도 적용). 트랙패드/마우스 모두 지원
        try:
            delta = dx if abs(dx) >= abs(dy) else dy
            if delta:
                sb = self.horizontalScrollBar()
                sb.setValue(sb.value() - delta)
                event.accept()
                return
        except Exception:
            pass
        # 폴백: 기본 처리
        super().wheelEvent(event)

    def keyPressEvent(self, event):
        # Alt+숫자: 평점(델리게이트/모델 확장 여지 보장)
        try:
            if event.modifiers() & Qt.KeyboardModifier.AltModifier:
                key = event.key()
                if Qt.Key.Key_0 <= key <= Qt.Key.Key_5:
                    # TODO: rating 적용 지점(메타 저장소와 연계 시 구현)
                    event.accept()
                    return
        except Exception:
            pass
        super().keyPressEvent(event)


