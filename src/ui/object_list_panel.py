from __future__ import annotations

from typing import List, Dict, Tuple, Optional

from PyQt6.QtCore import pyqtSignal, Qt, QSize  # type: ignore[import]
from PyQt6.QtGui import QIcon, QPixmap, QImage  # type: ignore[import]
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QWidget, QLabel, QHBoxLayout  # type: ignore[import]

from PIL import Image  # type: ignore
import numpy as np  # type: ignore


def _crop_thumbnail(image_path: str, bbox: Tuple[int, int, int, int], max_size: int = 128) -> QPixmap:
    # bbox: (x1, y1, x2, y2) in original image coordinates
    with Image.open(image_path) as im:
        im = im.convert("RGBA")
        x1, y1, x2, y2 = bbox
        # Clamp bbox to image bounds
        w, h = im.size
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        crop = im.crop((x1, y1, x2, y2))
        # Resize preserving aspect ratio
        crop.thumbnail((max_size, max_size), resample=Image.LANCZOS)
        arr = np.array(crop)  # RGBA
        hh, ww, _ = arr.shape
        # bytesPerLine must be width * 4; ensure deep copy for safety
        qimg = QImage(arr.data, ww, hh, ww * 4, QImage.Format.Format_RGBA8888).copy()
        pm = QPixmap.fromImage(qimg)
        return pm


class _ItemWidget(QWidget):
    def __init__(self, pm: QPixmap, text: str, thumb_size: int, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        try:
            lay.setContentsMargins(6, 4, 6, 4)
            lay.setSpacing(10)
        except Exception:
            pass
        self.img = QLabel(self)
        try:
            self.img.setFixedSize(int(thumb_size), int(thumb_size))
        except Exception:
            pass
        try:
            pm2 = pm.scaled(thumb_size, thumb_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        except Exception:
            pm2 = pm
        self.img.setPixmap(pm2)
        self.lbl = QLabel(text, self)
        # 텍스트: 항상 하얀색, 줄바꿈 없이 표시
        try:
            self.lbl.setStyleSheet("QLabel { color: #EAEAEA; }")
            self.lbl.setWordWrap(False)
            self.lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        except Exception:
            pass
        lay.addWidget(self.img, 0)
        lay.addWidget(self.lbl, 1)


class ObjectListPanel(QListWidget):
    """
    Left-side panel listing detected objects as icon + text.
    Emits objectSelected(dict) with keys: bbox, label, score, index.
    """

    objectSelected = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            self.setIconSize(self.iconSize())  # keep default but allow scaling by style
            self.setUniformItemSizes(True)
            self.setAlternatingRowColors(True)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        except Exception:
            pass
        # 다크 테마에 맞춘 텍스트/배경 스타일
        try:
            self.setStyleSheet(
                "QListWidget { color: #EAEAEA; background-color: #2B2B2B; border: 1px solid #444; } "
                "QListWidget::item { color: #EAEAEA; } "
                "QListWidget::item:hover { background-color: #2F3337; } "
                "QListWidget::item:selected:active { background-color: #3A3F44; color: #FFFFFF; } "
                "QListWidget::item:selected:!active { background-color: #3A3F44; color: #FFFFFF; }"
            )
        except Exception:
            pass
        self._items_meta: List[dict] = []
        # Selection behavior
        try:
            self.currentItemChanged.connect(self._on_current_changed)
        except Exception:
            pass

    def resizeEvent(self, event):
        # 뷰포트 폭 변화 시 각 행의 가로 크기를 뷰포트 폭에 맞춤
        try:
            row_w = max(1, int(self.viewport().width()))
            for i in range(self.count()):
                it = self.item(i)
                if it is not None:
                    sz = it.sizeHint()
                    it.setSizeHint(QSize(row_w, sz.height()))
        except Exception:
            pass
        return super().resizeEvent(event)

    def clear_items(self) -> None:
        self.clear()
        self._items_meta = []

    def populate(self, image_path: str, dets: List[Dict], thumb_size: int = 96) -> None:
        """
        dets: list of {"bbox": (x1,y1,x2,y2), "label": str, "score": float}
        """
        self.clear_items()
        if not dets:
            it = QListWidgetItem("검출된 객체가 없습니다.")
            self.addItem(it)
            return
        try:
            self.setIconSize(QSize(int(thumb_size), int(thumb_size)))
        except Exception:
            pass
        for idx, d in enumerate(dets):
            try:
                pm = _crop_thumbnail(image_path, tuple(d["bbox"]), max_size=thumb_size)
            except Exception:
                pm = QPixmap()
            label = str(d.get("label") or "")
            try:
                score = float(d.get("score", 0.0))
            except Exception:
                score = 0.0
            text = f"{label} ({score:.2f})"
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, idx)
            # 고정 아이템 높이/레이아웃
            try:
                row_w = max(1, int(self.viewport().width()))
                item.setSizeHint(QSize(row_w, int(thumb_size + 8)))  # 폭=뷰포트 폭, 높이 고정
            except Exception:
                pass
            self.addItem(item)
            try:
                widget = _ItemWidget(pm, text, thumb_size, self)
                self.setItemWidget(item, widget)
            except Exception:
                pass
            self._items_meta.append({"index": idx, **d})

    def _on_current_changed(self, cur: Optional[QListWidgetItem], _prev: Optional[QListWidgetItem]) -> None:
        if cur is None:
            return
        try:
            idx = int(cur.data(Qt.ItemDataRole.UserRole))
        except Exception:
            idx = -1
        meta: Optional[dict] = None
        if 0 <= idx < len(self._items_meta):
            meta = self._items_meta[idx]
        if meta is not None:
            self.objectSelected.emit(meta)


