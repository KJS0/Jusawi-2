import numpy as np
from PyQt6.QtGui import QImageReader, QPixmap, QImage  # type: ignore[import]
from skimage.metrics import structural_similarity as ssim
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QFileDialog, QSplitter, QVBoxLayout, QLabel, QMessageBox # type: ignore[import]
from PyQt6.QtCore import Qt  # type: ignore[import]

def load_gray_qimage(path: str) -> QImage:
    return QImageReader(path).read().convertToFormat(QImage.Format.Format_Grayscale8)

def qimage_gray_to_ndarray(gray: QImage) -> np.ndarray:
    w, h = int(gray.width()), int(gray.height())
    ptr = gray.bits()
    ptr.setsize(gray.sizeInBytes())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, gray.bytesPerLine())
    return arr[:, :w]

class ComparsionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("사진 비교")
        self.setGeometry(100, 100, 800, 600)
        self.show()

        top = QHBoxLayout()
        self.btn_open = QPushButton("열기")
        self.btn_open.clicked.connect(self.open_file)
        top.addWidget(self.btn_open)

        self.left_view = QLabel()
        self.right_view = QLabel()

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.addWidget(self.left_view)
        self.splitter.addWidget(self.right_view)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(self.splitter)

        self.text = QLabel()
        root.addWidget(self.text)

    def image_filter(self) -> str:
        return "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff;*.tif;*.webp"

    def load_image(self, view: QLabel, path: str) -> None:
        pm = QPixmap.fromImage(QImage(path))
        pm = pm.scaled(view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        view.setPixmap(pm)

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileNames(self, "열기", "", self.image_filter())
        if path and len(path) == 2:
            self.load_image(self.left_view, path[0])
            self.load_image(self.right_view, path[1])
            self.compute_similarity(path[0], path[1])
        else:
            QMessageBox.warning(self, "열기", "이미지를 2개 선택해주세요.")

    def open_comparsion_dialog(self) -> None:
        from .comparsion_dialog import ComparsionDialog  # type: ignore
        dlg = ComparsionDialog(self)
        dlg.show()
        return dlg

    def compute_similarity(self, path_a: str, path_b: str) -> None:
        qa = load_gray_qimage(path_a)
        qb = load_gray_qimage(path_b)
        A = qimage_gray_to_ndarray(qa)
        B = qimage_gray_to_ndarray(qb)
        h = min(A.shape[0], B.shape[0])
        w = min(A.shape[1], B.shape[1])
        A = A[:h, :w]
        B = B[:h, :w]
        score = ssim(A, B, data_range=255)
        self.text.setText(f"유사도 : {float(score):.3f}")

    def closeEvent(self, event) -> None:
        event.accept()

    def close(self) -> None:
        self.closeEvent(None)