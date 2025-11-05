import os
from typing import Tuple, List, Callable, Optional
from collections import OrderedDict
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt, QRunnable, QThreadPool  # type: ignore[import]
from PyQt6.QtGui import QImage, QImageReader, QTransform, QColorSpace  # type: ignore[import]

from ..utils.file_utils import scan_directory_util, safe_write_bytes
from .metadata_service import extract_metadata, encode_with_metadata
from .save_service import SaveService
from ..utils.logging_setup import get_logger

_log = get_logger("svc.ImageService")


_DECODE_MEMORY_CAP_MB_DEFAULT = 512
_DECODE_MEMORY_CAP_MB = _DECODE_MEMORY_CAP_MB_DEFAULT

def _apply_safe_scaled_size_for_reader(reader: QImageReader, memory_cap_mb: int) -> None:
    """원본 해상도가 주어진 메모리 상한을 초과할 경우, 안전한 스케일로 디코딩하도록 설정한다.
    - 상한은 대략 RGBA 4바이트/픽셀을 기준으로 추정한다.
    - reader.size()가 유효하지 않거나 계산에 실패하면 아무 것도 하지 않는다.
    """
    try:
        cap_mb = int(memory_cap_mb)
    except Exception:
        cap_mb = _DECODE_MEMORY_CAP_MB_DEFAULT
    try:
        osize = reader.size()
        width = int(osize.width())
        height = int(osize.height())
    except Exception:
        width = 0
        height = 0
    if width <= 0 or height <= 0:
        return
    try:
        cap_bytes = max(1, cap_mb) * 1024 * 1024
        # 대략 RGBA 4B/px 기준으로 추정
        estimated_bytes = int(width) * int(height) * 4
        if estimated_bytes > cap_bytes:
            from PyQt6.QtCore import QSize  # type: ignore[import]
            # 면적 비례하므로 제곱근 비율로 축소
            import math
            scale = math.sqrt(cap_bytes / float(max(1, estimated_bytes)))
            new_w = max(1, int(width * scale))
            new_h = max(1, int(height * scale))
            reader.setScaledSize(QSize(new_w, new_h))
    except Exception:
        pass

class _ImageWorker(QObject):
    done = pyqtSignal(str, QImage, bool, str)

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def run(self):
        try:
            img, ok, err = _read_qimage_with_exif_auto_transform(self._path)
            if not ok:
                self.done.emit(self._path, QImage(), False, err)
                return
            self.done.emit(self._path, img, True, "")
        except Exception as e:
            self.done.emit(self._path, QImage(), False, str(e))


class _CancellationToken(QObject):
    def __init__(self):
        super().__init__()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


class _SaveWorker(QObject):
    progress = pyqtSignal(int)
    done = pyqtSignal(bool, str)

    def __init__(self, img: QImage, src_path: str, dest_path: str, rotation_degrees: int,
                 flip_horizontal: bool, flip_vertical: bool, quality: int, token: _CancellationToken):
        super().__init__()
        self._img = img
        self._src = src_path
        self._dst = dest_path
        self._rot = rotation_degrees
        self._fh = flip_horizontal
        self._fv = flip_vertical
        self._q = quality
        self._token = token

    def run(self):
        # 단계적 진행률 신호: 0->20 변환, 20->80 인코딩, 80->100 저장
        try:
            if self._token.is_cancelled():
                self.done.emit(False, "작업이 취소되었습니다.")
                return
            self.progress.emit(10)
            rot = _normalize_rotation(self._rot)
            q = _sanitize_quality(self._q)
            transformed = _apply_transform(self._img, rot, bool(self._fh), bool(self._fv))

            if self._token.is_cancelled():
                self.done.emit(False, "작업이 취소되었습니다.")
                return
            self.progress.emit(30)

            from PIL import Image as PILImage  # type: ignore
            fmt = _guess_format_from_path(self._dst) or 'JPEG'
            qt_format = QImage.Format.Format_RGBA8888
            converted = transformed if transformed.format() == qt_format else transformed.convertToFormat(qt_format)
            width = converted.width()
            height = converted.height()
            ptr = converted.bits()
            ptr.setsize(converted.sizeInBytes())
            raw = bytes(ptr)
            pil_image = PILImage.frombytes('RGBA', (width, height), raw)
            if fmt.upper() == 'JPEG':
                pil_image = pil_image.convert('RGB')

            if self._token.is_cancelled():
                self.done.emit(False, "작업이 취소되었습니다.")
                return
            self.progress.emit(60)

            meta = extract_metadata(self._src)
            ok, encoded_bytes, err = encode_with_metadata(pil_image, fmt, q, meta)
            if not ok:
                self.done.emit(False, err or "인코딩 실패")
                return

            if self._token.is_cancelled():
                self.done.emit(False, "작업이 취소되었습니다.")
                return
            self.progress.emit(85)

            ok2, err2 = safe_write_bytes(self._dst, encoded_bytes, write_through=True, retries=6)
            if not ok2:
                self.done.emit(False, err2 or "원자적 저장 실패")
                return
            self.progress.emit(100)
            self.done.emit(True, "")
        except Exception as e:
            self.done.emit(False, str(e))
        finally:
            try:
                _ok = 'e' not in locals()
                _log.info("save_worker_done | dst=%s | ok=%s", os.path.basename(self._dst), _ok)
            except Exception:
                pass


class ImageService(QObject):
    loaded = pyqtSignal(str, QImage, bool, str)  # path, img, success, error

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _ImageWorker | None = None
        # 간단한 LRU QImage 캐시 (용량 제한: 바이트 단위)
        self._img_cache = _QImageCache(max_bytes=256 * 1024 * 1024)  # 기본 256MB
        # 스케일별 다운샘플 QImage 캐시 (원본과 분리, 약간 더 여유)
        self._scaled_cache = _QImageCache(max_bytes=384 * 1024 * 1024)
        # 프리로드용 스레드풀 및 세대 토큰
        self._pool = QThreadPool.globalInstance()
        self._preload_generation = 0
        # 프리로드 정책
        self._preload_max_concurrency = 0
        self._preload_retry_count = 0
        self._preload_retry_delay_ms = 0
        # 애니메이션 정보 캐시: path -> frame_count(>1이면 애니메이션), -1 미상/계산 실패
        self._anim_frame_count: dict[str, int] = {}
        # 저장 위임 서비스
        self._save_service = SaveService()
        # 색상 관리 정책(뷰어 설정/단축키로 제어)
        self._color_view_mode = 'managed'  # 'managed' | 'original'
        self._icc_ignore_embedded = False
        self._assumed_colorspace = 'sRGB'  # ICC 미탑재/무시 시 가정 색공간
        self._preview_target = 'sRGB'      # sRGB | Display P3 | Adobe RGB
        self._fallback_policy = 'ignore'   # 'warn' | 'ignore' | 'force_sRGB'
        # 디코드 메모리 상한(안전 스케일 기준), 전역에도 반영
        try:
            self._decode_memory_cap_mb = int(getattr(self, "_decode_memory_cap_mb", _DECODE_MEMORY_CAP_MB_DEFAULT))
        except Exception:
            self._decode_memory_cap_mb = _DECODE_MEMORY_CAP_MB_DEFAULT
        try:
            global _DECODE_MEMORY_CAP_MB
            _DECODE_MEMORY_CAP_MB = int(self._decode_memory_cap_mb)
        except Exception:
            pass
        # Qt 기본 256MB 할당 제한 완화(가능한 환경에서만)
        try:
            alloc_limit_mb = int(getattr(self, "_qimage_alloc_limit_mb", max(1024, _DECODE_MEMORY_CAP_MB)))
            if hasattr(QImageReader, 'setAllocationLimit'):
                QImageReader.setAllocationLimit(alloc_limit_mb)
        except Exception:
            pass

    def set_cache_limits(self, image_cache_max_bytes: int | None = None, scaled_cache_max_bytes: int | None = None) -> None:
        try:
            if isinstance(image_cache_max_bytes, int) and image_cache_max_bytes > 0:
                self._img_cache._max_bytes = int(image_cache_max_bytes)
                self._img_cache._evict_if_needed()
        except Exception:
            pass
        try:
            if isinstance(scaled_cache_max_bytes, int) and scaled_cache_max_bytes > 0:
                self._scaled_cache._max_bytes = int(scaled_cache_max_bytes)
                self._scaled_cache._evict_if_needed()
        except Exception:
            pass

    def scan_directory(self, dir_path: str, current_image_path: str | None):
        image_files, cur_idx = scan_directory_util(dir_path, current_image_path)
        # 정렬 정책 고정: Windows 탐색기식(자연 정렬) 파일명 기준
        try:
            # QObject의 parent() 메서드를 통해 부모 뷰어를 가져온다.
            try:
                owner = self.parent()
            except Exception:
                owner = None
        except Exception:
            owner = None
        try:
            # 숨김/시스템 파일은 항상 제외 (설정 무시, 보안/일관성 강제)
            def _is_hidden(p: str) -> bool:
                try:
                    name = os.path.basename(p)
                    if name.startswith('.'):
                        return True
                    if os.name == 'nt':
                        try:
                            import ctypes
                            FILE_ATTRIBUTE_HIDDEN = 0x2
                            FILE_ATTRIBUTE_SYSTEM = 0x4
                            attrs = ctypes.windll.kernel32.GetFileAttributesW(ctypes.c_wchar_p(p))
                            if attrs != -1 and (attrs & (FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM)):
                                return True
                        except Exception:
                            return False
                    return False
                except Exception:
                    return False
            image_files = [p for p in image_files if not _is_hidden(p)]
            # 정렬 방식 고정: 파일명(자연 정렬)
            try:
                base_names = [(os.path.basename(p), p) for p in image_files]
                from ..utils.file_utils import windows_style_sort_key
                import functools
                base_names.sort(key=functools.cmp_to_key(lambda a,b: windows_style_sort_key(a[0], b[0])))
                image_files = [p for (_, p) in base_names]
            except Exception:
                try:
                    image_files = sorted(image_files, key=lambda p: os.path.basename(p).lower())
                except Exception:
                    pass
            # 현재 인덱스 재계산
            try:
                if current_image_path and image_files:
                    nc = os.path.normcase
                    try:
                        cur_idx = [nc(p) for p in image_files].index(nc(current_image_path))
                    except ValueError:
                        cur_idx = 0 if image_files else -1
            except Exception:
                pass
        except Exception:
            pass
        return image_files, cur_idx

    def load(self, path: str) -> Tuple[str, QImage | None, bool, str]:
        # 캐시 히트 시 즉시 반환
        cached = self._img_cache.get(path)
        if cached is not None and not cached.isNull():
            try:
                _log.info("load_cache_hit | file=%s | w=%d | h=%d", os.path.basename(path), int(cached.width()), int(cached.height()))
            except Exception:
                pass
            return path, cached, True, ""
        try:
            img, ok, err = self._read_qimage_with_exif_auto_transform_with_policy(path)
            if not ok:
                try:
                    _log.error("load_decode_fail | file=%s | err=%s", os.path.basename(path), err or "")
                except Exception:
                    pass
                return path, None, False, err
            self._img_cache.put(path, img)
            try:
                _log.info("load_decode_ok | file=%s | w=%d | h=%d", os.path.basename(path), int(img.width()), int(img.height()))
            except Exception:
                pass
            return path, img, True, ""
        except Exception as e:
            try:
                _log.exception("load_exception | file=%s | err=%s", os.path.basename(path), str(e))
            except Exception:
                pass
            return path, None, False, str(e)

    # --- 스케일 캐시 API -------------------------------------------------
    def _quantize_scale(self, s: float) -> float:
        try:
            sf = float(s)
        except Exception:
            return 1.0
        if sf >= 1.0:
            return 1.0
        # 대표 스냅 단계(줌 UX와 유사한 단계)
        steps = [
            0.0625, 0.0833, 0.1, 0.125, 0.167, 0.2,
            0.25, 0.333, 0.4, 0.5, 0.667, 0.8, 1.0,
        ]
        # 가장 가까운 단계 선택
        nearest = min(steps, key=lambda v: abs(v - sf))
        return nearest

    def get_scaled_image(self, path: str, scale: float, dpr: float = 1.0) -> Optional[QImage]:
        """원본 QImage를 기준으로 주어진 스케일(<=1.0)과 DPR로 다운샘플된 QImage를 반환.
        - 캐시 키: path|s=<qscale*1000>|dpr=<dpr*100>
        - 업스케일은 수행하지 않음(요청 스케일이 1.0 이상이면 None 반환하여 원본 사용).
        """
        try:
            if not path:
                return None
            try:
                s = float(scale)
            except Exception:
                s = 1.0
            if s >= 1.0:
                return None
            qscale = self._quantize_scale(max(0.01, s))
            try:
                dprf = float(dpr)
            except Exception:
                dprf = 1.0
            # 키 구성
            key = f"{path}|s={int(round(qscale * 1000))}|dpr={int(round(dprf * 100))}"
            cached = self._scaled_cache.get(key)
            if cached is not None and not cached.isNull():
                return cached
            # 1) 원본이 이미 캐시에 있으면 빠른 경로: 메모리 내에서 스케일
            base = self._img_cache.get(path)
            if base is not None and not base.isNull():
                bw = int(round(base.width() * qscale * dprf))
                bh = int(round(base.height() * qscale * dprf))
                if bw < 1 or bh < 1:
                    return None
                scaled = base.scaled(bw, bh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                if scaled.isNull():
                    return None
                self._scaled_cache.put(key, scaled)
                return scaled

            # 2) 원본 미보유: QImageReader 스케일 디코딩 경로로 풀해상도 디코드 회피
            try:
                reader = QImageReader(path)
                reader.setAutoTransform(True)
                # 원본(오토트랜스폼 반영) 해상도 조회
                try:
                    orig_size = reader.size()
                    ow = int(getattr(orig_size, 'width')() if hasattr(orig_size, 'width') else int(orig_size.width()))
                    oh = int(getattr(orig_size, 'height')() if hasattr(orig_size, 'height') else int(orig_size.height()))
                except Exception:
                    ow = 0
                    oh = 0
                if ow <= 0 or oh <= 0:
                    # 사이즈 조회 실패 시 마지막 폴백: 직접 읽고 크기 확인(여전히 스케일 디코드 시도)
                    # 다만 이 경우 풀해상도 잠재 비용이 발생할 수 있음
                    pass
                # 목표 크기 산출(비율 유지)
                bw = max(1, int(round(max(1.0, ow) * qscale * dprf))) if ow > 0 else 0
                bh = max(1, int(round(max(1.0, oh) * qscale * dprf))) if oh > 0 else 0
                if bw > 0 and bh > 0:
                    from PyQt6.QtCore import QSize  # type: ignore[import]
                    reader.setScaledSize(QSize(bw, bh))
                img = reader.read()
                if img.isNull():
                    return None
                try:
                    img = self._convert_image_for_display(img)  # type: ignore[attr-defined]
                except Exception:
                    img = _convert_to_srgb(img)
                self._scaled_cache.put(key, img)
                return img
            except Exception:
                return None
        except Exception:
            return None

    def get_scaled_for_viewport(self, path: str, viewport_width: int, viewport_height: int,
                                 view_mode: str = "fit", dpr: float = 1.0, headroom: float = 1.0) -> Optional[QImage]:
        """뷰포트/DPR/뷰모드 기준으로 목표 크기를 산출해 QImageReader 스케일 디코딩.
        - view_mode: 'fit' | 'fit_width' | 'fit_height'
        - headroom: 1.0 이상이면 약간 크게 읽어 미세 줌인 시 재디코드 빈도 감소
        - 캐시 키는 내부적으로 s(비율)과 dpr을 조합해 재사용
        """
        try:
            if not path:
                return None
            vw = max(1, int(viewport_width))
            vh = max(1, int(viewport_height))
            try:
                dprf = float(dpr)
            except Exception:
                dprf = 1.0
            reader = QImageReader(path)
            reader.setAutoTransform(True)
            try:
                osize = reader.size()
                ow = int(osize.width())
                oh = int(osize.height())
            except Exception:
                ow = 0
                oh = 0
            if ow <= 0 or oh <= 0:
                return None
            # 원하는 화면 픽셀(장치 픽셀 기준)
            target_w_px = vw * dprf
            target_h_px = vh * dprf
            # 모드별 스케일 산출(오토트랜스폼 반영된 ow/oh 기준)
            if view_mode == 'fit_width':
                s = target_w_px / float(ow)
            elif view_mode == 'fit_height':
                s = target_h_px / float(oh)
            else:
                s = min(target_w_px / float(ow), target_h_px / float(oh))
            # 업스케일 금지
            s = max(0.01, min(1.0, s))
            # 헤드룸
            try:
                hr = float(headroom)
            except Exception:
                hr = 1.0
            if hr < 1.0:
                hr = 1.0
            s = min(1.0, s * hr)
            qscale = self._quantize_scale(s)
            # 캐시 키 구성
            key = f"{path}|s={int(round(qscale * 1000))}|dpr={int(round(dprf * 100))}"
            cached = self._scaled_cache.get(key)
            if cached is not None and not cached.isNull():
                return cached
            # 목표 크기
            from PyQt6.QtCore import QSize  # type: ignore[import]
            bw = max(1, int(round(ow * qscale * dprf)))
            bh = max(1, int(round(oh * qscale * dprf)))
            reader.setScaledSize(QSize(bw, bh))
            img = reader.read()
            if img.isNull():
                return None
            try:
                img = self._convert_image_for_display(img)  # type: ignore[attr-defined]
            except Exception:
                img = _convert_to_srgb(img)
            self._scaled_cache.put(key, img)
            return img
        except Exception:
            return None

    # 지능형 스케일 프리젠: 선호 배율 세트를 미리 생성하여 스크롤/줌 지연 감소
    def pregen_preferred_scales(self, path: str, base_viewport_w: int, base_viewport_h: int, dpr: float,
                                preferred_scales: list[float], view_mode: str = "fit") -> None:
        if not path or not os.path.isfile(path):
            return
        try:
            for s in preferred_scales or []:
                try:
                    scale = float(s)
                except Exception:
                    continue
                if scale <= 0:
                    continue
                w = max(1, int(round(base_viewport_w * scale)))
                h = max(1, int(round(base_viewport_h * scale)))
                try:
                    _ = self.get_scaled_for_viewport(path, w, h, view_mode=view_mode, dpr=dpr, headroom=1.0)
                except Exception:
                    pass
        except Exception:
            pass

    def load_async(self, path: str) -> None:
        # 이전 작업 취소/정리
        if self._thread:
            try:
                self._thread.quit()
                # 충분히 대기하여 안전 종료 유도
                self._thread.wait(2000)
            except Exception:
                pass
            finally:
                # 남아있으면 강제 종료 (최후 수단)
                try:
                    if self._thread.isRunning():
                        self._thread.terminate()
                        self._thread.wait(1000)
                except Exception:
                    pass
                self._cleanup_thread()

        self._thread = QThread()
        self._worker = _ImageWorker(path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_worker_done)
        self._worker.done.connect(lambda *_: self._thread.quit())
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _on_worker_done(self, path: str, img: QImage, success: bool, error: str):
        self.loaded.emit(path, img, success, error)

    def _on_thread_finished(self):
        self._cleanup_thread()

    def _cleanup_thread(self):
        try:
            if self._worker is not None:
                self._worker.deleteLater()
        except Exception:
            pass
        try:
            if self._thread is not None:
                self._thread.deleteLater()
        except Exception:
            pass
        self._worker = None
        self._thread = None

    def shutdown(self):
        # 서비스 종료 시 안전하게 스레드 중지
        if self._thread:
            try:
                self._thread.quit()
                self._thread.wait(2000)
            except Exception:
                pass
            finally:
                try:
                    if self._thread.isRunning():
                        self._thread.terminate()
                        self._thread.wait(1000)
                except Exception:
                    pass
                self._cleanup_thread()
        # 저장 스레드 종료
        try:
            self.cancel_save()
        except Exception:
            pass

    def clear_all_caches(self) -> None:
        """이미지/스케일/애니메이션 관련 모든 캐시를 초기화한다."""
        try:
            if hasattr(self, "_img_cache") and self._img_cache is not None:
                self._img_cache.clear()
        except Exception:
            pass
        try:
            if hasattr(self, "_scaled_cache") and self._scaled_cache is not None:
                self._scaled_cache.clear()
        except Exception:
            pass
        try:
            if isinstance(getattr(self, "_anim_frame_count", None), dict):
                self._anim_frame_count.clear()
        except Exception:
            pass
        try:
            # 프리로드 세대 초기화
            self._preload_generation = int(self._preload_generation) + 1
        except Exception:
            pass

    # --- 애니메이션/프레임 관련 유틸 ---
    def probe_animation(self, path: str) -> tuple[bool, int]:
        """파일이 애니메이션인지 여부와 프레임 수(-1: 미상)를 반환. 캐시 사용."""
        try:
            # 캐시 우선
            cached = self._anim_frame_count.get(path)
            if isinstance(cached, int):
                # 0: 비애니메이션(확정), >1: 프레임 수(애니메이션)
                if cached == 0:
                    return False, -1
                if cached > 1:
                    return True, cached
            reader = QImageReader(path)
            is_anim = False
            frame_count = -1
            # 지원 여부 확인
            try:
                if hasattr(reader, 'supportsAnimation'):
                    is_anim = bool(reader.supportsAnimation())
                    try:
                        _log.debug(f"anim_supports | file={os.path.basename(path)} | supports={is_anim}")
                    except Exception:
                        pass
            except Exception:
                pass
            # imageCount 시도(일부 포맷은 0 또는 1 반환)
            try:
                c = int(reader.imageCount())
                if c > 1:
                    frame_count = c
                    is_anim = True
                try:
                    _log.debug(f"anim_image_count | file={os.path.basename(path)} | count={c}")
                except Exception:
                    pass
            except Exception:
                pass
            # 확장자 힌트
            if not is_anim:
                ext = os.path.splitext(path)[1].lower()
                if ext in ('.gif', '.webp'):
                    is_anim = True
                    try:
                        _log.debug(f"anim_ext_hint | file={os.path.basename(path)} | ext={ext}")
                    except Exception:
                        pass
            # 필요한 경우 프레임 수 직접 계수(한 번만)
            if is_anim and (frame_count is None or frame_count <= 1):
                try:
                    # 처음부터 순회하여 카운트 추산
                    count_reader = QImageReader(path)
                    count_reader.setAutoTransform(True)
                    count = 0
                    # 첫 프레임 포함
                    if not count_reader.read().isNull():
                        count = 1
                        while count_reader.jumpToNextImage():
                            img = count_reader.read()
                            if img.isNull():
                                break
                            count += 1
                    frame_count = count if count > 1 else -1
                    try:
                        _log.debug(f"anim_count_scan | file={os.path.basename(path)} | frames={frame_count}")
                    except Exception:
                        pass
                except Exception:
                    frame_count = -1
            # 캐시 저장(비애니메이션은 0으로 표시하여 재탐색 억제)
            try:
                cache_val = int(frame_count) if (isinstance(frame_count, int) and frame_count > 1) else 0
            except Exception:
                cache_val = 0
            self._anim_frame_count[path] = cache_val
            try:
                _log.debug(f"anim_probe_result | file={os.path.basename(path)} | is_anim={is_anim} | frames={frame_count}")
            except Exception:
                pass
            if cache_val == 0:
                return False, -1
            return True, cache_val
        except Exception:
            return False, -1

    def load_frame(self, path: str, index: int) -> tuple[QImage | None, bool, str]:
        """지정 프레임을 로드(색관리 포함). index는 0 기반."""
        try:
            # 인덱스 래핑(알고 있는 경우)
            fc = self._anim_frame_count.get(path, -1)
            if isinstance(fc, int) and fc > 0:
                if index >= fc:
                    index = index % fc
                if index < 0:
                    index = (index % fc + fc) % fc
            reader = QImageReader(path)
            reader.setAutoTransform(True)
            # jumpToImage가 지원되면 직접 점프, 아니면 next로 순차 이동
            moved = False
            try:
                if hasattr(reader, 'jumpToImage'):
                    moved = bool(reader.jumpToImage(int(index)))
                else:
                    # 순차 이동 폴백
                    cur = 0
                    try:
                        cur = int(reader.currentImageNumber())
                    except Exception:
                        cur = 0
                    if index < cur:
                        # 처음부터 다시 시작하는 것이 안전
                        reader = QImageReader(path)
                        reader.setAutoTransform(True)
                        cur = 0
                    while cur < index:
                        if not reader.jumpToNextImage():
                            break
                        cur += 1
                    moved = (cur == index)
            except Exception:
                moved = False
            img = reader.read()
            if img.isNull():
                return None, False, reader.errorString() or "프레임을 불러올 수 없습니다."
            try:
                img = self._convert_image_for_display(img)  # type: ignore[attr-defined]
            except Exception:
                img = _convert_to_srgb(img)
            return img, True, ""
        except Exception as e:
            return None, False, str(e)

    # --- 프리로드/캐시 API ---
    def get_cached_image(self, path: str) -> Optional[QImage]:
        return self._img_cache.get(path)

    def invalidate_path(self, path: str) -> None:
        self._img_cache.delete(path)
        # 스케일 캐시도 같은 경로 접두사로 모두 무효화
        try:
            self._scaled_cache.delete_prefix(f"{path}|")
        except Exception:
            pass

    def preload(self, paths: List[str], priority: int = 0) -> None:
        """경로 목록을 백그라운드에서 디코드하여 이미지 캐시에 저장.
        취소는 세대 카운터 증가로 구현(새 호출 시 이전 작업은 효과적으로 무시됨).
        """
        if not paths:
            return
        # 세대 증가: 기존 작업은 결과가 와도 화면에 영향 없음. 캐시에 들어가도 용량 제한으로 관리됨.
        self._preload_generation += 1
        generation = self._preload_generation

        # 중복/이미 캐시된 항목은 제외
        unique: List[str] = []
        seen: set[str] = set()
        for p in paths:
            if p and p not in seen and self._img_cache.get(p) is None:
                seen.add(p)
                unique.append(p)
        if not unique:
            return

        # 동시 실행 상한 적용: 설정 값이 있으면 슬라이스 제한
        try:
            max_tasks = int(getattr(self, "_preload_max_concurrency", 0))
        except Exception:
            max_tasks = 0
        queue = unique[:max_tasks] if isinstance(max_tasks, int) and max_tasks > 0 else unique
        for p in queue:
            task = _PreloadTask(
                path=p,
                generation=generation,
                done=self._on_preload_done,
                retry_count=int(getattr(self, "_preload_retry_count", 0)),
                retry_delay_ms=int(getattr(self, "_preload_retry_delay_ms", 0)),
            )
            # 우선순위 힌트 사용
            try:
                self._pool.start(task, priority)
            except Exception:
                pass

    def _on_preload_done(self, path: str, img: QImage, success: bool, error: str, generation: int) -> None:
        # 최신 세대가 아니어도 캐시에 넣는 것은 허용(다음에 히트될 수 있음)
        if success and not img.isNull():
            self._img_cache.put(path, img)
        # 실패는 무시 (로그 없음)

    # --- 저장 비동기/취소 지원 ---
    def save_async(self,
                   img: QImage,
                   src_path: str,
                   dest_path: str,
                   rotation_degrees: int,
                   flip_horizontal: bool,
                   flip_vertical: bool,
                   quality: int,
                   on_progress: Callable[[int], None] | None,
                   on_done: Callable[[bool, str], None] | None) -> None:
        self._save_service.save_async(img, src_path, dest_path, rotation_degrees, flip_horizontal, flip_vertical, quality, on_progress, on_done)

    def cancel_save(self) -> None:
        self._save_service.cancel_save()

    def save_with_transform(self,
                            img: QImage,
                            src_path: str,
                            dest_path: str,
                            rotation_degrees: int,
                            flip_horizontal: bool,
                            flip_vertical: bool,
                            quality: int = 95) -> tuple[bool, str]:
        return self._save_service.save_with_transform(img, src_path, dest_path, rotation_degrees, flip_horizontal, flip_vertical, quality)

    # --- 색상 관리: 인스턴스 메서드 ---
    def _convert_image_for_display(self, img: QImage) -> QImage:
        try:
            return _apply_color_policy(self, img)
        except Exception:
            return _convert_to_srgb(img)

    def _read_qimage_with_exif_auto_transform_with_policy(self, path: str) -> tuple[QImage, bool, str]:
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        # 초대형 이미지에 대해 안전 스케일 적용
        try:
            _apply_safe_scaled_size_for_reader(reader, int(getattr(self, '_decode_memory_cap_mb', _DECODE_MEMORY_CAP_MB)))
        except Exception:
            pass
        img = reader.read()
        if img.isNull():
            try:
                _log.warning("qimage_read_null | file=%s | qerr=%s", os.path.basename(path), reader.errorString() or "")
            except Exception:
                pass
            return QImage(), False, reader.errorString() or "이미지를 불러올 수 없습니다."
        try:
            img2 = self._convert_image_for_display(img)
        except Exception:
            img2 = _convert_to_srgb(img)
        return img2, True, ""


def _read_qimage_with_exif_auto_transform(path: str) -> tuple[QImage, bool, str]:
    reader = QImageReader(path)
    # EXIF Orientation 등 자동 변환 활성화
    reader.setAutoTransform(True)
    # 초대형 이미지에 대해 안전 스케일 적용(프리로드 작업 등 공용 경로)
    try:
        _apply_safe_scaled_size_for_reader(reader, _DECODE_MEMORY_CAP_MB)
    except Exception:
        pass
    img = reader.read()
    if img.isNull():
        try:
            _log.warning("qimage_read_null | file=%s | qerr=%s", os.path.basename(path), reader.errorString() or "")
        except Exception:
            pass
        return QImage(), False, reader.errorString() or "이미지를 불러올 수 없습니다."
    img = _convert_to_srgb(img)
    return img, True, ""


def _read_qimage_with_exif_auto_transform_with_policy(self, path: str) -> tuple[QImage, bool, str]:
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    img = reader.read()
    if img.isNull():
        try:
            _log.warning("qimage_read_null | file=%s | qerr=%s", os.path.basename(path), reader.errorString() or "")
        except Exception:
            pass
        return QImage(), False, reader.errorString() or "이미지를 불러올 수 없습니다."
    try:
        img2 = _apply_color_policy(self, img)
    except Exception:
        img2 = _convert_to_srgb(img)
    return img2, True, ""


def _convert_to_srgb(img: QImage) -> QImage:
    """가능하면 sRGB로 변환하여 반환. 실패 시 원본 반환."""
    try:
        cs = img.colorSpace()
        needs_convert = False
        try:
            if cs.isValid():
                srgb = QColorSpace(QColorSpace.NamedColorSpace.SRgb)
                if cs != srgb:
                    needs_convert = True
        except Exception:
            needs_convert = False
        if needs_convert:
            converted = img.convertToColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
            if not converted.isNull():
                return converted
        return img
    except Exception:
        return img


def _named_colorspace_from_string(name: str) -> QColorSpace | None:
    try:
        n = (name or "").strip().lower()
        if n in ("srgb", "s-rgb", "rec709", "rec.709"):
            return QColorSpace(QColorSpace.NamedColorSpace.SRgb)
        if n in ("display p3", "displayp3", "p3"):
            return QColorSpace(QColorSpace.NamedColorSpace.DisplayP3)
        if n in ("adobergb", "adobe rgb", "adobe-rgb"):
            return QColorSpace(QColorSpace.NamedColorSpace.AdobeRgb)
        return None
    except Exception:
        return None


def _apply_color_policy(image_service: "ImageService", img: QImage) -> QImage:
    """ImageService 설정에 따라 색공간을 변환한다."""
    try:
        mode = str(getattr(image_service, "_color_view_mode", "managed") or "managed")
        if mode == "original":
            return img
        # ICC 무시 또는 미탑재시 가정 색공간 부여
        ignore_icc = bool(getattr(image_service, "_icc_ignore_embedded", False))
        try:
            cs = img.colorSpace()
        except Exception:
            cs = QColorSpace()
        if ignore_icc or (not cs.isValid()):
            assumed = str(getattr(image_service, "_assumed_colorspace", "sRGB") or "sRGB")
            assumed_cs = _named_colorspace_from_string(assumed)
            if assumed_cs is not None:
                try:
                    img.setColorSpace(assumed_cs)
                    cs = img.colorSpace()
                except Exception:
                    pass
        # 타깃 변환(소프트 프루핑/뷰 변환)
        target_name = str(getattr(image_service, "_preview_target", "sRGB") or "sRGB")
        target_cs = _named_colorspace_from_string(target_name) or QColorSpace(QColorSpace.NamedColorSpace.SRgb)
        try:
            if cs.isValid() and cs != target_cs:
                converted = img.convertToColorSpace(target_cs)
                if not converted.isNull():
                    return converted
        except Exception:
            pass
        # 폴백 정책
        pol = str(getattr(image_service, "_fallback_policy", "ignore") or "ignore")
        if pol == "force_sRGB":
            try:
                forced = img.convertToColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
                if not forced.isNull():
                    return forced
            except Exception:
                pass
        return img
    except Exception:
        return img


def _apply_transform(img: QImage, rotation_degrees: int, flip_h: bool, flip_v: bool) -> QImage:
    try:
        t = QTransform()
        rot = int(rotation_degrees) % 360
        if rot:
            t.rotate(rot)
        sx = -1.0 if flip_h else 1.0
        sy = -1.0 if flip_v else 1.0
        if sx != 1.0 or sy != 1.0:
            t.scale(sx, sy)
        # Smooth for quality; QImage handles bounds expansion automatically
        return img.transformed(t, Qt.TransformationMode.SmoothTransformation)
    except Exception:
        return img


def _guess_format_from_path(path: str) -> str:
    try:
        ext = os.path.splitext(path)[1].lower().lstrip('.')
        if ext == 'jpg':
            return 'JPEG'
        if ext == 'tif':
            return 'TIFF'
        if ext:
            return ext.upper()
    except Exception:
        pass
    return ''


def _save_qimage(img: QImage, dest_path: str, quality: int) -> tuple[bool, str]:
    fmt = _guess_format_from_path(dest_path)
    try:
        # 품질은 JPEG 등에 적용. 포맷 추정 실패 시 Qt가 확장자로 추정
        ok = img.save(dest_path, fmt if fmt else None, quality)
        if not ok:
            return False, "이미지를 저장할 수 없습니다."
        return True, ""
    except Exception as e:
        return False, str(e)


def _normalize_rotation(rot: int) -> int:
    rot = int(rot) % 360
    if rot % 90 != 0:
        rot = (round(rot / 90.0) * 90) % 360
    return rot


def _sanitize_quality(q: int) -> int:
    try:
        qi = int(q)
        return 1 if qi < 1 else (100 if qi > 100 else qi)
    except Exception:
        return 95


class _QImageCache:
    """바이트 상한 기반 간단 LRU 캐시(QImage)."""

    def __init__(self, max_bytes: int = 256 * 1024 * 1024):
        self._max_bytes = int(max_bytes)
        self._store: "OrderedDict[str, QImage]" = OrderedDict()
        self._bytes_used = 0

    def _estimate_bytes(self, img: QImage) -> int:
        try:
            return int(img.sizeInBytes())
        except Exception:
            # 폴백 추정
            bpp = max(1, int(img.depth() / 8))
            return img.width() * img.height() * bpp

    def get(self, key: str) -> Optional[QImage]:
        try:
            img = self._store.pop(key)
        except KeyError:
            return None
        # 최근 사용으로 이동
        self._store[key] = img
        return img

    def put(self, key: str, img: QImage) -> None:
        if not key:
            return
        # 기존 항목 제거
        if key in self._store:
            try:
                old = self._store.pop(key)
                self._bytes_used -= self._estimate_bytes(old)
            except Exception:
                pass
        self._store[key] = img
        self._bytes_used += self._estimate_bytes(img)
        self._evict_if_needed()

    def delete(self, key: str) -> None:
        if key in self._store:
            try:
                img = self._store.pop(key)
                self._bytes_used -= self._estimate_bytes(img)
            except Exception:
                pass

    def clear(self) -> None:
        self._store.clear()
        self._bytes_used = 0

    def delete_prefix(self, prefix: str) -> None:
        """주어진 접두사로 시작하는 모든 키를 제거한다."""
        try:
            keys = list(self._store.keys())
        except Exception:
            keys = []
        for k in keys:
            try:
                if k.startswith(prefix):
                    img = self._store.pop(k)
                    try:
                        self._bytes_used -= self._estimate_bytes(img)
                    except Exception:
                        pass
            except Exception:
                pass

    def _evict_if_needed(self) -> None:
        while self._bytes_used > self._max_bytes and self._store:
            k, v = self._store.popitem(last=False)  # LRU 제거
            try:
                self._bytes_used -= self._estimate_bytes(v)
            except Exception:
                pass

    def shrink_to(self, max_bytes: int) -> None:
        try:
            self._max_bytes = max(1, int(max_bytes))
            self._evict_if_needed()
        except Exception:
            pass


class _PreloadTask(QRunnable):
    """경량 프리로드 작업: QImage를 디코드해 콜백으로 전달."""

    def __init__(self, path: str, generation: int, done: Callable[[str, QImage, bool, str, int], None], retry_count: int = 0, retry_delay_ms: int = 0):
        super().__init__()
        self._path = path
        self._generation = generation
        self._done = done
        self._retry_count = max(0, int(retry_count))
        self._retry_delay_ms = max(0, int(retry_delay_ms))

    def run(self) -> None:
        try:
            attempts = 0
            last_err = ""
            while True:
                img, ok, err = _read_qimage_with_exif_auto_transform(self._path)
                if ok and img and not img.isNull():
                    self._done(self._path, img, True, "", self._generation)
                    return
                last_err = err or ""
                if attempts >= self._retry_count:
                    break
                attempts += 1
                if self._retry_delay_ms > 0:
                    try:
                        import time
                        time.sleep(self._retry_delay_ms / 1000.0)
                    except Exception:
                        pass
            self._done(self._path, QImage(), False, last_err or "프리로드 실패", self._generation)
        except Exception as e:
            self._done(self._path, QImage(), False, str(e), self._generation)
