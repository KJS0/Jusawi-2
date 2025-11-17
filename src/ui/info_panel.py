from __future__ import annotations

import os
from typing import Optional, Tuple

# PyQt imports are type-ignored to avoid hard deps at import time
from PyQt6.QtCore import QTimer  # type: ignore[import]
from PyQt6.QtGui import QPixmap  # type: ignore[import]
from PyQt6.QtWidgets import QTextEdit, QLabel  # type: ignore[import]


def setup_info_panel(owner) -> None:
    """Initialize info panel related timers, tokens, and connect signals on owner.

    Owner is expected to be the main window instance providing:
    - _kick_map_fetch(), _on_map_ready()
    - attributes: _map_req_token, _map_debounce, _map_emitter
    - widgets: info_text, info_map_label
    """
    # debounce/token
    owner._map_req_token = 0
    owner._map_debounce = QTimer(owner)
    owner._map_debounce.setSingleShot(True)
    owner._map_debounce.setInterval(300)
    owner._map_debounce.timeout.connect(owner._kick_map_fetch)

    # simple emitter object with Qt signal for map readiness
    from PyQt6.QtCore import QObject, pyqtSignal  # type: ignore[import]

    class _MapEmitter(QObject):
        ready = pyqtSignal(int, QPixmap)

    owner._map_emitter = _MapEmitter(owner)
    try:
        owner._map_emitter.ready.connect(owner._on_map_ready)
    except Exception:
        pass

    # 지도 라벨 포커스/휠/키 입력 처리(확대/축소)
    try:
        from PyQt6.QtCore import QObject, QEvent, Qt  # type: ignore[import]

        class _MapEventFilter(QObject):
            def eventFilter(self, _obj, event):
                try:
                    et = event.type()
                except Exception:
                    return False
                # Ctrl+휠로 줌 변경
                try:
                    if et == QEvent.Type.Wheel:
                        try:
                            mods = int(event.modifiers())
                        except Exception:
                            mods = 0
                        if (mods & int(Qt.KeyboardModifier.ControlModifier)) == 0:
                            return False
                        dy = event.angleDelta().y() if hasattr(event, 'angleDelta') else 0
                        step = 1 if dy > 0 else (-1 if dy < 0 else 0)
                        if step != 0:
                            _bump_map_zoom(owner, step)
                            return True
                        return False
                    # 지도 라벨 포커스/호버 상태에서 + / - 키로 줌 변경
                    if et == QEvent.Type.KeyPress:
                        try:
                            key = int(event.key())
                        except Exception:
                            key = 0
                        # 호버 또는 포커스 중일 때만 처리
                        hovered = False
                        try:
                            pos = owner.mapFromGlobal(owner.cursor().pos())
                            hovered = owner.info_map_label.geometry().contains(pos)
                        except Exception:
                            hovered = False
                        if not (hovered or owner.info_map_label.hasFocus()):
                            return False
                        from PyQt6.QtCore import Qt as _Qt  # type: ignore
                        if key in (ord('+'), _Qt.Key.Key_Plus, _Qt.Key.Key_Equal):
                            _bump_map_zoom(owner, +1)
                            return True
                        if key in (ord('-'), _Qt.Key.Key_Minus, _Qt.Key.Key_Underscore):
                            _bump_map_zoom(owner, -1)
                            return True
                        return False
                except Exception:
                    return False
                return False

        # 내부 헬퍼: 줌 변경 및 재요청
        def _bump_map_zoom(_owner, delta: int) -> None:
            try:
                z = int(getattr(_owner, "_info_map_zoom", 0) or 0)
                if z <= 0:
                    try:
                        z = int(getattr(_owner, "_info_map_default_zoom", 12) or 12)
                    except Exception:
                        z = 12
                z = max(1, min(20, z + int(delta)))
                _owner._info_map_zoom = z
            except Exception:
                pass
            # 현재 좌표로 재요청
            try:
                lat = lon = None
                if hasattr(_owner, "_pending_map"):
                    try:
                        lat, lon = float(_owner._pending_map[0]), float(_owner._pending_map[1])
                    except Exception:
                        lat = lon = None
                if lat is None or lon is None:
                    return
                w = int(getattr(_owner.info_map_label, 'width', lambda: 600)())
                h = int(getattr(_owner.info_map_label, 'height', lambda: 360)())
                schedule_map_fetch(_owner, float(lat), float(lon), int(max(64, w)), int(max(64, h)), int(getattr(_owner, "_info_map_zoom", z)))
            except Exception:
                pass

        if getattr(owner, "info_map_label", None) is not None:
            try:
                owner.info_map_label.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            except Exception:
                pass
            _f = _MapEventFilter(owner)
            try:
                owner.info_map_label.installEventFilter(_f)
                owner._map_event_filter = _f  # 보관(수명 유지)
            except Exception:
                pass
    except Exception:
        pass


def toggle_info_panel(owner) -> None:
    try:
        target = getattr(owner, "info_tabs", None) or getattr(owner, "info_text", None)
        visible = not bool(target.isVisible()) if target is not None else True
    except Exception:
        visible = True
    try:
        if getattr(owner, "info_panel", None) is not None:
            owner.info_panel.setVisible(visible)
    except Exception:
        pass
    if visible:
        try:
            update_info_panel(owner)
        except Exception:
            pass


def format_bytes(num_bytes: int) -> str:
    try:
        n = float(num_bytes)
    except Exception:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if i == 0:
        return f"{int(n)} {units[i]}"
    return f"{n:.2f} {units[i]}"


def _safe_frac_to_float(v):
    try:
        if hasattr(v, "numerator") and hasattr(v, "denominator"):
            num = float(getattr(v, "numerator"))
            den = float(getattr(v, "denominator"))
            if den == 0:
                return None
            return num / den
        if isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            if float(v[1]) == 0:
                return None
            return float(v[0]) / float(v[1])
        s = str(v)
        if "/" in s:
            a, b = s.split("/", 1)
            fa, fb = float(a), float(b)
            if fb != 0:
                return fa / fb
        return float(s)
    except Exception:
        return None


def update_info_panel(owner) -> None:
    path = owner.current_image_path or ""
    if not path or not os.path.exists(path):
        try:
            if getattr(owner, "info_text", None) is not None:
                owner.info_text.setPlainText("")
            if getattr(owner, "info_map_label", None) is not None:
                owner.info_map_label.setText("여기에 지도가 표시됩니다.")
        except Exception:
            pass
        return

    file_name = os.path.basename(path)
    dir_name = os.path.dirname(path)
    try:
        size_bytes = os.path.getsize(path)
    except Exception:
        size_bytes = 0

    try:
        w = int(getattr(owner, "_fullres_image", None).width()) if getattr(owner, "_fullres_image", None) is not None else 0
        h = int(getattr(owner, "_fullres_image", None).height()) if getattr(owner, "_fullres_image", None) is not None else 0
        if w <= 0 or h <= 0:
            px = owner.image_display_area.originalPixmap()
            if px is not None and not px.isNull():
                w = int(px.width())
                h = int(px.height())
    except Exception:
        w = h = 0

    mp_text = "-"
    try:
        if w > 0 and h > 0:
            mp = (w * h) / 1_000_000.0
            mp_text = f"{mp:.1f}MP"
    except Exception:
        pass

    summary_text = ""
    try:
        from ..services.exif_utils import extract_with_pillow, format_summary_text  # type: ignore
        exif_raw = extract_with_pillow(path) or {}
        summary_text = format_summary_text(exif_raw, path)
    except Exception:
        summary_text = ""

    exposure_bias_ev = None
    try:
        from PIL import Image, ExifTags  # type: ignore
        if Image is not None:
            with Image.open(path) as im:
                ev_exif = im.getexif()
                name_map = getattr(ExifTags, 'TAGS', {})
                for tag_id, val in (ev_exif.items() if ev_exif else []):
                    name = name_map.get(tag_id, str(tag_id))
                    if name == 'ExposureBiasValue':
                        fv = _safe_frac_to_float(val)
                        if fv is not None:
                            exposure_bias_ev = fv
                            break
    except Exception:
        exposure_bias_ev = None

    address_text = None
    try:
        lat = exif_raw.get("lat") if isinstance(exif_raw, dict) else None  # type: ignore[name-defined]
        lon = exif_raw.get("lon") if isinstance(exif_raw, dict) else None  # type: ignore[name-defined]
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            # 프라이버시 보호: 위치 숨김이면 역지오코딩/지도 비활성화
            if not bool(getattr(owner, "_privacy_hide_location", False)):
                try:
                    from ..services.geocoding import geocoding_service  # type: ignore
                    # 언어: 뷰어 설정(없으면 ko)
                    try:
                        _lang = str(getattr(owner, "_ai_language", "ko") or "ko")
                    except Exception:
                        _lang = "ko"
                    addr = geocoding_service.get_address_from_coordinates(float(lat), float(lon), language=_lang)
                    if addr and isinstance(addr, dict):
                        address_text = str(addr.get("formatted") or addr.get("full_address") or "")
                except Exception:
                    address_text = None
        if getattr(owner, "info_map_label", None) is not None:
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and owner.info_panel.isVisible() and not bool(getattr(owner, "_privacy_hide_location", False)):
                try:
                    owner.info_map_label.setVisible(True)
                except Exception:
                    pass
                # 초기 줌/크기 적용
                try:
                    if not hasattr(owner, "_info_map_zoom") or int(getattr(owner, "_info_map_zoom", 0) or 0) <= 0:
                        owner._info_map_zoom = int(getattr(owner, "_info_map_default_zoom", 12) or 12)
                except Exception:
                    owner._info_map_zoom = 12
                # 라벨 크기에 맞춰 요청
                try:
                    w = int(owner.info_map_label.width())
                    h = int(owner.info_map_label.height())
                except Exception:
                    w, h = 600, 360
                schedule_map_fetch(owner, float(lat), float(lon), int(max(64, w)), int(max(64, h)), int(getattr(owner, "_info_map_zoom", 12)))
                # 외부 지도 링크 툴팁 설정 (Google Maps)
                try:
                    owner.info_map_label.setToolTip(f"https://maps.google.com/?q={float(lat)},{float(lon)}")
                except Exception:
                    pass
            else:
                try:
                    owner.info_map_label.setVisible(False)
                except Exception:
                    pass
    except Exception:
        try:
            if getattr(owner, "info_map_label", None) is not None:
                owner.info_map_label.setVisible(False)
        except Exception:
            pass

        if not summary_text:
            lines = []
            dt = "-"
            lines.append(f"촬영 날짜 및 시간: {dt}")
            lines.append(f"파일명: {file_name}")
            lines.append(f"디렉토리명: {dir_name}")
            # 촬영 기기/초점거리/ISO/조리개/셔터속도는 값이 없을 때 공란으로 표시
            lines.append("촬영 기기: ")
            res = f"{w} x {h}" if w > 0 and h > 0 else "-"
            # 용량 | 해상도 | 화소 한 줄로 표시
            lines.append(f"용량: {format_bytes(size_bytes)} | 해상도: {res} | 화소수: {mp_text}")
            lines.append("ISO: ")
            lines.append("초점 거리: ")
            lines.append("노출도: ")
            lines.append("조리개값: ")
            lines.append("셔터속도: ")
            lines.append("GPS 위도, 경도: -")
            summary_text = "\n".join(lines)

    try:
        if address_text:
            lines = (summary_text or "").splitlines()
            inserted = False
            for i, line in enumerate(lines):
                if line.strip().startswith("GPS 위도, 경도:"):
                    lines.insert(i + 1, f"주소 : {address_text}")
                    inserted = True
                    break
            if not inserted and summary_text:
                lines.append(f"{address_text}")
            if lines:
                summary_text = "\n".join(lines)
    except Exception:
        pass

    # 요약 표시 항목/표기 옵션 적용
    try:
        # 표시 항목 구성
        show_map = {
            "촬영 날짜 및 시간:": bool(getattr(owner, "_info_show_dt", True)),
            "파일명:": bool(getattr(owner, "_info_show_file", True)),
            "디렉토리명:": bool(getattr(owner, "_info_show_dir", True)),
            "촬영 기기:": bool(getattr(owner, "_info_show_cam", True)),
            "용량:": bool(getattr(owner, "_info_show_size", True)),
            "해상도:": bool(getattr(owner, "_info_show_res", True)),
            "화소수:": bool(getattr(owner, "_info_show_mp", True)),
            "ISO:": bool(getattr(owner, "_info_show_iso", True)),
            "초점 거리:": bool(getattr(owner, "_info_show_focal", True)),
            "조리개값:": bool(getattr(owner, "_info_show_aperture", True)),
            "셔터속도:": bool(getattr(owner, "_info_show_shutter", True)),
            "GPS 위도, 경도:": bool(getattr(owner, "_info_show_gps", True)),
        }
        # 줄 필터링 및 셔터 단위 변환
        unit = str(getattr(owner, "_info_shutter_unit", "auto") or "auto")
        new_lines: list[str] = []
        for ln in (summary_text or "").splitlines():
            try:
                key = ln.split(' ', 1)[0] + (':' if ':' not in ln.split(' ', 1)[0] else '')
            except Exception:
                key = ln[:ln.find(':')+1] if ':' in ln else ln
            # 표시 여부
            keep = True
            for prefix, enabled in show_map.items():
                if ln.strip().startswith(prefix):
                    keep = bool(enabled)
                    break
            if not keep:
                continue
            # 셔터 표기 강제 변환
            if unit in ("sec", "frac") and ln.strip().startswith("셔터속도:"):
                try:
                    # exif_raw에서 초 단위 값 사용
                    sec = None
                    try:
                        sec = float(exif_raw.get('exposure_time')) if isinstance(exif_raw, dict) and exif_raw.get('exposure_time') is not None else None
                    except Exception:
                        sec = None
                    if sec and sec > 0:
                        if unit == 'sec':
                            if abs(sec - round(sec)) < 1e-3:
                                sh_txt = f"{int(round(sec))}s"
                            else:
                                sh_txt = f"{sec:.1f}s"
                        else:
                            if sec >= 1.0:
                                sh_txt = f"{int(round(sec))}s"
                            else:
                                den = max(1, round(1.0 / sec))
                                sh_txt = f"1/{den}s"
                        ln = "셔터속도: " + sh_txt
                except Exception:
                    pass
            new_lines.append(ln)
        # 줄 수 제한 적용
        try:
            max_lines = int(getattr(owner, "_info_max_lines", 50) or 50)
        except Exception:
            max_lines = 50
        if max_lines > 0 and len(new_lines) > max_lines:
            new_lines = new_lines[:max_lines]
        summary_text = "\n".join(new_lines)
    except Exception:
        pass

    try:
        if getattr(owner, "info_text", None) is not None:
            owner.info_text.setPlainText(summary_text)
    except Exception:
        pass


def schedule_map_fetch(owner, lat: float, lon: float, w: int, h: int, zoom: int) -> None:
    # 설정: 지도 프리페치 허용 시에만 예약
    if not bool(getattr(owner, "_enable_map_prefetch", True)):
        return
    owner._pending_map = (lat, lon, w, h, zoom)
    try:
        owner._map_debounce.start()
    except Exception:
        kick_map_fetch(owner)


def kick_map_fetch(owner) -> None:
    if not hasattr(owner, "_pending_map"):
        return
    lat, lon, w, h, zoom = owner._pending_map
    owner._map_req_token += 1
    token = int(owner._map_req_token)
    try:
        from ..services.map_cache import submit_fetch  # type: ignore
        # 제공자 고정: Google (키 미설정/실패 시 geocoding.get_static_map_png에서 OSM으로 자동 폴백)
        submit_fetch(lat, lon, int(w), int(h), int(zoom), token, owner._map_emitter, "ready", provider="google")
    except Exception:
        pass


def on_map_ready(owner, token: int, pm) -> None:
    if token != getattr(owner, "_map_req_token", 0):
        return
    if getattr(owner, "info_map_label", None) is None:
        return
    try:
        if pm is not None and not pm.isNull():
            owner.info_map_label.setPixmap(pm)
            owner.info_map_label.setVisible(True)
        else:
            owner.info_map_label.setVisible(False)
    except Exception:
        pass


def update_info_panel_sizes(owner) -> None:
    try:
        # 정보 패널 폭은 기존 규칙 유지
        # 고정 소형 폭으로 유지하여 설정 적용 후에도 커지지 않도록 함
        try:
            panel_w = int(getattr(owner, "_info_panel_fixed_width", 320) or 320)
        except Exception:
            panel_w = 320
        # 지도 미리보기는 텍스트 에리어 폭에 맞추되, 높이를 작게 고정
        # 높이: 패널 폭의 0.35배, 최소 120px, 최대 200px
        panel_h = int(panel_w * 0.35)
        if panel_h < 120:
            panel_h = 120
        if panel_h > 200:
            panel_h = 200
        if getattr(owner, "info_map_label", None) is not None:
            try:
                owner.info_map_label.setFixedSize(panel_w, panel_h)
            except Exception:
                pass
        if getattr(owner, "info_text", None) is not None:
            try:
                scaled = 16
                owner.info_text.setFixedWidth(panel_w)
                owner.info_text.setStyleSheet(f"QTextEdit {{ color: #EAEAEA; background-color: #2B2B2B; border: 1px solid #444; font-size: {scaled}px; line-height: 140%; }}")
            except Exception:
                pass
        if getattr(owner, "info_panel", None) is not None:
            try:
                owner.info_panel.setFixedWidth(panel_w)
            except Exception:
                pass
    except Exception:
        pass


