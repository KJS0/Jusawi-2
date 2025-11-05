from __future__ import annotations

import os
import io
import json
import base64
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable
import hashlib
import pathlib
import time
import random

try:
    from PIL import Image, ImageCms  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageCms = None  # type: ignore

# 프록시/외부 HTTP 클라이언트 사용 안 함

from ..utils.logging_setup import get_logger
from ..storage.settings_store import load_settings as _load_settings  # only for type hints

_log = get_logger("svc.AIAnalysis")


SCHEMA_DEFAULT: Dict[str, Any] = {
    "short_caption": "",
    "long_caption": "",
    "tags": [],
    "subjects": [],
    "shooting_intent": None,
    "camera_settings": {
        "aperture": None,
        "shutter": None,
        "iso": None,
        "focal_length_mm": None,
        "focal_length_35mm_eq_mm": None,
    },
    "gps": {"lat": None, "lon": None, "place_guess": None},
    "safety": {"nsfw": False, "sensitive": []},
    "confidence": 0.0,
    "notes": "",
}


@dataclass
class AnalysisContext:
    purpose: str = "archive"  # blog | archive | sns
    tone: str = "중립"
    language: str = "ko"
    long_caption_chars: int = 120  # 80~160 권장
    short_caption_words: int = 16  # 12~18 권장
    prompt_version: str = "20250907_1"
    user_keywords: Optional[str] = None
    # OpenAI Vision detail level: "low" | "high" | "auto"
    detail: str = "auto"


@dataclass
class AIConfig:
    provider: str = "openai"
    model: str = "gpt-5-nano"  # allow-only policy is enforced by UI if needed
    api_key: str = ""
    # Behavior
    fast_mode: bool = False
    offline_mode: bool = False
    exif_level: str = "full"  # full|summary|none (summary는 현재 full과 동치로 처리)
    retry_count: int = 1
    retry_delay_s: float = 0.8
    http_timeout_s: float = 120.0
    # Cache
    cache_enable: bool = True
    cache_ttl_s: int = 0  # 0=forever
    cache_dir: str = os.path.join(os.path.expanduser("~"), ".jusawi_ai_cache")
    # Image preprocess
    img_max_side: int = 1024
    img_min_side: int = 512
    img_jpeg_quality: int = 80
    img_target_bytes: int = 600000


def _extract_exif_summary(image_path: str) -> Dict[str, Any]:
    """Pillow로 가벼운 EXIF 요약을 추출한다. 실패 시 빈 값으로 반환."""
    if Image is None or not os.path.exists(image_path):
        return {}
    try:
        from PIL import ExifTags  # type: ignore
    except Exception:
        return {}
    try:
        with Image.open(image_path) as im:
            exif_map: Dict[str, Any] = {}
            gps_map: Dict[str, Any] = {}
            try:
                exif = im.getexif()
                if not exif:
                    # 폴백: info['exif'] 바이트에서 로드 시도
                    try:
                        raw = im.info.get("exif")
                        if raw:
                            exif = Image.Exif()
                            exif.load(raw)
                    except Exception:
                        pass
                exif_map_ids: Dict[int, Any] = {}
                if exif:
                    for tag_id, value in exif.items():
                        try:
                            exif_map_ids[int(tag_id)] = value
                        except Exception:
                            pass
                        name = getattr(ExifTags, 'TAGS', {}).get(tag_id, str(tag_id))
                        if name == 'GPSInfo' and isinstance(value, dict):
                            # GPS: 이름/ID 모두 보관
                            try:
                                from PIL.ExifTags import GPSTAGS  # type: ignore
                                gps_map = {GPSTAGS.get(k, str(k)): value[k] for k in value.keys()}
                            except Exception:
                                gps_map = {str(k): value[k] for k in value.keys()}
                            # ID 맵도 함께 준비
                            try:
                                gps_map_ids = {int(k): value[k] for k in value.keys()}
                            except Exception:
                                gps_map_ids = {}
                            continue
                        exif_map[name] = value
            except Exception:
                pass
            # 우리가 필요로 하는 핵심 키만 매핑
            def _to_num(v: Any) -> float | None:
                """다양한 유리수 표현(Fraction, IFDRational, (num,den), 'a/b', 숫자)을 float으로."""
                try:
                    # Fraction 또는 IFDRational
                    if hasattr(v, "numerator") and hasattr(v, "denominator"):
                        num = float(getattr(v, "numerator"))
                        den = float(getattr(v, "denominator"))
                        if den == 0:
                            return None
                        return num / den
                    # (num, den) 튜플
                    if isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                        if float(v[1]) == 0:
                            return None
                        return float(v[0]) / float(v[1])
                    # 'a/b' 문자열
                    s = str(v).strip()
                    if "/" in s:
                        a, b = s.split("/", 1)
                        fa, fb = float(a), float(b)
                        if fb == 0:
                            return None
                        return fa / fb
                    return float(s)
                except Exception:
                    return None

            def _format_shutter(seconds: float) -> str:
                # 1초 미만은 1/Ns, 이상은 Ns로 표기
                try:
                    if seconds <= 0:
                        return "-"
                    if seconds < 1.0:
                        denom = max(1, int(round(1.0 / seconds)))
                        return f"1/{denom}s"
                    # 정수 초는 소수 제거
                    if abs(seconds - round(seconds)) < 1e-3:
                        return f"{int(round(seconds))}s"
                    return f"{seconds:.2f}s"
                except Exception:
                    return "-"

            def _apex_to_shutter(tv_apex: float | None) -> float | None:
                # ShutterSpeedValue(APEX) -> seconds; Tv = log2(1/t)
                try:
                    if tv_apex is None:
                        return None
                    return 2.0 ** (-float(tv_apex))
                except Exception:
                    return None

            out: Dict[str, Any] = {
                "make": str(exif_map.get("Make", "")) or None,
                "model": str(exif_map.get("Model", "")) or None,
                "lens": str(exif_map.get("LensModel", "")) or None,
                "aperture": None,
                "shutter": None,
                "iso": None,
                "focal_length_mm": None,
                "focal_length_35mm_eq_mm": None,
                # DateTimeOriginal(36867) 우선, 폴백: DateTime
                "datetime_original": None,
                "gps": {"lat": None, "lon": None},
            }
            try:
                dto = None
                # 36867: DateTimeOriginal
                if locals().get('exif_map_ids'):
                    dto = exif_map_ids.get(36867)
                if not dto:
                    dto = exif_map.get("DateTimeOriginal", exif_map.get("DateTime", ""))
                out["datetime_original"] = str(dto) or None
            except Exception:
                pass
            # Aperture (FNumber 우선, 없으면 ApertureValue(APEX) 변환)
            try:
                # FNumber(33437)
                fnum = None
                if locals().get('exif_map_ids'):
                    fnum = exif_map_ids.get(33437)
                if fnum is None:
                    fnum = exif_map.get("FNumber")
                if fnum is not None:
                    fv = _to_num(fnum)
                    if fv is not None and fv > 0:
                        out["aperture"] = f"f/{fv:.1f}"
                if out.get("aperture") is None:
                    # ApertureValue(APEX)
                    av = exif_map.get("ApertureValue")
                    avf = _to_num(av)
                    if avf is not None:
                        # Av = log2(N^2) => N = 2^(Av/2)
                        try:
                            import math
                            n = 2.0 ** (float(avf) / 2.0)
                            if n > 0:
                                out["aperture"] = f"f/{n:.1f}"
                        except Exception:
                            pass
            except Exception:
                pass
            # Shutter (ExposureTime 우선, 없으면 ShutterSpeedValue(APEX))
            try:
                shutter_set = False
                # ExposureTime(33434)
                et = None
                if locals().get('exif_map_ids'):
                    et = exif_map_ids.get(33434)
                if et is None:
                    et = exif_map.get("ExposureTime")
                etf = _to_num(et)
                if etf is not None and etf > 0:
                    out["shutter"] = _format_shutter(etf)
                    shutter_set = True
                if not shutter_set:
                    sv = exif_map.get("ShutterSpeedValue")
                    svf = _to_num(sv)
                    secs = _apex_to_shutter(svf)
                    if secs is not None and secs > 0:
                        out["shutter"] = _format_shutter(secs)
            except Exception:
                pass
            # ISO
            try:
                # ISOSpeedRatings/PhotographicSensitivity: 34855 우선
                iso = None
                if locals().get('exif_map_ids'):
                    iso = exif_map_ids.get(34855)
                if iso is None:
                    iso = (
                        exif_map.get("PhotographicSensitivity",
                            exif_map.get("ISOSpeedRatings",
                                exif_map.get("ISOSpeed")
                            )
                        )
                    )
                if iso is not None:
                    # 배열/리스트일 수 있음 -> 첫 값
                    if isinstance(iso, (list, tuple)) and iso:
                        iso_val = iso[0]
                    else:
                        iso_val = iso
                    iso_num = _to_num(iso_val)
                    if iso_num is None:
                        try:
                            iso_num = float(str(iso_val).split()[0])
                        except Exception:
                            iso_num = None
                    if iso_num is not None and iso_num > 0:
                        out["iso"] = int(round(iso_num))
            except Exception:
                pass
            # Focal lengths
            try:
                # FocalLength(37386)
                fl = None
                if locals().get('exif_map_ids'):
                    fl = exif_map_ids.get(37386)
                if fl is None:
                    fl = exif_map.get("FocalLength")
                if fl is not None:
                    fv = _to_num(fl)
                    if fv is not None and fv > 0:
                        out["focal_length_mm"] = int(round(fv))
            except Exception:
                pass
            try:
                # FocalLengthIn35mmFilm(41989) 우선
                fleq = None
                if locals().get('exif_map_ids'):
                    fleq = exif_map_ids.get(41989)
                if fleq is None:
                    fleq = exif_map.get("FocalLengthIn35mmFilm", exif_map.get("FocalLengthIn35mmFormat"))
                if fleq is not None:
                    fv = _to_num(fleq)
                    if fv is None:
                        try:
                            fv = float(str(fleq).split()[0])
                        except Exception:
                            fv = None
                    if fv is not None and fv > 0:
                        out["focal_length_35mm_eq_mm"] = int(round(fv))
            except Exception:
                pass
            # GPS
            try:
                def _to_deg_component(val):
                    n = _to_num(val)
                    return 0.0 if n is None else float(n)
                def _dms_to_deg(dms):
                    d = _to_deg_component(dms[0]) if len(dms) > 0 else 0.0
                    m = _to_deg_component(dms[1]) if len(dms) > 1 else 0.0
                    s = _to_deg_component(dms[2]) if len(dms) > 2 else 0.0
                    return d + (m / 60.0) + (s / 3600.0)
                # GPSInfo(34853) 서브IFD에서 좌표 추출: 이름/ID 모두 지원
                lat = lon = None
                ref_lat = ref_lon = None
                try:
                    if locals().get('gps_map_ids') and isinstance(gps_map_ids, dict) and gps_map_ids:
                        # 1: LatRef, 2: Lat, 3: LonRef, 4: Lon
                        glat = gps_map_ids.get(2)
                        glon = gps_map_ids.get(4)
                        ref_lat = str(gps_map_ids.get(1) or "").upper()
                        ref_lon = str(gps_map_ids.get(3) or "").upper()
                        if isinstance(glat, (list, tuple)) and len(glat) >= 3:
                            lat = _dms_to_deg(glat)
                        if isinstance(glon, (list, tuple)) and len(glon) >= 3:
                            lon = _dms_to_deg(glon)
                    if (lat is None or lon is None) and gps_map:
                        if 'GPSLatitude' in gps_map and 'GPSLatitudeRef' in gps_map:
                            lat = _dms_to_deg(gps_map['GPSLatitude'])
                            ref_lat = str(gps_map.get('GPSLatitudeRef') or "").upper()
                        if 'GPSLongitude' in gps_map and 'GPSLongitudeRef' in gps_map:
                            lon = _dms_to_deg(gps_map['GPSLongitude'])
                            ref_lon = str(gps_map.get('GPSLongitudeRef') or "").upper()
                    if lat is not None and ref_lat and ref_lat.startswith('S'):
                        lat = -abs(lat)
                    if lon is not None and ref_lon and ref_lon.startswith('W'):
                        lon = -abs(lon)
                except Exception:
                    lat = lon = None
                out["gps"] = {"lat": lat, "lon": lon}
            except Exception:
                pass
            return out
    except Exception:
        return {}


def _preprocess_image_for_model(
    image_path: str,
    max_side: int | None = None,
    jpeg_quality: int | None = None,
    target_bytes: int | None = None,
    min_side: int | None = None,
) -> bytes | None:
    """이미지를 sRGB JPEG로 리사이즈/재압축해 바이트로 반환.

    - 크기 예산을 맞추기 위해 품질과 해상도를 점진적으로 낮춤(최대 6회 시도).
    - JPEG 옵션: optimize=True, progressive=True, subsampling=4:2:0.
    """
    if Image is None or not os.path.exists(image_path):
        return None
    try:
        cfg = getattr(globals().get('CURRENT_AI_CONFIG', None), 'value', None)
        # 서비스 인스턴스의 cfg에 접근하기 어렵기 때문에, 상위에서 호출 전 설정을 반영하도록 유지.
        # 여기서는 AIAnalysisService.analyze에서 FAST와 인코딩 파라미터를 직접 전달하지 않으므로 서비스의 _cfg를 조회하도록 변경.
        try:
            # 동적 import 없이 서비스 인스턴스에서 접근되도록, 모듈 전역의 백업을 사용하지 않고 기본값으로 계산
            from inspect import currentframe, getouterframes
            frame = currentframe()
            outer = getouterframes(frame)
            svc = None
            for f in outer:
                self_obj = f.frame.f_locals.get('self')
                if self_obj and hasattr(self_obj, '_cfg') and hasattr(self_obj, '_model'):
                    svc = self_obj
                    break
            if svc is not None:
                FAST = bool(getattr(svc, '_cfg').fast_mode)
                max_side_val = max_side if isinstance(max_side, int) else int(getattr(svc, '_cfg').img_max_side)
                if FAST:
                    max_side_val = min(max_side_val, 768)
                max_side_val = max(256, min(2048, int(max_side_val)))
                min_side_val = min_side if isinstance(min_side, int) else int(getattr(svc, '_cfg').img_min_side)
                min_side_val = max(128, min(max_side_val, int(min_side_val)))
                base_quality = jpeg_quality if isinstance(jpeg_quality, int) else int(getattr(svc, '_cfg').img_jpeg_quality)
                if FAST:
                    base_quality = min(base_quality, 70)
                base_quality = max(40, min(95, int(base_quality)))
                budget = target_bytes if isinstance(target_bytes, int) else int(getattr(svc, '_cfg').img_target_bytes)
                if FAST:
                    budget = min(budget, 350000)
                budget = max(100_000, min(2_000_000, int(budget)))
            else:
                FAST = False
                max_side_val = 1024
                min_side_val = 512
                base_quality = 80
                budget = 600000
        except Exception:
            FAST = False
            max_side_val = 1024
            min_side_val = 512
            base_quality = 80
            budget = 600000

        with Image.open(image_path) as im:
            im = im.convert("RGB")
            # sRGB 변환(ICC 있으면 변환 시도) — FAST 모드에서는 건너뜀
            if not FAST:
                try:
                    prof = im.info.get("icc_profile")
                    if ImageCms is not None and prof:
                        src = ImageCms.ImageCmsProfile(io.BytesIO(prof))
                        dst = ImageCms.createProfile("sRGB")
                        im = ImageCms.profileToProfile(im, src, dst, outputMode="RGB")
                except Exception:
                    pass

            w, h = im.size
            long_side = max(w, h)
            scale = 1.0
            if long_side > max_side_val:
                scale = max_side_val / float(long_side)
                im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)

            def _encode(img: Image.Image, q: int) -> bytes:
                buf = io.BytesIO()
                # subsampling=2 -> 4:2:0, progressive로 전송 최적화
                img.save(buf, format="JPEG", quality=int(q), optimize=True, subsampling=2, progressive=True)
                return buf.getvalue()

            # 1차 시도: 기본 품질로
            out = _encode(im, base_quality)
            if len(out) <= budget or FAST:
                return out

            # 2차: 품질 단계 하향, 그래도 크면 해상도도 단계 하향
            quality_steps = [max(40, base_quality - d) for d in (10, 20, 30)]
            scale_steps = [1.0, 0.85, 0.72, 0.6]
            tried = 1
            best_bytes = len(out)
            best_out = out
            cur = im
            for s in scale_steps:
                if s < 1.0:
                    nw = max(1, int(round(cur.size[0] * s)))
                    nh = max(1, int(round(cur.size[1] * s)))
                    if max(nw, nh) < min_side_val:
                        break
                    cur = cur.resize((nw, nh), Image.LANCZOS)
                for q in quality_steps:
                    tried += 1
                    out2 = _encode(cur, q)
                    blen = len(out2)
                    if blen < best_bytes:
                        best_bytes = blen
                        best_out = out2
                    if blen <= budget:
                        return out2
                    if tried >= 6:
                        break
                if tried >= 6:
                    break
            return best_out
    except Exception as e:
        try:
            _log.error("preprocess_fail | file=%s | err=%s", os.path.basename(image_path), str(e))
        except Exception:
            pass
        return None


def _build_prompt(context: AnalysisContext, exif_summary: Dict[str, Any]) -> str:
    """시스템/지시 메시지는 서버측에서 구성된다고 가정하고, 여기서는 사용자 입력 블록만 작성."""
    lines = []
    lines.append("[컨텍스트]")
    lines.append(f"- purpose: {context.purpose}")
    lines.append(f"- tone: {context.tone}")
    lines.append(f"- language: {context.language}")
    if context.user_keywords:
        lines.append(f"- user_keywords: {context.user_keywords}")
    lines.append("")
    if exif_summary:
        lines.append("[EXIF 요약]")
        for k in [
            "make","model","lens","aperture","shutter","iso",
            "focal_length_mm","focal_length_35mm_eq_mm","datetime_original",
        ]:
            v = exif_summary.get(k)
            if v is not None:
                lines.append(f"{k}: {v}")
        gps = exif_summary.get("gps") or {}
        lat = gps.get("lat")
        lon = gps.get("lon")
        if lat is not None and lon is not None:
            lines.append(f"gps: {{lat: {lat}, lon: {lon}}}")
    return "\n".join(lines)


class AIAnalysisService:
    """멀티모달 모델 호출 래퍼. 프로그램 설정만 사용한다."""

    def __init__(self):
        # 기본 구성(실제 값은 apply_config로 주입)
        self._cfg = AIConfig()
        self._provider = self._cfg.provider
        self._model = self._cfg.model
        self._api_key = self._cfg.api_key
        self._last_error: str = ""

    def apply_config(self, cfg: AIConfig) -> None:
        """런타임 설정을 주입한다. 환경 변수는 사용하지 않는다."""
        if not isinstance(cfg, AIConfig):
            return
        self._cfg = cfg
        self._provider = cfg.provider
        self._model = cfg.model
        self._api_key = cfg.api_key
        self._last_error = ""

    def get_last_error(self) -> str:
        try:
            return str(self._last_error or "")
        except Exception:
            return ""

    def _extract_error_info(self, e: Exception) -> dict:
        info: dict[str, any] = {
            "type": type(e).__name__,
            "message": str(e),
            "status": None,
            "code": None,
            "request_id": None,
            "body": None,
            "text": None,
        }
        try:
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    info["status"] = getattr(resp, "status_code", None)
                except Exception:
                    pass
                try:
                    info["request_id"] = getattr(resp, "request_id", None) or (resp.headers.get("x-request-id") if hasattr(resp, "headers") else None)
                except Exception:
                    pass
                # body/json
                body = None
                try:
                    body = resp.json()
                except Exception:
                    body = None
                info["body"] = body
                # code/type/message from body
                try:
                    err = (body or {}).get("error") if isinstance(body, dict) else None
                    if isinstance(err, dict):
                        info["code"] = err.get("code") or err.get("type")
                        if not info.get("message") and err.get("message"):
                            info["message"] = err.get("message")
                except Exception:
                    pass
                # raw text fallback
                try:
                    txt = getattr(resp, "text", None)
                    if isinstance(txt, str) and txt:
                        info["text"] = txt[:500]
                except Exception:
                    pass
            # some SDK errors expose .status_code/.code directly
            try:
                if not info.get("status"):
                    info["status"] = getattr(e, "status_code", None)
            except Exception:
                pass
            try:
                if not info.get("code"):
                    info["code"] = getattr(e, "code", None)
            except Exception:
                pass
        except Exception:
            pass
        return info

    def analyze(
        self,
        image_path: str,
        context: Optional[AnalysisContext] = None,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        def _p(p: int, msg: str) -> None:
            try:
                if progress_cb:
                    progress_cb(int(p), str(msg))
            except Exception:
                pass
        def _c() -> bool:
            try:
                return bool(is_cancelled and is_cancelled())
            except Exception:
                return False

        ctx = context or AnalysisContext()
        FAST = bool(getattr(self, "_cfg", AIConfig()).fast_mode)
        exif_level = str(getattr(self, "_cfg", AIConfig()).exif_level or "").strip().lower()
        self._last_error = ""
        _p(5, "EXIF 요약 추출")
        if exif_level == "none":
            exif_summary = {}
        else:
            exif_summary = {} if FAST else _extract_exif_summary(image_path)
        if _c():
            self._last_error = "사용자 취소"
            return self._fallback_result(exif_summary, note=self._last_error)
        _p(10, "프롬프트 구성")
        user_prompt = _build_prompt(ctx, exif_summary)
        # 캐시 조회
        try:
            cache_enabled = bool(getattr(self, "_cfg", AIConfig()).cache_enable)
            if cache_enabled:
                key = self._cache_key(image_path, ctx)
                cpath = self._cache_path(key)
                cached = self._load_cache(cpath)
                if cached is not None:
                    _p(95, "캐시 적중")
                    _p(100, "완료")
                    return cached
        except Exception:
            pass
        _p(20, "이미지 전처리")
        img_bytes = _preprocess_image_for_model(image_path)
        if img_bytes is None:
            self._last_error = "이미지 전처리에 실패하여 텍스트 전용으로 폴백"
            return self._fallback_result(exif_summary, note=self._last_error)
        if _c():
            return self._fallback_result(exif_summary, note="사용자 취소")

        try:
            _p(60, "AI 모델 호출")
            result = self._call_openai_with_retry(image_bytes=img_bytes, prompt=user_prompt, context=ctx, progress=_p, is_cancelled=_c)
            if _c():
                self._last_error = "사용자 취소"
                return self._fallback_result(exif_summary, note=self._last_error)
            _p(90, "결과 검증")
            out = self._validate_and_normalize(result, exif_summary)
            # 캐시 저장
            try:
                cache_enabled = bool(getattr(self, "_cfg", AIConfig()).cache_enable)
                if cache_enabled:
                    self._save_cache(cpath, out)
            except Exception:
                pass
            _p(100, "완료")
            self._last_error = ""
            return out
        except Exception as e:
            try:
                _log.warning("ai_call_fallback | err=%s", str(e))
            except Exception:
                pass
            self._last_error = f"요청 실패: {type(e).__name__} - {str(e)}"
            return self._fallback_result(exif_summary, note=f"폴백: {type(e).__name__} | {str(e)}")

    def _cache_key(self, path: str, ctx: AnalysisContext) -> str:
        def _sha1_file(p: str) -> str:
            try:
                h = hashlib.sha1()
                with open(p, "rb") as fh:
                    while True:
                        chunk = fh.read(1024 * 1024)
                        if not chunk:
                            break
                        h.update(chunk)
                return h.hexdigest()
            except Exception:
                return hashlib.sha1((p + "|stat").encode("utf-8")).hexdigest()
        try:
            content = _sha1_file(path)
            fast = "1" if bool(getattr(self, "_cfg", AIConfig()).fast_mode) else "0"
            sig = f"{content}|{ctx.language}|{ctx.long_caption_chars}|{ctx.short_caption_words}|{ctx.purpose}|{ctx.tone}|{self._model}|fast={fast}"
            return hashlib.sha1(sig.encode("utf-8")).hexdigest()
        except Exception:
            return hashlib.sha1((path + "|fallback").encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> str:
        try:
            cfg = getattr(self, "_cfg", AIConfig())
            base_dir = str(getattr(cfg, "cache_dir", os.path.join(os.path.expanduser("~"), ".jusawi_ai_cache")))
        except Exception:
            base_dir = os.path.join(os.path.expanduser("~"), ".jusawi_ai_cache")
        base = pathlib.Path(base_dir)
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return str(base / f"{key}.json")

    def _load_cache(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            if not path or not os.path.isfile(path):
                return None
            # TTL 검사(초). 0=무제한
            try:
                ttl_s = int(getattr(self, "_cfg", AIConfig()).cache_ttl_s)
            except Exception:
                ttl_s = 0
            if ttl_s and ttl_s > 0:
                try:
                    import time as _t
                    mtime = os.path.getmtime(path)
                    if (_t.time() - mtime) > ttl_s:
                        return None
                except Exception:
                    pass
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def _save_cache(self, path: str, data: Dict[str, Any]) -> None:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
        except Exception:
            pass

    def _call_openai_with_retry(self, image_bytes: bytes, prompt: str, context: AnalysisContext,
                                 progress: Callable[[int, str], None], is_cancelled: Callable[[], bool]) -> Dict[str, Any]:
        # 요청이 오래 걸리므로 1회만 시도(요청사항)
        max_attempts = 1
        base_delay = float(getattr(self, "_cfg", AIConfig()).retry_delay_s)
        last_err: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            if is_cancelled():
                raise RuntimeError("취소됨")
            try:
                progress(60, f"모델 호출 시도 {attempt}/{max_attempts}")
                return self._call_openai(image_bytes=image_bytes, prompt=prompt, context=context)
            except Exception as e:
                last_err = e
                if attempt >= max_attempts:
                    break
                delay = base_delay * (2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
                progress(60, f"재시도 대기 {delay:.1f}s")
                time.sleep(delay)
        raise last_err if last_err else RuntimeError("알 수 없는 오류")

    def _call_openai(self, image_bytes: bytes, prompt: str, context: AnalysisContext) -> Dict[str, Any]:
        # 오프라인 모드: 외부 호출 차단
        try:
            if bool(getattr(self, "_cfg", AIConfig()).offline_mode):
                raise RuntimeError("오프라인 모드")
        except Exception:
            pass
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY 없음")
        # 이미지 base64 인라인 전송
        b64 = base64.b64encode(image_bytes).decode("ascii")
        system_msg = (
            "당신은 사진 편집 어시스턴트다. 사용자는 사진 캡션/설명/태그/촬영 의도 추정을 원한다.\n"
            f"- 항상 지정된 언어({context.language})로 출력한다. 이모지·이모티콘·해시태그 금지.\n"
            "- 반드시 JSON 스키마에 맞게만 출력한다. 여분 텍스트 금지.\n"
            "- 사실성 우선, EXIF는 보조 근거. 불확실 요소는 null 또는 notes에 '추정' 기록.\n"
        )
        instruction = (
            "- short_caption: 12~18단어\n"
            "- long_caption: 1~2문단(80~160자)\n"
            "- tags: 6~10개, 소문자/한글 원형화\n"
            "- subjects: [사람, 풍경, 도시, 야생동물, 제품] 중에서 선택\n"
            "- camera_settings: EXIF 없으면 null\n"
            "- gps.place_guess: 좌표 없으면 null\n"
            "- safety/sensitive: 민감 요소 표시\n"
            "- confidence: 0.0~1.0\n"
            "- 금지: 인종/성별 단정, 상표 추측, 광고 문구\n"
        )
        # Responses API (OpenAI 1.x 권장 경로만 사용)
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(f"openai SDK 로드 실패: {e}")

        try:
            timeout_s = float(getattr(self, "_cfg", AIConfig()).http_timeout_s)
        except Exception:
            timeout_s = 20.0
        client = OpenAI(api_key=self._api_key, timeout=timeout_s)

        # 1) Chat Completions 멀티모달 경로(권장)
        try:
            messages = [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction + "\n\n" + prompt},
                        # 일부 모델에서 detail 필드가 400을 유발하므로 제외
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                    ],
                },
            ]
            comp = client.chat.completions.create(
                model=self._model,
                messages=messages
            )
            text = comp.choices[0].message.content or "{}"
            return json.loads(text)
        except Exception as e:
            # 상태/코드/본문/요청ID까지 최대한 추출해 로깅
            info = self._extract_error_info(e)
            try:
                _log.warning(
                    "openai_chat_fail | model=%s | status=%s | code=%s | rid=%s | body=%s | text=%s | type=%s",
                    self._model,
                    info.get("status"),
                    info.get("code"),
                    info.get("request_id"),
                    info.get("body"),
                    info.get("text"),
                    info.get("type"),
                )
            except Exception:
                pass
            # UI용 상세 오류도 보관
            try:
                rid = info.get("request_id")
                self._last_error = f"OpenAI 오류 status={info.get('status')} code={info.get('code')} rid={rid} msg={info.get('message')}"
            except Exception:
                self._last_error = str(e)
            # 실패는 상위 재시도/폴백으로 넘긴다
            raise

        # Responses API는 사용하지 않음

    def _validate_and_normalize(self, data: Dict[str, Any], exif_summary: Dict[str, Any]) -> Dict[str, Any]:
        # 기본 스키마 채우기
        result = json.loads(json.dumps(SCHEMA_DEFAULT))
        try:
            for k, v in data.items():
                result[k] = v
        except Exception:
            pass
        # camera_settings, gps 보정
        cam = result.get("camera_settings") or {}
        if exif_summary:
            # EXIF 기반 보정(충돌 시 notes에 추정으로 남김)
            if cam.get("aperture") is None and exif_summary.get("aperture"):
                cam["aperture"] = exif_summary.get("aperture")
            if cam.get("shutter") is None and exif_summary.get("shutter"):
                cam["shutter"] = exif_summary.get("shutter")
            if cam.get("iso") is None and exif_summary.get("iso") is not None:
                cam["iso"] = exif_summary.get("iso")
            if cam.get("focal_length_mm") is None and exif_summary.get("focal_length_mm") is not None:
                cam["focal_length_mm"] = exif_summary.get("focal_length_mm")
            if cam.get("focal_length_35mm_eq_mm") is None and exif_summary.get("focal_length_35mm_eq_mm") is not None:
                cam["focal_length_35mm_eq_mm"] = exif_summary.get("focal_length_35mm_eq_mm")
        result["camera_settings"] = cam

        gps = result.get("gps") or {"lat": None, "lon": None, "place_guess": None}
        exif_gps = (exif_summary or {}).get("gps") or {}
        if gps.get("lat") is None and exif_gps.get("lat") is not None:
            gps["lat"] = exif_gps.get("lat")
        if gps.get("lon") is None and exif_gps.get("lon") is not None:
            gps["lon"] = exif_gps.get("lon")
        # 역지오코딩으로 place_guess 보강(라이트웨이트, 오류 무시)
        try:
            if (gps.get("place_guess") in (None, "",)) and (gps.get("lat") is not None) and (gps.get("lon") is not None):
                try:
                    from .geocoding import geocoding_service  # type: ignore
                except Exception:
                    geocoding_service = None  # type: ignore
                if geocoding_service is not None:
                    # 언어: 분석 컨텍스트 언어 우선
                    lang = "ko"
                    try:
                        # context는 호출부에서만 접근 가능하므로, place_guess는 한국어 기본으로 설정
                        # 호출부에서 언어가 필요한 경우 NaturalSearch/뷰어에서 표시 언어로 포맷됨
                        lang = str(getattr(getattr(self, "_ctx", None), "language", "ko") or "ko")  # type: ignore[attr-defined]
                    except Exception:
                        lang = "ko"
                    addr = geocoding_service.get_address_from_coordinates(float(gps.get("lat")), float(gps.get("lon")), language=lang)
                    if isinstance(addr, dict):
                        pg = addr.get("formatted") or addr.get("full_address")
                        if isinstance(pg, str) and pg.strip():
                            gps["place_guess"] = str(pg)
        except Exception:
            pass
        result["gps"] = gps

        # 최소 검증
        if not isinstance(result.get("tags"), list):
            result["tags"] = []
        if not isinstance(result.get("subjects"), list):
            result["subjects"] = []
        if not isinstance(result.get("safety"), dict):
            result["safety"] = {"nsfw": False, "sensitive": []}
        if not isinstance(result.get("confidence"), (int, float)):
            result["confidence"] = 0.0
        if not isinstance(result.get("notes"), str):
            result["notes"] = ""
        # 태그 정규화: 소문자/공백정리/중복 제거/금지어 필터
        try:
            raw_tags = [str(t) for t in (result.get("tags") or [])]
            norm: list[str] = []
            seen = set()
            deny = {"nike", "apple", "samsung", "facebook", "google"}
            for t in raw_tags:
                s = t.strip().lower().lstrip("#")
                if not s or s in deny:
                    continue
                if s not in seen:
                    seen.add(s)
                    norm.append(s)
            # 사용자 규칙 적용(~/.jusawi/ai_tag_rules.json)
            try:
                import json as _json, os as _os
                rules_path = _os.path.join(_os.path.expanduser("~"), ".jusawi", "ai_tag_rules.json")
                if _os.path.isfile(rules_path):
                    with open(rules_path, "r", encoding="utf-8") as fh:
                        rules = _json.load(fh) or {}
                    to_remove = set([str(x).strip().lower() for x in (rules.get("remove") or [])])
                    repl = {str(k).strip().lower(): str(v).strip().lower() for k, v in (rules.get("replace") or {}).items()}
                    out2: list[str] = []
                    seen2 = set()
                    for x in norm:
                        if x in to_remove:
                            continue
                        y = repl.get(x, x)
                        if y and y not in seen2:
                            seen2.add(y)
                            out2.append(y)
                    norm = out2
            except Exception:
                pass
            result["tags"] = norm
        except Exception:
            pass
        return result

    def _fallback_result(self, exif_summary: Dict[str, Any], note: str = "") -> Dict[str, Any]:
        result = json.loads(json.dumps(SCHEMA_DEFAULT))
        # EXIF는 보조로 채움
        cam = result["camera_settings"]
        for k in ["aperture", "shutter", "iso", "focal_length_mm", "focal_length_35mm_eq_mm"]:
            if exif_summary.get(k) is not None:
                cam[k] = exif_summary.get(k)
        gps = result["gps"]
        exif_gps = (exif_summary or {}).get("gps") or {}
        gps["lat"] = exif_gps.get("lat")
        gps["lon"] = exif_gps.get("lon")
        # 폴백 노트 및 보수적 신뢰도
        result["notes"] = (note or "")
        result["confidence"] = 0.3
        return result


