import os
from typing import Any, Dict
import base64
import hashlib

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


def _default_config_paths() -> list[str]:
    # 최상단(실행 디렉터리) config.yaml만 사용
    paths: list[str] = []
    try:
        exe_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        p = os.path.join(exe_dir, "config.yaml")
        paths.append(p)
    except Exception:
        pass
    return paths


def _load_yaml_file(path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    try:
        if not path or not os.path.isfile(path):
            return data
        if yaml is None:
            return data
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if isinstance(raw, dict):
            return raw  # type: ignore[return-value]
        return {}
    except Exception:
        return {}


def _merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_dict(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def _kdf() -> bytes:
    try:
        user = (os.getenv("USERNAME") or os.getenv("USER") or "user").encode("utf-8")
        host = (os.getenv("COMPUTERNAME") or (os.uname().nodename if hasattr(os, "uname") else "host")).encode("utf-8")
        return hashlib.sha256(user + b"|" + host).digest()
    except Exception:
        return b"\x00" * 32


def _xor_enc(data: bytes, key: bytes) -> bytes:
    if not key:
        return data
    out = bytearray(len(data))
    for i, b in enumerate(data):
        out[i] = b ^ key[i % len(key)]
    return bytes(out)


def _encrypt_str(plain: str) -> str:
    try:
        if not plain:
            return ""
        raw = plain.encode("utf-8")
        enc = _xor_enc(raw, _kdf())
        return "enc:" + base64.urlsafe_b64encode(enc).decode("ascii")
    except Exception:
        return plain


def _decrypt_str(value: str) -> str:
    try:
        if not value or not value.startswith("enc:"):
            return value
        b64 = value[4:]
        enc = base64.urlsafe_b64decode(b64.encode("ascii"))
        raw = _xor_enc(enc, _kdf())
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _ensure_user_config_exists() -> str | None:
    # 최상위 실행 디렉터리의 config.yaml을 사용. 없으면 생성하지 않음(읽기 전용)
    try:
        for p in _default_config_paths():
            if os.path.isfile(p):
                return p
        return None
    except Exception:
        return None


def _load_yaml_configs() -> Dict[str, Any]:
    # 최상단 config.yaml만 읽는다(병합 없음)
    try:
        paths = _default_config_paths()
        if not paths:
            return {}
        return _load_yaml_file(paths[0])
    except Exception:
        return {}


def _primary_config_path() -> str:
    try:
        exe_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        return os.path.join(exe_dir, "config.yaml")
    except Exception:
        return os.path.abspath("config.yaml")


# ---- Simplified QSettings helpers (stepwise refactor) ----
def _get(settings, key: str, default: Any, caster: Any) -> Any:
    try:
        v = settings.value(key, default)
        return caster(v) if caster else v
    except Exception:
        return default


def _load_core_open(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        viewer._open_scan_dir_after_open = True
        viewer._remember_last_open_dir = True
        return
    viewer._open_scan_dir_after_open = bool(_get(s, "open/scan_dir_after_open", True, bool))
    viewer._remember_last_open_dir = bool(_get(s, "open/remember_last_dir", True, bool))


def _load_core_session_recent(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    pol = str(_get(s, "session/startup_restore_policy", "never", str)) if s else "never"
    viewer._startup_restore_policy = pol if pol in ("always", "ask", "never") else "never"
    viewer._recent_auto_prune_missing = bool(_get(s, "recent/auto_prune_missing", True, bool)) if s else True


def _load_core_anim(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    viewer._anim_autoplay = bool(_get(s, "anim/autoplay", True, bool)) if s else True
    viewer._anim_loop = bool(_get(s, "anim/loop", True, bool)) if s else True


def _load_core_view_color(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    dvm = str(_get(s, "ui/default_view_mode", "fit", str)) if s else "fit"
    if dvm not in ("fit", "fit_width", "fit_height", "actual"):
        dvm = "fit"
    viewer._default_view_mode = dvm
    viewer._preview_target = str(_get(s, "color/preview_target", "sRGB", str)) if s else "sRGB"


def _load_core_prefetch(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    viewer._enable_thumb_prefetch = bool(_get(s, "prefetch/thumbs_enabled", True, bool)) if s else True
    # 추가 프리로드/성능 관련 옵션 로드(기본 안전값 적용)
    try:
        viewer._preload_radius = int(_get(s, "prefetch/preload_radius", getattr(viewer, "_preload_radius", 1), int)) if s else getattr(viewer, "_preload_radius", 1)
    except Exception:
        viewer._preload_radius = getattr(viewer, "_preload_radius", 1)
    try:
        viewer._preload_direction = str(_get(s, "prefetch/preload_direction", "both", str)) if s else "both"
        if viewer._preload_direction not in ("both", "forward", "backward"):
            viewer._preload_direction = "both"
    except Exception:
        viewer._preload_direction = "both"
    try:
        viewer._preload_priority = int(_get(s, "prefetch/preload_priority", -1, int)) if s else -1
    except Exception:
        viewer._preload_priority = -1
    try:
        viewer._preload_only_when_idle = bool(_get(s, "prefetch/only_when_idle", True, bool)) if s else True
    except Exception:
        viewer._preload_only_when_idle = True
    try:
        viewer._prefetch_on_dir_enter = int(_get(s, "prefetch/prefetch_on_dir_enter", 0, int)) if s else 0
    except Exception:
        viewer._prefetch_on_dir_enter = 0
    try:
        viewer._slideshow_prefetch_count = int(_get(s, "prefetch/slideshow_prefetch_count", 0, int)) if s else 0
    except Exception:
        viewer._slideshow_prefetch_count = 0
    # ImageService 연동 설정(존재 시)
    try:
        max_cc = int(_get(s, "prefetch/preload_max_concurrency", 2, int)) if s else 2
    except Exception:
        max_cc = 2
    try:
        retry_c = int(_get(s, "prefetch/preload_retry_count", 0, int)) if s else 0
    except Exception:
        retry_c = 0
    try:
        retry_delay = int(_get(s, "prefetch/preload_retry_delay_ms", 0, int)) if s else 0
    except Exception:
        retry_delay = 0
    try:
        if hasattr(viewer, "image_service") and viewer.image_service is not None:
            viewer._preload_max_concurrency = max_cc
            viewer._preload_retry_count = retry_c
            viewer._preload_retry_delay_ms = retry_delay
            viewer.image_service._preload_max_concurrency = max_cc
            viewer.image_service._preload_retry_count = retry_c
            viewer.image_service._preload_retry_delay_ms = retry_delay
    except Exception:
        pass


def _load_dir_tiff(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        viewer._dir_sort_mode = "metadata"
        viewer._dir_natural_sort = True
        viewer._dir_exclude_hidden_system = True
        viewer._tiff_open_first_page_only = True
        return
    sort_mode = str(_get(s, "dir/sort_mode", "metadata", str))
    viewer._dir_sort_mode = sort_mode if sort_mode in ("metadata", "name") else "metadata"
    viewer._dir_natural_sort = bool(_get(s, "dir/natural_sort", True, bool))
    viewer._dir_exclude_hidden_system = bool(_get(s, "dir/exclude_hidden_system", True, bool))
    viewer._tiff_open_first_page_only = bool(_get(s, "tiff/open_first_page_only", True, bool))


def _load_drag_drop(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        viewer._drop_allow_folder = False
        viewer._drop_use_parent_scan = True
        viewer._drop_show_overlay = True
        viewer._drop_confirm_over_threshold = True
        viewer._drop_large_threshold = 500
        return
    viewer._drop_allow_folder = bool(_get(s, "drop/allow_folder_drop", False, bool))
    viewer._drop_use_parent_scan = bool(_get(s, "drop/use_parent_scan", True, bool))
    viewer._drop_show_overlay = bool(_get(s, "drop/show_progress_overlay", True, bool))
    viewer._drop_confirm_over_threshold = bool(_get(s, "drop/confirm_over_threshold", True, bool))
    viewer._drop_large_threshold = int(_get(s, "drop/large_drop_threshold", 500, int))


def _load_nav_and_zoom_policy(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    viewer._nav_wrap_ends = bool(_get(s, "nav/wrap_ends", False, bool)) if s else False
    viewer._nav_min_interval_ms = int(_get(s, "nav/min_interval_ms", 100, int)) if s else 100
    viewer._filmstrip_auto_center = bool(_get(s, "ui/filmstrip_auto_center", True, bool)) if s else True
    zpol = str(_get(s, "view/zoom_policy", "mode", str)) if s else "mode"
    viewer._zoom_policy = zpol if zpol in ("reset", "mode", "scale") else "mode"


def _load_fullscreen_overlay(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        viewer._fs_auto_hide_ms = 1500
        viewer._fs_auto_hide_cursor_ms = 1200
        viewer._fs_enter_view_mode = "keep"
        viewer._fs_show_filmstrip_overlay = False
        viewer._fs_safe_exit = True
        viewer._overlay_enabled_default = False
        return
    viewer._fs_auto_hide_ms = int(_get(s, "fullscreen/auto_hide_ms", 1500, int))
    viewer._fs_auto_hide_cursor_ms = int(_get(s, "fullscreen/auto_hide_cursor_ms", 1200, int))
    fvm = str(_get(s, "fullscreen/enter_view_mode", "keep", str))
    viewer._fs_enter_view_mode = fvm if fvm in ("keep", "fit", "fit_width", "fit_height", "actual") else "keep"
    viewer._fs_show_filmstrip_overlay = bool(_get(s, "fullscreen/show_filmstrip_overlay", False, bool))
    viewer._fs_safe_exit = bool(_get(s, "fullscreen/safe_exit_rule", True, bool))
    viewer._overlay_enabled_default = bool(_get(s, "overlay/enabled_default", False, bool))


def _load_view_zoom_details(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        viewer._smooth_transform = True
        viewer._fit_margin_pct = 0
        viewer._wheel_zoom_requires_ctrl = True
        viewer._wheel_zoom_alt_precise = True
        viewer._use_fixed_zoom_steps = False
        viewer._zoom_step_factor = 1.25
        viewer._precise_zoom_step_factor = 1.1
        viewer._double_click_action = "toggle"
        viewer._middle_click_action = "none"
        viewer._refit_on_transform = True
        viewer._anchor_preserve_on_transform = True
        viewer._preserve_visual_size_on_dpr_change = False
        viewer._pregen_scales_enabled = False
        viewer._pregen_scales = [0.25, 0.5, 1.0, 2.0]
        return
    viewer._smooth_transform = bool(_get(s, "view/smooth_transform", True, bool))
    viewer._fit_margin_pct = int(_get(s, "view/fit_margin_pct", 0, int))
    viewer._wheel_zoom_requires_ctrl = bool(_get(s, "view/wheel_zoom_requires_ctrl", True, bool))
    viewer._wheel_zoom_alt_precise = bool(_get(s, "view/wheel_zoom_alt_precise", True, bool))
    viewer._use_fixed_zoom_steps = bool(_get(s, "view/use_fixed_zoom_steps", False, bool))
    viewer._zoom_step_factor = float(_get(s, "view/zoom_step_factor", 1.25, float))
    viewer._precise_zoom_step_factor = float(_get(s, "view/precise_zoom_step_factor", 1.1, float))
    viewer._double_click_action = str(_get(s, "view/double_click_action", "toggle", str))
    viewer._middle_click_action = str(_get(s, "view/middle_click_action", "none", str))
    viewer._refit_on_transform = bool(_get(s, "view/refit_on_transform", True, bool))
    viewer._anchor_preserve_on_transform = bool(_get(s, "view/anchor_preserve_on_transform", True, bool))
    viewer._preserve_visual_size_on_dpr_change = bool(_get(s, "view/preserve_visual_size_on_dpr_change", False, bool))
    viewer._pregen_scales_enabled = bool(_get(s, "view/pregen_scales_enabled", False, bool))
    raw = str(_get(s, "view/pregen_scales", "0.25,0.5,1.0,2.0", str))
    arr: list[float] = []
    for p in [t.strip() for t in raw.split(',') if t.strip()]:
        try:
            arr.append(float(p))
        except Exception:
            pass
    viewer._pregen_scales = arr if arr else [0.25, 0.5, 1.0, 2.0]


def _load_cache_limits(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    viewer._img_cache_max_bytes = int(_get(s, "advanced/image_cache_max_bytes", 512*1024*1024, int)) if s else 512*1024*1024
    viewer._scaled_cache_max_bytes = int(_get(s, "advanced/scaled_cache_max_bytes", 768*1024*1024, int)) if s else 768*1024*1024
    viewer._cache_auto_shrink_pct = int(_get(s, "advanced/cache_auto_shrink_pct", 50, int)) if s else 50
    viewer._cache_gc_interval_s = int(_get(s, "advanced/cache_gc_interval_s", 0, int)) if s else 0


def _load_thumb_cache(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    viewer._thumb_cache_quality = int(_get(s, "thumb_cache/quality", 85, int)) if s else 85
    viewer._thumb_cache_dir = str(_get(s, "thumb_cache/dir", "", str)) if s else ""


def _load_ai_automation(viewer: Any) -> None:
    # 사용자 UI에서 제어하지 않는 내부 기능: 기본 비활성화(항상 False)
    viewer._auto_ai_on_open = False
    viewer._auto_ai_on_drop = False
    viewer._auto_ai_on_nav = False
    viewer._auto_ai_delay_ms = 0
    viewer._ai_skip_if_cached = False


def _load_ai_defaults(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    viewer._ai_language = str(_get(s, "ai/language", "ko", str)) if s else "ko"
    viewer._ai_tone = str(_get(s, "ai/tone", "중립", str)) if s else "중립"
    viewer._ai_purpose = str(_get(s, "ai/purpose", "archive", str)) if s else "archive"
    viewer._ai_short_words = int(_get(s, "ai/short_words", 16, int)) if s else 16
    viewer._ai_long_chars = int(_get(s, "ai/long_chars", 120, int)) if s else 120
    viewer._ai_fast_mode = bool(_get(s, "ai/fast_mode", False, bool)) if s else False
    viewer._ai_exif_level = str(_get(s, "ai/exif_level", "full", str)) if s else "full"
    viewer._ai_retry_count = int(_get(s, "ai/retry_count", 2, int)) if s else 2
    viewer._ai_retry_delay_ms = int(_get(s, "ai/retry_delay_ms", 800, int)) if s else 800


def _load_ai_search(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    # verify/search defaults
    viewer._ai_conf_threshold_pct = int(_get(s, "ai/conf_threshold_pct", 80, int)) if s else 80
    viewer._ai_apply_policy = str(_get(s, "ai/apply_policy", "보류", str)) if s else "보류"
    viewer._ai_batch_workers = int(_get(s, "ai/batch_workers", 4, int)) if s else 4
    viewer._ai_batch_delay_ms = int(_get(s, "ai/batch_delay_ms", 0, int)) if s else 0
    viewer._ai_batch_retry_count = int(_get(s, "ai/batch_retry_count", 0, int)) if s else 0
    viewer._ai_batch_retry_delay_ms = int(_get(s, "ai/batch_retry_delay_ms", 0, int)) if s else 0
    viewer._search_verify_mode_default = str(_get(s, "ai/search_verify_mode_default", "strict", str)) if s else "strict"
    viewer._search_verify_topn_default = int(_get(s, "ai/search_verify_topn_default", 20, int)) if s else 20
    viewer._search_sort_order = str(_get(s, "ai/search_sort_order", "similarity", str)) if s else "similarity"
    viewer._search_filter_min_rating = int(_get(s, "ai/search_filter/min_rating", 0, int)) if s else 0
    viewer._search_filter_flag_mode = str(_get(s, "ai/search_filter/flag_mode", "any", str)) if s else "any"
    viewer._search_filter_keywords = str(_get(s, "ai/search_filter/keywords", "", str)) if s else ""
    viewer._search_filter_date_from = str(_get(s, "ai/search_filter/date_from", "", str)) if s else ""
    viewer._search_filter_date_to = str(_get(s, "ai/search_filter/date_to", "", str)) if s else ""
    viewer._search_result_thumb_size = int(_get(s, "ai/search_result/thumb_size", 192, int)) if s else 192
    viewer._search_result_view_mode = str(_get(s, "ai/search_result/view_mode", "grid", str)) if s else "grid"
    viewer._search_show_score = bool(_get(s, "ai/search_result/show_score", True, bool)) if s else True
    viewer._search_show_in_filmstrip = bool(_get(s, "ai/search_result/show_in_filmstrip", False, bool)) if s else False
    viewer._search_bg_prep_enabled = bool(_get(s, "ai/search/bg_prep_enabled", False, bool)) if s else False
    viewer._search_use_embedding = bool(_get(s, "ai/search/use_embedding", True, bool)) if s else True
    viewer._search_verify_strict_only = bool(_get(s, "ai/search/verify_strict_only", True, bool)) if s else True
    viewer._search_verify_max_candidates = int(_get(s, "ai/search/verify_max_candidates", 200, int)) if s else 200
    viewer._search_verify_workers = int(_get(s, "ai/search/verify_workers", 16, int)) if s else 16
    viewer._search_blend_alpha = float(_get(s, "ai/search/blend_alpha", 0.7, float)) if s else 0.7
    viewer._embed_batch_size = int(_get(s, "ai/embed/batch_size", 64, int)) if s else 64
    viewer._embed_model = str(_get(s, "ai/embed/model", "text-embedding-3-small", str)) if s else "text-embedding-3-small"
    viewer._search_tag_weight = int(_get(s, "ai/search_tag_weight", 2, int)) if s else 2
    viewer._bg_index_max = int(_get(s, "ai/bg_index_max", 200, int)) if s else 200
    viewer._privacy_hide_location = bool(_get(s, "ai/privacy_hide_location", False, bool)) if s else False
    viewer._offline_mode = bool(_get(s, "ai/offline_mode", False, bool)) if s else False


def _load_info_panel(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        viewer._info_show_dt = True
        viewer._info_show_file = True
        viewer._info_show_dir = True
        viewer._info_show_cam = True
        viewer._info_show_size = True
        viewer._info_show_res = True
        viewer._info_show_mp = True
        viewer._info_show_iso = True
        viewer._info_show_focal = True
        viewer._info_show_aperture = True
        viewer._info_show_shutter = True
        viewer._info_show_gps = True
        viewer._info_max_lines = 50
        viewer._info_shutter_unit = "auto"
        return
    g = getattr(viewer, "__dict__", {})
    viewer._info_show_dt = bool(_get(s, "info/show_dt", g.get("_info_show_dt", True), bool))
    viewer._info_show_file = bool(_get(s, "info/show_file", g.get("_info_show_file", True), bool))
    viewer._info_show_dir = bool(_get(s, "info/show_dir", g.get("_info_show_dir", True), bool))
    viewer._info_show_cam = bool(_get(s, "info/show_cam", g.get("_info_show_cam", True), bool))
    viewer._info_show_size = bool(_get(s, "info/show_size", g.get("_info_show_size", True), bool))
    viewer._info_show_res = bool(_get(s, "info/show_res", g.get("_info_show_res", True), bool))
    viewer._info_show_mp = bool(_get(s, "info/show_mp", g.get("_info_show_mp", True), bool))
    viewer._info_show_iso = bool(_get(s, "info/show_iso", g.get("_info_show_iso", True), bool))
    viewer._info_show_focal = bool(_get(s, "info/show_focal", g.get("_info_show_focal", True), bool))
    viewer._info_show_aperture = bool(_get(s, "info/show_aperture", g.get("_info_show_aperture", True), bool))
    viewer._info_show_shutter = bool(_get(s, "info/show_shutter", g.get("_info_show_shutter", True), bool))
    viewer._info_show_gps = bool(_get(s, "info/show_gps", g.get("_info_show_gps", True), bool))
    viewer._info_max_lines = int(_get(s, "info/max_lines", g.get("_info_max_lines", 50), int))
    viewer._info_shutter_unit = str(_get(s, "info/shutter_unit", g.get("_info_shutter_unit", "auto"), str))


def _load_map_info(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        viewer._map_static_provider = "auto"
        viewer._info_map_size_mode = "medium"
        viewer._info_map_default_zoom = 12
        viewer._map_cache_max_mb = 128
        viewer._map_cache_max_days = 30
        viewer._map_kakao_api_key = ""
        viewer._map_google_api_key = ""
        return
    viewer._map_static_provider = str(_get(s, "map/static_provider", getattr(viewer, "_map_static_provider", "auto"), str))
    viewer._info_map_size_mode = str(_get(s, "map/preview_size", getattr(viewer, "_info_map_size_mode", "medium"), str))
    viewer._info_map_default_zoom = int(_get(s, "map/default_zoom", getattr(viewer, "_info_map_default_zoom", 12), int))
    viewer._map_cache_max_mb = int(_get(s, "map/cache_max_mb", getattr(viewer, "_map_cache_max_mb", 128), int))
    viewer._map_cache_max_days = int(_get(s, "map/cache_max_days", getattr(viewer, "_map_cache_max_days", 30), int))
    viewer._map_kakao_api_key = str(_get(s, "map/kakao_api_key", "", str))
    viewer._map_google_api_key = str(_get(s, "map/google_api_key", "", str))


def _save_info_panel(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    s.setValue("info/show_dt", bool(getattr(viewer, "_info_show_dt", True)))
    s.setValue("info/show_file", bool(getattr(viewer, "_info_show_file", True)))
    s.setValue("info/show_dir", bool(getattr(viewer, "_info_show_dir", True)))
    s.setValue("info/show_cam", bool(getattr(viewer, "_info_show_cam", True)))
    s.setValue("info/show_size", bool(getattr(viewer, "_info_show_size", True)))
    s.setValue("info/show_res", bool(getattr(viewer, "_info_show_res", True)))
    s.setValue("info/show_mp", bool(getattr(viewer, "_info_show_mp", True)))
    s.setValue("info/show_iso", bool(getattr(viewer, "_info_show_iso", True)))
    s.setValue("info/show_focal", bool(getattr(viewer, "_info_show_focal", True)))
    s.setValue("info/show_aperture", bool(getattr(viewer, "_info_show_aperture", True)))
    s.setValue("info/show_shutter", bool(getattr(viewer, "_info_show_shutter", True)))
    s.setValue("info/show_gps", bool(getattr(viewer, "_info_show_gps", True)))
    s.setValue("info/max_lines", int(getattr(viewer, "_info_max_lines", 50)))
    s.setValue("info/shutter_unit", str(getattr(viewer, "_info_shutter_unit", "auto")))


def _save_map_info(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    s.setValue("map/static_provider", str(getattr(viewer, "_map_static_provider", "auto")))
    s.setValue("map/preview_size", str(getattr(viewer, "_info_map_size_mode", "medium")))
    s.setValue("map/default_zoom", int(getattr(viewer, "_info_map_default_zoom", 12)))
    s.setValue("map/cache_max_mb", int(getattr(viewer, "_map_cache_max_mb", 128)))
    s.setValue("map/cache_max_days", int(getattr(viewer, "_map_cache_max_days", 30)))
    s.setValue("map/kakao_api_key", str(getattr(viewer, "_map_kakao_api_key", "")))
    s.setValue("map/google_api_key", str(getattr(viewer, "_map_google_api_key", "")))


def _save_recent_session_and_edit(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    s.setValue("recent/files", getattr(viewer, "recent_files", []))
    s.setValue("recent/folders", getattr(viewer, "recent_folders", []))
    s.setValue("recent/last_open_dir", getattr(viewer, "last_open_dir", ""))
    s.setValue("session/startup_restore_policy", str(getattr(viewer, "_startup_restore_policy", "always")))
    s.setValue("recent/max_items", int(getattr(viewer, "_recent_max_items", 10)))
    s.setValue("recent/auto_prune_missing", bool(getattr(viewer, "_recent_auto_prune_missing", True)))
    s.setValue("edit/save_policy", str(getattr(viewer, "_save_policy", "discard")))
    s.setValue("edit/jpeg_quality", int(getattr(viewer, "_jpeg_quality", 95)))


def _save_prefetch_all(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    s.setValue("prefetch/thumbs_enabled", bool(getattr(viewer, "_enable_thumb_prefetch", True)))
    s.setValue("prefetch/preload_radius", int(getattr(viewer, "_preload_radius", 2)))
    s.setValue("prefetch/map_enabled", bool(getattr(viewer, "_enable_map_prefetch", True)))
    s.setValue("prefetch/preload_direction", str(getattr(viewer, "_preload_direction", "both")))
    s.setValue("prefetch/preload_priority", int(getattr(viewer, "_preload_priority", -1)))
    s.setValue("prefetch/preload_max_concurrency", int(getattr(viewer, "_preload_max_concurrency", 0)))
    s.setValue("prefetch/preload_retry_count", int(getattr(viewer, "_preload_retry_count", 0)))
    s.setValue("prefetch/preload_retry_delay_ms", int(getattr(viewer, "_preload_retry_delay_ms", 0)))
    s.setValue("prefetch/only_when_idle", bool(getattr(viewer, "_preload_only_when_idle", False)))
    s.setValue("prefetch/prefetch_on_dir_enter", int(getattr(viewer, "_prefetch_on_dir_enter", 0)))
    s.setValue("prefetch/slideshow_prefetch_count", int(getattr(viewer, "_slideshow_prefetch_count", 0)))


def _save_cache_and_thumb(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    s.setValue("advanced/image_cache_max_bytes", int(getattr(viewer, "_img_cache_max_bytes", 256*1024*1024)))
    s.setValue("advanced/scaled_cache_max_bytes", int(getattr(viewer, "_scaled_cache_max_bytes", 384*1024*1024)))
    s.setValue("advanced/cache_auto_shrink_pct", int(getattr(viewer, "_cache_auto_shrink_pct", 50)))
    s.setValue("advanced/cache_gc_interval_s", int(getattr(viewer, "_cache_gc_interval_s", 0)))
    s.setValue("thumb_cache/quality", int(getattr(viewer, "_thumb_cache_quality", 85)))
    s.setValue("thumb_cache/dir", str(getattr(viewer, "_thumb_cache_dir", "")))


def _save_open_and_anim(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    s.setValue("open/scan_dir_after_open", bool(getattr(viewer, "_open_scan_dir_after_open", True)))
    s.setValue("open/remember_last_dir", bool(getattr(viewer, "_remember_last_open_dir", True)))
    s.setValue("anim/autoplay", bool(getattr(viewer, "_anim_autoplay", True)))
    s.setValue("anim/loop", bool(getattr(viewer, "_anim_loop", True)))
    s.setValue("anim/keep_state_on_switch", bool(getattr(viewer, "_anim_keep_state_on_switch", False)))
    s.setValue("anim/pause_on_unfocus", bool(getattr(viewer, "_anim_pause_on_unfocus", False)))
    s.setValue("anim/click_toggle", bool(getattr(viewer, "_anim_click_toggle", False)))
    s.setValue("anim/overlay/enabled", bool(getattr(viewer, "_anim_overlay_enabled", False)))
    s.setValue("anim/overlay/show_index", bool(getattr(viewer, "_anim_overlay_show_index", True)))
    s.setValue("anim/overlay/position", str(getattr(viewer, "_anim_overlay_position", "top-right")))
    s.setValue("anim/overlay/opacity", float(getattr(viewer, "_anim_overlay_opacity", 0.6)))


def _save_dir_tiff(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    s.setValue("dir/sort_mode", str(getattr(viewer, "_dir_sort_mode", "metadata")))
    s.setValue("dir/natural_sort", bool(getattr(viewer, "_dir_natural_sort", True)))
    s.setValue("dir/exclude_hidden_system", bool(getattr(viewer, "_dir_exclude_hidden_system", True)))
    s.setValue("tiff/open_first_page_only", bool(getattr(viewer, "_tiff_open_first_page_only", True)))


def _save_nav_and_zoom_policy(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    s.setValue("nav/wrap_ends", bool(getattr(viewer, "_nav_wrap_ends", False)))
    s.setValue("nav/min_interval_ms", int(getattr(viewer, "_nav_min_interval_ms", 100)))
    s.setValue("ui/filmstrip_auto_center", bool(getattr(viewer, "_filmstrip_auto_center", True)))
    s.setValue("view/zoom_policy", str(getattr(viewer, "_zoom_policy", "mode")))


def _save_fullscreen_overlay(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    s.setValue("fullscreen/auto_hide_ms", int(getattr(viewer, "_fs_auto_hide_ms", 1500)))
    s.setValue("fullscreen/auto_hide_cursor_ms", int(getattr(viewer, "_fs_auto_hide_cursor_ms", 1200)))
    s.setValue("fullscreen/enter_view_mode", str(getattr(viewer, "_fs_enter_view_mode", "keep")))
    s.setValue("fullscreen/show_filmstrip_overlay", bool(getattr(viewer, "_fs_show_filmstrip_overlay", False)))
    s.setValue("fullscreen/safe_exit_rule", bool(getattr(viewer, "_fs_safe_exit", True)))
    s.setValue("overlay/enabled_default", bool(getattr(viewer, "_overlay_enabled_default", False)))


def _save_view_zoom_details(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    s.setValue("ui/default_view_mode", str(getattr(viewer, "_default_view_mode", "fit")))
    s.setValue("view/smooth_transform", bool(getattr(viewer, "_smooth_transform", True)))
    s.setValue("view/refit_on_transform", bool(getattr(viewer, "_refit_on_transform", True)))
    s.setValue("view/anchor_preserve_on_transform", bool(getattr(viewer, "_anchor_preserve_on_transform", True)))
    s.setValue("view/fit_margin_pct", int(getattr(viewer, "_fit_margin_pct", 0)))
    s.setValue("view/wheel_zoom_requires_ctrl", bool(getattr(viewer, "_wheel_zoom_requires_ctrl", True)))
    s.setValue("view/wheel_zoom_alt_precise", bool(getattr(viewer, "_wheel_zoom_alt_precise", True)))
    s.setValue("view/use_fixed_zoom_steps", bool(getattr(viewer, "_use_fixed_zoom_steps", False)))
    s.setValue("view/zoom_step_factor", float(getattr(viewer, "_zoom_step_factor", 1.25)))
    s.setValue("view/precise_zoom_step_factor", float(getattr(viewer, "_precise_zoom_step_factor", 1.1)))
    s.setValue("view/double_click_action", str(getattr(viewer, "_double_click_action", 'toggle')))
    s.setValue("view/middle_click_action", str(getattr(viewer, "_middle_click_action", 'none')))
    s.setValue("view/preserve_visual_size_on_dpr_change", bool(getattr(viewer, "_preserve_visual_size_on_dpr_change", False)))
    # pregen_scales array -> CSV
    try:
        txt = ",".join([str(x) for x in (getattr(viewer, "_pregen_scales", [0.25,0.5,1.0,2.0]))])
    except Exception:
        txt = "0.25,0.5,1.0,2.0"
    s.setValue("view/pregen_scales_enabled", bool(getattr(viewer, "_pregen_scales_enabled", False)))
    s.setValue("view/pregen_scales", txt)


def _save_color_management(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    s.setValue("color/icc_ignore_embedded", bool(getattr(viewer, "_icc_ignore_embedded", False)))
    s.setValue("color/assumed_colorspace", str(getattr(viewer, "_assumed_colorspace", "sRGB")))
    s.setValue("color/preview_target", str(getattr(viewer, "_preview_target", "sRGB")))
    s.setValue("color/fallback_policy", str(getattr(viewer, "_fallback_policy", "ignore")))
    s.setValue("color/anim_convert", bool(getattr(viewer, "_convert_movie_frames_to_srgb", True)))
    s.setValue("color/thumb_convert", bool(getattr(viewer, "_thumb_convert_to_srgb", True)))


def _save_drag_drop(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    s.setValue("drop/allow_folder_drop", bool(getattr(viewer, "_drop_allow_folder", False)))
    s.setValue("drop/use_parent_scan", bool(getattr(viewer, "_drop_use_parent_scan", True)))
    s.setValue("drop/show_progress_overlay", bool(getattr(viewer, "_drop_show_overlay", True)))
    s.setValue("drop/confirm_over_threshold", bool(getattr(viewer, "_drop_confirm_over_threshold", True)))
    s.setValue("drop/large_drop_threshold", int(getattr(viewer, "_drop_large_threshold", 500)))


def _save_ai_automation(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    # 비활성화 상태를 강제로 저장하여 과거 설정을 덮어씀
    s.setValue("ai/auto_on_open", False)
    s.setValue("ai/auto_on_drop", False)
    s.setValue("ai/auto_on_nav", False)
    s.setValue("ai/auto_delay_ms", 0)
    s.setValue("ai/skip_if_cached", False)


def _save_ai_defaults(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    s.setValue("ai/language", str(getattr(viewer, "_ai_language", "ko")))
    s.setValue("ai/tone", str(getattr(viewer, "_ai_tone", "중립")))
    s.setValue("ai/purpose", str(getattr(viewer, "_ai_purpose", "archive")))
    s.setValue("ai/short_words", int(getattr(viewer, "_ai_short_words", 16)))
    s.setValue("ai/long_chars", int(getattr(viewer, "_ai_long_chars", 120)))
    s.setValue("ai/fast_mode", bool(getattr(viewer, "_ai_fast_mode", False)))
    s.setValue("ai/exif_level", str(getattr(viewer, "_ai_exif_level", "full")))
    s.setValue("ai/retry_count", int(getattr(viewer, "_ai_retry_count", 2)))
    s.setValue("ai/retry_delay_ms", int(getattr(viewer, "_ai_retry_delay_ms", 800)))
    s.setValue("ai/offline_mode", bool(getattr(viewer, "_offline_mode", False)))


def _save_ai_search_and_ext(viewer: Any) -> None:
    s = getattr(viewer, "settings", None)
    if not s:
        return
    # 확장 저장
    s.setValue("ai/conf_threshold_pct", int(getattr(viewer, "_ai_conf_threshold_pct", 80)))
    s.setValue("ai/apply_policy", str(getattr(viewer, "_ai_apply_policy", "보류")))
    s.setValue("ai/batch_workers", int(getattr(viewer, "_ai_batch_workers", 4)))
    s.setValue("ai/batch_delay_ms", int(getattr(viewer, "_ai_batch_delay_ms", 0)))
    s.setValue("ai/batch_retry_count", int(getattr(viewer, "_ai_batch_retry_count", 0)))
    s.setValue("ai/batch_retry_delay_ms", int(getattr(viewer, "_ai_batch_retry_delay_ms", 0)))
    s.setValue("ai/search_verify_mode_default", str(getattr(viewer, "_search_verify_mode_default", "strict")))
    s.setValue("ai/search_verify_topn_default", int(getattr(viewer, "_search_verify_topn_default", 20)))
    s.setValue("ai/search_tag_weight", int(getattr(viewer, "_search_tag_weight", 2)))
    s.setValue("ai/bg_index_max", int(getattr(viewer, "_bg_index_max", 200)))
    s.setValue("ai/privacy_hide_location", bool(getattr(viewer, "_privacy_hide_location", False)))
    # 자연어 검색/결과/백그라운드
    s.setValue("ai/search_sort_order", str(getattr(viewer, "_search_sort_order", "similarity")))
    s.setValue("ai/search_filter/min_rating", int(getattr(viewer, "_search_filter_min_rating", 0)))
    s.setValue("ai/search_filter/flag_mode", str(getattr(viewer, "_search_filter_flag_mode", "any")))
    s.setValue("ai/search_filter/keywords", str(getattr(viewer, "_search_filter_keywords", "")))
    s.setValue("ai/search_filter/date_from", str(getattr(viewer, "_search_filter_date_from", "")))
    s.setValue("ai/search_filter/date_to", str(getattr(viewer, "_search_filter_date_to", "")))
    s.setValue("ai/search_result/thumb_size", int(getattr(viewer, "_search_result_thumb_size", 192)))
    s.setValue("ai/search_result/view_mode", str(getattr(viewer, "_search_result_view_mode", "grid")))
    s.setValue("ai/search_result/show_score", bool(getattr(viewer, "_search_show_score", True)))
    s.setValue("ai/search_result/show_in_filmstrip", bool(getattr(viewer, "_search_show_in_filmstrip", False)))
    s.setValue("ai/search/bg_prep_enabled", bool(getattr(viewer, "_search_bg_prep_enabled", False)))
    # 임베딩/검증 고급 옵션
    s.setValue("ai/search/use_embedding", bool(getattr(viewer, "_search_use_embedding", True)))
    s.setValue("ai/search/verify_strict_only", bool(getattr(viewer, "_search_verify_strict_only", True)))
    s.setValue("ai/search/verify_max_candidates", int(getattr(viewer, "_search_verify_max_candidates", 200)))
    s.setValue("ai/search/verify_workers", int(getattr(viewer, "_search_verify_workers", 16)))
    s.setValue("ai/search/blend_alpha", float(getattr(viewer, "_search_blend_alpha", 0.7)))
    s.setValue("ai/embed/batch_size", int(getattr(viewer, "_embed_batch_size", 64)))
    s.setValue("ai/embed/model", str(getattr(viewer, "_embed_model", "text-embedding-3-small")))

def _export_viewer_to_yaml(viewer: Any) -> None:  # noqa: ANN401
    if yaml is None:
        return
    cfg: Dict[str, Any] = {}
    try:
        # edit
        cfg["edit"] = {
            "save_policy": str(getattr(viewer, "_save_policy", "discard")),
            "jpeg_quality": int(getattr(viewer, "_jpeg_quality", 95)),
        }
        # color
        cfg["color"] = {
            "icc_ignore_embedded": bool(getattr(viewer, "_icc_ignore_embedded", False)),
            "assumed_colorspace": str(getattr(viewer, "_assumed_colorspace", "sRGB")),
            "preview_target": str(getattr(viewer, "_preview_target", "sRGB")),
            "fallback_policy": str(getattr(viewer, "_fallback_policy", "ignore")),
            "anim_convert": bool(getattr(viewer, "_convert_movie_frames_to_srgb", True)),
            "thumb_convert": bool(getattr(viewer, "_thumb_convert_to_srgb", True)),
        }
        # open/session/recent
        cfg["open"] = {
            "scan_dir_after_open": bool(getattr(viewer, "_open_scan_dir_after_open", True)),
            "remember_last_dir": bool(getattr(viewer, "_remember_last_open_dir", True)),
        }
        cfg["session"] = {
            "startup_restore_policy": str(getattr(viewer, "_startup_restore_policy", "always")),
        }
        cfg["recent"] = {
            "max_items": int(getattr(viewer, "_recent_max_items", 10)),
            "auto_prune_missing": bool(getattr(viewer, "_recent_auto_prune_missing", True)),
        }
        # dir/tiff
        cfg["dir"] = {
            "sort_mode": str(getattr(viewer, "_dir_sort_mode", "metadata")),
            "natural_sort": bool(getattr(viewer, "_dir_natural_sort", True)),
            "exclude_hidden_system": bool(getattr(viewer, "_dir_exclude_hidden_system", True)),
        }
        cfg["tiff"] = {
            "open_first_page_only": bool(getattr(viewer, "_tiff_open_first_page_only", True)),
        }
        # nav/ui
        cfg["nav"] = {
            "wrap_ends": bool(getattr(viewer, "_nav_wrap_ends", False)),
            "min_interval_ms": int(getattr(viewer, "_nav_min_interval_ms", 100)),
        }
        cfg["ui"] = {
            "filmstrip_auto_center": bool(getattr(viewer, "_filmstrip_auto_center", True)),
        }
        # view
        cfg["view"] = {
            "zoom_policy": str(getattr(viewer, "_zoom_policy", "mode")),
            "default_view_mode": str(getattr(viewer, "_default_view_mode", "fit")),
            "smooth_transform": bool(getattr(viewer, "_smooth_transform", True)),
            "fit_margin_pct": int(getattr(viewer, "_fit_margin_pct", 0)),
            "wheel_zoom_requires_ctrl": bool(getattr(viewer, "_wheel_zoom_requires_ctrl", True)),
            "wheel_zoom_alt_precise": bool(getattr(viewer, "_wheel_zoom_alt_precise", True)),
            "use_fixed_zoom_steps": bool(getattr(viewer, "_use_fixed_zoom_steps", False)),
            "zoom_step_factor": float(getattr(viewer, "_zoom_step_factor", 1.25)),
            "precise_zoom_step_factor": float(getattr(viewer, "_precise_zoom_step_factor", 1.1)),
            "double_click_action": str(getattr(viewer, "_double_click_action", "toggle")),
            "middle_click_action": str(getattr(viewer, "_middle_click_action", "none")),
            "refit_on_transform": bool(getattr(viewer, "_refit_on_transform", True)),
            "anchor_preserve_on_transform": bool(getattr(viewer, "_anchor_preserve_on_transform", True)),
            "preserve_visual_size_on_dpr_change": bool(getattr(viewer, "_preserve_visual_size_on_dpr_change", False)),
            "pregen_scales_enabled": bool(getattr(viewer, "_pregen_scales_enabled", False)),
            "pregen_scales": ",".join([str(x) for x in getattr(viewer, "_pregen_scales", [0.25,0.5,1.0,2.0])]),
            "min_scale_pct": int(round(float(getattr(viewer, "min_scale", 0.01)) * 100)),
            "max_scale_pct": int(round(float(getattr(viewer, "max_scale", 16.0)) * 100)),
        }
        # fullscreen/overlay
        cfg["fullscreen"] = {
            "auto_hide_ms": int(getattr(viewer, "_fs_auto_hide_ms", 1500)),
            "auto_hide_cursor_ms": int(getattr(viewer, "_fs_auto_hide_cursor_ms", 1200)),
            "enter_view_mode": str(getattr(viewer, "_fs_enter_view_mode", "keep")),
            "show_filmstrip_overlay": bool(getattr(viewer, "_fs_show_filmstrip_overlay", False)),
            "safe_exit_rule": bool(getattr(viewer, "_fs_safe_exit", True)),
        }
        cfg["overlay"] = {
            "enabled_default": bool(getattr(viewer, "_overlay_enabled_default", False)),
        }
        # anim (overlay etc.)
        cfg["anim"] = {
            "autoplay": bool(getattr(viewer, "_anim_autoplay", True)),
            "loop": bool(getattr(viewer, "_anim_loop", True)),
            "keep_state_on_switch": bool(getattr(viewer, "_anim_keep_state_on_switch", False)),
            "pause_on_unfocus": bool(getattr(viewer, "_anim_pause_on_unfocus", False)),
            "click_toggle": bool(getattr(viewer, "_anim_click_toggle", False)),
            "overlay": {
                "enabled": bool(getattr(viewer, "_anim_overlay_enabled", False)),
                "show_index": bool(getattr(viewer, "_anim_overlay_show_index", True)),
                "position": str(getattr(viewer, "_anim_overlay_position", "top-right")),
                "opacity": float(getattr(viewer, "_anim_overlay_opacity", 0.6)),
            },
        }
        # prefetch/performance
        cfg["prefetch"] = {
            "thumbs_enabled": bool(getattr(viewer, "_enable_thumb_prefetch", True)),
            "preload_radius": int(getattr(viewer, "_preload_radius", 2)),
            "map_enabled": bool(getattr(viewer, "_enable_map_prefetch", True)),
            "preload_direction": str(getattr(viewer, "_preload_direction", "both")),
            "preload_priority": int(getattr(viewer, "_preload_priority", -1)),
            "preload_max_concurrency": int(getattr(viewer, "_preload_max_concurrency", 0)),
            "preload_retry_count": int(getattr(viewer, "_preload_retry_count", 0)),
            "preload_retry_delay_ms": int(getattr(viewer, "_preload_retry_delay_ms", 0)),
            "only_when_idle": bool(getattr(viewer, "_preload_only_when_idle", False)),
            "prefetch_on_dir_enter": int(getattr(viewer, "_prefetch_on_dir_enter", 0)),
            "slideshow_prefetch_count": int(getattr(viewer, "_slideshow_prefetch_count", 0)),
            "fullres_upgrade_delay_ms": int(getattr(viewer, "_fullres_upgrade_delay_ms", 300)),
            "preview_headroom": float(getattr(viewer, "_preview_headroom", 1.2)),
            "disable_scaled_cache_below_100": bool(getattr(viewer, "_disable_scaled_cache_below_100", False)),
            "preserve_visual_size_on_dpr_change": bool(getattr(viewer, "_preserve_visual_size_on_dpr_change", False)),
        }
        # advanced/cache
        cfg["advanced"] = {
            "image_cache_max_bytes": int(getattr(viewer, "_img_cache_max_bytes", 256*1024*1024)),
            "scaled_cache_max_bytes": int(getattr(viewer, "_scaled_cache_max_bytes", 384*1024*1024)),
            "cache_auto_shrink_pct": int(getattr(viewer, "_cache_auto_shrink_pct", 50)),
            "cache_gc_interval_s": int(getattr(viewer, "_cache_gc_interval_s", 0)),
        }
        # thumb_cache
        cfg["thumb_cache"] = {
            "quality": int(getattr(viewer, "_thumb_cache_quality", 85)),
            "dir": str(getattr(viewer, "_thumb_cache_dir", "")),
        }
        # ai
        # 기존 config.yaml의 키를 보존하기 위해 현재 값이 비어있을 경우 과거 값을 유지한다
        try:
            _prev_yaml = _load_yaml_configs()
            _prev_ai = _prev_yaml.get("ai", {}) if isinstance(_prev_yaml, dict) else {}
            _prev_openai_key = str(_prev_ai.get("openai_api_key", "")) if isinstance(_prev_ai.get("openai_api_key"), str) else ""
        except Exception:
            _prev_openai_key = ""
        cfg["ai"] = {
            "auto_on_open": bool(getattr(viewer, "_auto_ai_on_open", False)),
            "auto_on_drop": bool(getattr(viewer, "_auto_ai_on_drop", False)),
            "auto_on_nav": bool(getattr(viewer, "_auto_ai_on_nav", False)),
            "auto_delay_ms": int(getattr(viewer, "_auto_ai_delay_ms", 0)),
            "skip_if_cached": bool(getattr(viewer, "_ai_skip_if_cached", False)),
            "language": str(getattr(viewer, "_ai_language", "ko")),
            "tone": str(getattr(viewer, "_ai_tone", "중립")),
            "purpose": str(getattr(viewer, "_ai_purpose", "archive")),
            "short_words": int(getattr(viewer, "_ai_short_words", 16)),
            "long_chars": int(getattr(viewer, "_ai_long_chars", 120)),
            "fast_mode": bool(getattr(viewer, "_ai_fast_mode", False)),
            "exif_level": str(getattr(viewer, "_ai_exif_level", "full")),
            "retry_count": int(getattr(viewer, "_ai_retry_count", 2)),
            "retry_delay_ms": int(getattr(viewer, "_ai_retry_delay_ms", 800)),
            "openai_api_key": str(getattr(viewer, "_ai_openai_api_key", "")).strip(),
            "http_timeout_s": float(getattr(getattr(viewer, "_ai_cfg", None), "http_timeout_s", 120.0) if hasattr(viewer, "_ai_cfg") else 120.0),
            # 확장 설정
            "conf_threshold_pct": int(getattr(viewer, "_ai_conf_threshold_pct", 80)),
            "apply_policy": str(getattr(viewer, "_ai_apply_policy", "보류")),
            "batch_workers": int(getattr(viewer, "_ai_batch_workers", 4)),
            "batch_delay_ms": int(getattr(viewer, "_ai_batch_delay_ms", 0)),
            "batch_retry_count": int(getattr(viewer, "_ai_batch_retry_count", 0)),
            "batch_retry_delay_ms": int(getattr(viewer, "_ai_batch_retry_delay_ms", 0)),
            "search_verify_mode_default": str(getattr(viewer, "_search_verify_mode_default", "strict")),
            "search_verify_topn_default": int(getattr(viewer, "_search_verify_topn_default", 20)),
            "search_tag_weight": int(getattr(viewer, "_search_tag_weight", 2)),
            "bg_index_max": int(getattr(viewer, "_bg_index_max", 200)),
            "privacy_hide_location": bool(getattr(viewer, "_privacy_hide_location", False)),
            "offline_mode": bool(getattr(viewer, "_offline_mode", False)),
            # 자연어 검색(정렬/필터/표시/백그라운드)
            "search_sort_order": str(getattr(viewer, "_search_sort_order", "similarity")),
            "search_filter": {
                "min_rating": int(getattr(viewer, "_search_filter_min_rating", 0)),
                "flag_mode": str(getattr(viewer, "_search_filter_flag_mode", "any")),
                "keywords": str(getattr(viewer, "_search_filter_keywords", "")),
                "date_from": str(getattr(viewer, "_search_filter_date_from", "")),
                "date_to": str(getattr(viewer, "_search_filter_date_to", "")),
            },
            "search_result": {
                "thumb_size": int(getattr(viewer, "_search_result_thumb_size", 192)),
                "view_mode": str(getattr(viewer, "_search_result_view_mode", "grid")),
                "show_score": bool(getattr(viewer, "_search_show_score", True)),
                "show_in_filmstrip": bool(getattr(viewer, "_search_show_in_filmstrip", False)),
            },
            "search_bg_prep_enabled": bool(getattr(viewer, "_search_bg_prep_enabled", False)),
            "search_top_k": int(getattr(viewer, "_search_top_k", 80)),
            "verify_model": str(getattr(viewer, "_verify_model", "gpt-5-nano")),
            # 임베딩/검증 고급 옵션
            "use_embedding": bool(getattr(viewer, "_search_use_embedding", True)),
            "verify_strict_only": bool(getattr(viewer, "_search_verify_strict_only", True)),
            "verify_max_candidates": int(getattr(viewer, "_search_verify_max_candidates", 200)),
            "verify_workers": int(getattr(viewer, "_search_verify_workers", 16)),
            "blend_alpha": float(getattr(viewer, "_search_blend_alpha", 0.7)),
            "embed_batch_size": int(getattr(viewer, "_embed_batch_size", 64)),
            "embed_model": str(getattr(viewer, "_embed_model", "text-embedding-3-small")),
        }
        # map/info
        cfg["map"] = {
            "static_provider": str(getattr(viewer, "_map_static_provider", "auto")),
            "preview_size": str(getattr(viewer, "_info_map_size_mode", "medium")),
            "default_zoom": int(getattr(viewer, "_info_map_default_zoom", 12)),
            "cache_max_mb": int(getattr(viewer, "_map_cache_max_mb", 128)),
            "cache_max_days": int(getattr(viewer, "_map_cache_max_days", 30)),
            "api_keys": {
                "kakao": str(getattr(viewer, "_map_kakao_api_key", "")),
                "google": str(getattr(viewer, "_map_google_api_key", "")),
            },
        }
        # info summary
        cfg["info"] = {
            "show": {
                "dt": bool(getattr(viewer, "_info_show_dt", True)),
                "file": bool(getattr(viewer, "_info_show_file", True)),
                "dir": bool(getattr(viewer, "_info_show_dir", True)),
                "cam": bool(getattr(viewer, "_info_show_cam", True)),
                "size": bool(getattr(viewer, "_info_show_size", True)),
                "res": bool(getattr(viewer, "_info_show_res", True)),
                "mp": bool(getattr(viewer, "_info_show_mp", True)),
                "iso": bool(getattr(viewer, "_info_show_iso", True)),
                "focal": bool(getattr(viewer, "_info_show_focal", True)),
                "aperture": bool(getattr(viewer, "_info_show_aperture", True)),
                "shutter": bool(getattr(viewer, "_info_show_shutter", True)),
                "gps": bool(getattr(viewer, "_info_show_gps", True)),
            },
            "max_lines": int(getattr(viewer, "_info_max_lines", 50)),
            "shutter_unit": str(getattr(viewer, "_info_shutter_unit", "auto")),  # auto|sec|frac
        }
    except Exception:
        pass

    # 파일로 저장 (화이트리스트만 내보내기)
    try:
        # 일반 사용자 핵심 항목만 보존
        minimal: Dict[str, Any] = {}
        try:
            openai_key = str(cfg.get("ai", {}).get("openai_api_key", "")) if isinstance(cfg.get("ai", {}), dict) else ""
        except Exception:
            openai_key = ""
        # 항상 키 항목을 노출(빈 문자열 포함)
        minimal.setdefault("ai", {})["openai_api_key"] = str(openai_key or "")
        try:
            api_keys = cfg.get("map", {}).get("api_keys", {}) if isinstance(cfg.get("map", {}), dict) else {}
        except Exception:
            api_keys = {}
        if isinstance(api_keys, dict):
            mk = str(api_keys.get("kakao", "")) if isinstance(api_keys.get("kakao"), str) else ""
            mg = str(api_keys.get("google", "")) if isinstance(api_keys.get("google"), str) else ""
            minimal.setdefault("map", {})["api_keys"] = {"kakao": mk, "google": mg}
        # 기본 기능: 세션 복원/프리페치(thumbs)/자동재생/보기모드/색상 타깃
        try:
            pol = str(getattr(viewer, "_startup_restore_policy", "always"))
            if pol in ("always", "ask", "never"):
                minimal.setdefault("session", {})["startup_restore_policy"] = pol
        except Exception:
            pass
        try:
            minimal.setdefault("prefetch", {})["thumbs_enabled"] = bool(getattr(viewer, "_enable_thumb_prefetch", True))
        except Exception:
            pass
        try:
            minimal.setdefault("anim", {})["autoplay"] = bool(getattr(viewer, "_anim_autoplay", True))
            minimal["anim"]["loop"] = bool(getattr(viewer, "_anim_loop", True))
        except Exception:
            pass
        try:
            dvm = str(getattr(viewer, "_default_view_mode", "fit"))
            if dvm in ("fit", "fit_width", "fit_height", "actual"):
                minimal.setdefault("view", {})["default_view_mode"] = dvm
        except Exception:
            pass
        try:
            tgt = str(getattr(viewer, "_preview_target", "sRGB"))
            if tgt in ("sRGB", "Display P3", "Adobe RGB"):
                minimal.setdefault("color", {})["preview_target"] = tgt
        except Exception:
            pass
        path = _primary_config_path()
        parent_dir = os.path.dirname(path)
        if parent_dir and not os.path.isdir(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(minimal, f, allow_unicode=True, sort_keys=True)
    except Exception:
        pass


def _apply_yaml_settings(viewer, cfg: Dict[str, Any]) -> None:
    # config.yaml 화이트리스트 적용: 일반 사용자 핵심 항목만 허용
    if not isinstance(cfg, dict):
        return
    # AI key
    ai = cfg.get("ai") or {}
    if isinstance(ai, dict) and isinstance(ai.get("openai_api_key"), str):
        viewer._ai_openai_api_key = str(ai.get("openai_api_key"))
    # Map keys
    mp = cfg.get("map") or {}
    if isinstance(mp, dict):
        ak = mp.get("api_keys") or {}
        if isinstance(ak, dict):
            if isinstance(ak.get("kakao"), str):
                viewer._map_kakao_api_key = str(ak.get("kakao"))
            if isinstance(ak.get("google"), str):
                viewer._map_google_api_key = str(ak.get("google"))
    # Open
    op = cfg.get("open") or {}
    if isinstance(op, dict):
        if isinstance(op.get("scan_dir_after_open"), bool):
            viewer._open_scan_dir_after_open = bool(op.get("scan_dir_after_open"))
        if isinstance(op.get("remember_last_dir"), bool):
            viewer._remember_last_open_dir = bool(op.get("remember_last_dir"))
    # Session
    sess = cfg.get("session") or {}
    if isinstance(sess, dict):
        pol = str(sess.get("startup_restore_policy", ""))
        if pol in ("always", "ask", "never"):
            viewer._startup_restore_policy = pol
    # Prefetch
    pf = cfg.get("prefetch") or {}
    if isinstance(pf, dict) and isinstance(pf.get("thumbs_enabled"), bool):
        viewer._enable_thumb_prefetch = bool(pf.get("thumbs_enabled"))
    # Anim
    an = cfg.get("anim") or {}
    if isinstance(an, dict):
        if isinstance(an.get("autoplay"), bool):
            viewer._anim_autoplay = bool(an.get("autoplay"))
        if isinstance(an.get("loop"), bool):
            viewer._anim_loop = bool(an.get("loop"))
    # View
    vw = cfg.get("view") or {}
    if isinstance(vw, dict):
        dvm = str(vw.get("default_view_mode", ""))
        if dvm in ("fit", "fit_width", "fit_height", "actual"):
            viewer._default_view_mode = dvm
    # Color
    col = cfg.get("color") or {}
    if isinstance(col, dict):
        tgt = str(col.get("preview_target", ""))
        if tgt in ("sRGB", "Display P3", "Adobe RGB"):
            viewer._preview_target = tgt


def load_settings(viewer) -> None:
    try:
        # QSettings에 저장된 OpenAI Key 제거(레지스트리 의존 제거)
        try:
            if hasattr(viewer, "settings"):
                viewer.settings.remove("ai/openai_api_key")
        except Exception:
            pass
        # config.yaml 사전 로드(최상단만)
        try:
            _cfg_yaml_cached = _load_yaml_configs()
        except Exception:
            _cfg_yaml_cached = {}
        # QSettings 캐시 초기화: 잘못 저장된 고정/제거 키값 제거
        try:
            if hasattr(viewer, "settings"):
                # reset_to_100 강제 비움
                viewer.settings.remove("keys/custom/reset_to_100")
                # 색상보기 A/B 토글 키 비활성화: 사용자 커스텀 매핑 제거
                viewer.settings.remove("keys/custom/toggle_color_ab")
                # rotate_180에 숫자 '2'가 저장되어 있으면 제거
                try:
                    raw = str(viewer.settings.value("keys/custom/rotate_180", ""))
                    if raw and (raw.strip() == "2" or ";2" in raw or "2;" in raw):
                        # 숫자 키는 평점 용도로 예약, 회전에서 제거
                        viewer.settings.remove("keys/custom/rotate_180")
                except Exception:
                    pass
                # 구 기본 매핑 제거: Q,E,W,H,Shift+H,Shift+V, F(화면맞춤), Ctrl+-을 잘못 저장한 확대, Q를 잘못 저장한 축소, Shift+H를 잘못 저장한 180°
                for k in ("rotate_ccw_90", "rotate_cw_90", "fit_to_width", "fit_to_height", "flip_horizontal", "flip_vertical", "fit_to_window", "zoom_in", "zoom_out", "rotate_180"):
                    try:
                        raw = str(viewer.settings.value(f"keys/custom/{k}", ""))
                    except Exception:
                        raw = ""
                    if not raw:
                        continue
                    # 금지 키 제거
                    parts = [p.strip() for p in raw.split(";") if p.strip()]
                    filtered: list[str] = []
                    for p in parts:
                        # 공통 제거
                        if p in ("Q", "E", "W", "H", "Shift+H", "Shift+V"):
                            continue
                        # 화면 맞춤에 F 제거
                        if k == "fit_to_window" and p == "F":
                            continue
                        # 확대에 Ctrl+- 제거(오입력 방지)
                        if k == "zoom_in" and p == "Ctrl+-":
                            continue
                        # 축소에 Q 제거(오입력 방지)
                        if k == "zoom_out" and p == "Q":
                            continue
                        # 180도 회전에 Shift+H 제거(오입력 방지)
                        if k == "rotate_180" and p == "Shift+H":
                            continue
                        filtered.append(p)
                    parts = filtered
                    viewer.settings.setValue(f"keys/custom/{k}", ";".join(parts))
        except Exception:
            pass
        # --- core sections (refactored) ---
        _load_core_open(viewer)
        _load_core_session_recent(viewer)
        _load_core_anim(viewer)
        _load_core_view_color(viewer)
        _load_core_prefetch(viewer)
        _load_dir_tiff(viewer)
        _load_drag_drop(viewer)
        _load_nav_and_zoom_policy(viewer)
        _load_fullscreen_overlay(viewer)
        _load_view_zoom_details(viewer)
        _load_cache_limits(viewer)
        _load_thumb_cache(viewer)
        _load_ai_automation(viewer)
        _load_ai_defaults(viewer)
        _load_ai_search(viewer)

        s = getattr(viewer, "settings", None)
        # 최근 목록/마지막 경로
        viewer.recent_files = viewer.settings.value("recent/files", [], list)
        viewer.recent_folders = viewer.settings.value("recent/folders", [], list)
        if not isinstance(viewer.recent_files, list):
            viewer.recent_files = []
        if not isinstance(viewer.recent_folders, list):
            viewer.recent_folders = []
        viewer.last_open_dir = viewer.settings.value("recent/last_open_dir", "", str)
        if not isinstance(viewer.last_open_dir, str):
            viewer.last_open_dir = ""
        # 저장 정책/품질
        policy = viewer.settings.value("edit/save_policy", "discard", str)
        viewer._save_policy = policy if policy in ("overwrite", "save_as") else "discard"
        viewer._jpeg_quality = int(_get(s, "edit/jpeg_quality", 95, int)) if s else 95
        # UI 고정값
        viewer._theme = "dark"
        viewer._ui_margins = (5, 5, 5, 5)
        viewer._ui_spacing = 6
        viewer._remember_last_view_mode = True
        # 상태바 상세 표시(선택 항목)
        viewer._statusbar_show_profile_details = bool(_get(s, "status/show_profile_details", False, bool)) if s else False
        # YAML 구성 최종 적용: QSettings에서 읽은 값 위에 덮어써서 사용자가 명시한 config.yaml이 우선
        try:
            if _cfg_yaml_cached:
                _apply_yaml_settings(viewer, _cfg_yaml_cached)
        except Exception:
            pass
        # Info/Map 간소화 로드
        _load_info_panel(viewer)
        _load_map_info(viewer)
        # 강제 기본값: 위치 정보 숨김은 무조건 False(설정/이전 세션과 무관)
        try:
            viewer._privacy_hide_location = False
        except Exception:
            pass
        # config.yaml이 없으면 최소 항목 자동 생성
        try:
            path = _primary_config_path()
            if not os.path.isfile(path):
                # 현재 viewer 상태를 기반으로 최소 구성 생성
                minimal: Dict[str, Any] = {}
                # ai key (항상 포함, 빈 문자열 허용)
                k = str(getattr(viewer, "_ai_openai_api_key", "") or "").strip()
                minimal.setdefault("ai", {})["openai_api_key"] = k
                # map api keys (항상 두 키 포함, 빈 문자열 허용)
                gk = str(getattr(viewer, "_map_google_api_key", "") or "").strip()
                kk = str(getattr(viewer, "_map_kakao_api_key", "") or "").strip()
                minimal.setdefault("map", {}).setdefault("api_keys", {})
                minimal["map"]["api_keys"]["google"] = gk
                minimal["map"]["api_keys"]["kakao"] = kk
                # session
                pol = str(getattr(viewer, "_startup_restore_policy", "always"))
                if pol in ("always", "ask", "never"):
                    minimal.setdefault("session", {})["startup_restore_policy"] = pol
                # prefetch (thumbs only)
                minimal.setdefault("prefetch", {})["thumbs_enabled"] = bool(getattr(viewer, "_enable_thumb_prefetch", True))
                # anim
                minimal.setdefault("anim", {})["autoplay"] = bool(getattr(viewer, "_anim_autoplay", True))
                minimal["anim"]["loop"] = bool(getattr(viewer, "_anim_loop", True))
                # recent 항목은 통합 목록에서 제외
                # view
                dvm = str(getattr(viewer, "_default_view_mode", "fit"))
                if dvm in ("fit", "fit_width", "fit_height", "actual"):
                    minimal.setdefault("view", {})["default_view_mode"] = dvm
                # color
                tgt = str(getattr(viewer, "_preview_target", "sRGB"))
                if tgt in ("sRGB", "Display P3", "Adobe RGB"):
                    minimal.setdefault("color", {})["preview_target"] = tgt
                parent_dir = os.path.dirname(path)
                if parent_dir and not os.path.isdir(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(minimal, f, allow_unicode=True, sort_keys=True)
        except Exception:
            pass
    except Exception:
        viewer.recent_files = []
        viewer.recent_folders = []
        viewer.last_open_dir = ""
        viewer._save_policy = "ask"
        viewer._jpeg_quality = 95
        viewer._theme = "dark"
        viewer._ui_margins = (5, 5, 5, 5)
        viewer._ui_spacing = 6
        viewer._default_view_mode = "fit"
        viewer._remember_last_view_mode = True


def save_settings(viewer) -> None:
    try:
        # QSettings 저장 (모듈)
        _save_recent_session_and_edit(viewer)
        # Open/Animation/Dir/TIFF 저장 (모듈)
        _save_open_and_anim(viewer)
        _save_dir_tiff(viewer)
        # Navigation/Fullscreen/View 저장 (모듈)
        _save_nav_and_zoom_policy(viewer)
        _save_fullscreen_overlay(viewer)
        _save_view_zoom_details(viewer)
        # 색상 관리 저장 (모듈)
        _save_color_management(viewer)
        # 지능형 스케일 프리젠 저장
        viewer.settings.setValue("view/pregen_scales_enabled", bool(getattr(viewer, "_pregen_scales_enabled", False)))
        try:
            txt = ",".join([str(x) for x in (getattr(viewer, "_pregen_scales", [0.25,0.5,1.0,2.0]))])
        except Exception:
            txt = "0.25,0.5,1.0,2.0"
        viewer.settings.setValue("view/pregen_scales", txt)
        # Drag & Drop / 목록 정책 저장 (모듈)
        _save_drag_drop(viewer)
        # Prefetch/성능 및 캐시/썸네일 저장 (모듈)
        _save_prefetch_all(viewer)
        _save_cache_and_thumb(viewer)
        # AI 자동화/기본값 저장 (모듈)
        _save_ai_automation(viewer)
        _save_ai_defaults(viewer)
        # OpenAI Key는 QSettings에 저장하지 않음(최상단 config.yaml만 사용)
        # Map/Info 저장(모듈화)
        _save_map_info(viewer)
        _save_info_panel(viewer)
        # 환경변수 동기화는 하지 않습니다. 모든 동작은 config.yaml/메모리 설정 기준으로 수행합니다.
        # AI 확장/검색/임베딩 저장 (모듈)
        _save_ai_search_and_ext(viewer)
        # 표시/정보 상세
        viewer.settings.setValue("status/show_profile_details", bool(getattr(viewer, "_statusbar_show_profile_details", False)))
        # 즉시 디스크에 동기화하여 크래시 시 손실 방지
        try:
            viewer.settings.sync()
        except Exception:
            pass
        # config.yaml로도 내보내기(없으면 생성)
        try:
            _export_viewer_to_yaml(viewer)
        except Exception:
            pass
    except Exception:
        pass


