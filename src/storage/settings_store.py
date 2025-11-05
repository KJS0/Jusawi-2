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
            "fullres_upgrade_delay_ms": int(getattr(viewer, "_fullres_upgrade_delay_ms", 120)),
            "preview_headroom": float(getattr(viewer, "_preview_headroom", 1.0)),
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
            "openai_api_key": (
                str(getattr(viewer, "_ai_openai_api_key", ""))
                if str(getattr(viewer, "_ai_openai_api_key", "")).strip()
                else _prev_openai_key
            ),
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
            "offline_mode": False,
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

    # 파일로 저장
    try:
        path = _primary_config_path()
        parent_dir = os.path.dirname(path)
        if parent_dir and not os.path.isdir(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=True)
    except Exception:
        pass


def _apply_yaml_settings(viewer, cfg: Dict[str, Any]) -> None:
    # 구조 예시:
    # ui: { theme: dark|light|system, margins: [L,T,R,B], spacing: 6, default_view_mode, remember_last_view_mode }
    # edit: { save_policy: discard|overwrite|save_as, jpeg_quality: 95 }
    # keys: { custom: { cmd_id: ["Ctrl+O"] } }
    try:
        edit = cfg.get("edit", {}) if isinstance(cfg, dict) else {}
        if isinstance(edit, dict):
            pol = edit.get("save_policy")
            if isinstance(pol, str) and pol in ("discard", "overwrite", "save_as"):
                viewer._save_policy = pol
            q = edit.get("jpeg_quality")
            if isinstance(q, int):
                viewer._jpeg_quality = int(q)
        # 단축키 커스텀 키맵: QSettings에 반영하여 기존 경로 재사용
        keys = cfg.get("keys", {}) if isinstance(cfg, dict) else {}
        if isinstance(keys, dict):
            custom = keys.get("custom", {}) if isinstance(keys.get("custom"), dict) else {}
            try:
                from ..shortcuts.shortcuts_manager import save_custom_keymap
            except Exception:
                save_custom_keymap = None  # type: ignore
            if custom and save_custom_keymap and hasattr(viewer, "settings"):
                # 문자열 리스트만 허용, 잘못된 값은 무시
                valid_map: Dict[str, list[str]] = {}
                for cmd_id, arr in custom.items():
                    if isinstance(cmd_id, str) and isinstance(arr, (list, tuple)):
                        seqs = [str(x) for x in arr if isinstance(x, (str, int))]
                        valid_map[cmd_id] = seqs[:1]  # 단일 시퀀스 정책
                save_custom_keymap(viewer.settings, valid_map)
        # 고급 옵션
        adv = cfg.get("advanced", {}) if isinstance(cfg, dict) else {}
        if isinstance(adv, dict):
            if isinstance(adv.get("preload_radius"), int):
                viewer._preload_radius = int(adv.get("preload_radius"))
            if isinstance(adv.get("scale_apply_delay_ms"), int):
                viewer._scale_apply_delay_ms = int(adv.get("scale_apply_delay_ms"))
                try:
                    if hasattr(viewer, "_scale_apply_timer"):
                        viewer._scale_apply_timer.setInterval(int(viewer._scale_apply_delay_ms))
                except Exception:
                    pass
            if isinstance(adv.get("disable_scaled_cache_below_100"), bool):
                viewer._disable_scaled_cache_below_100 = bool(adv.get("disable_scaled_cache_below_100"))
            if isinstance(adv.get("preserve_visual_size_on_dpr_change"), bool):
                viewer._preserve_visual_size_on_dpr_change = bool(adv.get("preserve_visual_size_on_dpr_change"))
            if isinstance(adv.get("convert_movie_frames_to_srgb"), bool):
                viewer._convert_movie_frames_to_srgb = bool(adv.get("convert_movie_frames_to_srgb"))
            if isinstance(adv.get("image_cache_max_bytes"), int):
                viewer._img_cache_max_bytes = int(adv.get("image_cache_max_bytes"))
            if isinstance(adv.get("scaled_cache_max_bytes"), int):
                viewer._scaled_cache_max_bytes = int(adv.get("scaled_cache_max_bytes"))
        # AI 설정
        ai = cfg.get("ai", {}) if isinstance(cfg, dict) else {}
        if isinstance(ai, dict):
            if isinstance(ai.get("auto_on_open"), bool):
                viewer._auto_ai_on_open = bool(ai.get("auto_on_open"))
            if isinstance(ai.get("auto_on_drop"), bool):
                viewer._auto_ai_on_drop = bool(ai.get("auto_on_drop"))
            if isinstance(ai.get("auto_on_nav"), bool):
                viewer._auto_ai_on_nav = bool(ai.get("auto_on_nav"))
            if isinstance(ai.get("auto_delay_ms"), int):
                viewer._auto_ai_delay_ms = int(ai.get("auto_delay_ms"))
            if isinstance(ai.get("skip_if_cached"), bool):
                viewer._ai_skip_if_cached = bool(ai.get("skip_if_cached"))
            if isinstance(ai.get("language"), str):
                lang = str(ai.get("language")).strip().lower()
                viewer._ai_language = "ko" if lang.startswith("ko") else ("en" if lang.startswith("en") else str(getattr(viewer, "_ai_language", "ko")))
            if isinstance(ai.get("tone"), str):
                t = str(ai.get("tone"))
                if t in ("중립", "친근한", "공식적인"):
                    viewer._ai_tone = t
            if isinstance(ai.get("purpose"), str):
                p = str(ai.get("purpose"))
                if p in ("archive", "sns", "blog"):
                    viewer._ai_purpose = p
            if isinstance(ai.get("short_words"), int):
                viewer._ai_short_words = int(ai.get("short_words"))
            if isinstance(ai.get("long_chars"), int):
                viewer._ai_long_chars = int(ai.get("long_chars"))
            if isinstance(ai.get("fast_mode"), bool):
                viewer._ai_fast_mode = bool(ai.get("fast_mode"))
            if isinstance(ai.get("exif_level"), str):
                e = str(ai.get("exif_level"))
                if e in ("full", "summary", "none"):
                    viewer._ai_exif_level = e
            if isinstance(ai.get("retry_count"), int):
                viewer._ai_retry_count = int(ai.get("retry_count"))
            if isinstance(ai.get("retry_delay_ms"), int):
                viewer._ai_retry_delay_ms = int(ai.get("retry_delay_ms"))
            if isinstance(ai.get("openai_api_key"), str):
                viewer._ai_openai_api_key = str(ai.get("openai_api_key"))
            if isinstance(ai.get("verify_model"), str):
                viewer._verify_model = str(ai.get("verify_model"))
            # 확장 로드
            try:
                viewer._ai_conf_threshold_pct = int(ai.get("conf_threshold_pct", 80))
            except Exception:
                pass
            try:
                viewer._ai_apply_policy = str(ai.get("apply_policy", "보류"))
            except Exception:
                pass
            try:
                viewer._ai_batch_workers = int(ai.get("batch_workers", 4))
            except Exception:
                pass
            try:
                viewer._ai_batch_delay_ms = int(ai.get("batch_delay_ms", 0))
            except Exception:
                pass
            try:
                viewer._ai_batch_retry_count = int(ai.get("batch_retry_count", 0))
            except Exception:
                pass
            try:
                viewer._ai_batch_retry_delay_ms = int(ai.get("batch_retry_delay_ms", 0))
            except Exception:
                pass
            try:
                viewer._search_verify_mode_default = str(ai.get("search_verify_mode_default", "strict"))
            except Exception:
                pass
            try:
                viewer._search_verify_topn_default = int(ai.get("search_verify_topn_default", 20))
            except Exception:
                pass
            try:
                viewer._search_tag_weight = int(ai.get("search_tag_weight", 2))
            except Exception:
                pass
            try:
                viewer._bg_index_max = int(ai.get("bg_index_max", 200))
            except Exception:
                pass
            try:
                viewer._privacy_hide_location = bool(ai.get("privacy_hide_location", False))
            except Exception:
                pass
            try:
                viewer._offline_mode = False
            except Exception:
                pass
            # 자연어 검색(정렬/필터/표시/백그라운드)
            try:
                so = ai.get("search_sort_order", "similarity")
                if isinstance(so, str):
                    viewer._search_sort_order = str(so)
            except Exception:
                pass
            try:
                f = ai.get("search_filter", {}) or {}
                if isinstance(f, dict):
                    if isinstance(f.get("min_rating"), int):
                        viewer._search_filter_min_rating = int(f.get("min_rating", 0))
                    if isinstance(f.get("flag_mode"), str):
                        viewer._search_filter_flag_mode = str(f.get("flag_mode", "any"))
                    if isinstance(f.get("keywords"), str):
                        viewer._search_filter_keywords = str(f.get("keywords", ""))
                    if isinstance(f.get("date_from"), str):
                        viewer._search_filter_date_from = str(f.get("date_from", ""))
                    if isinstance(f.get("date_to"), str):
                        viewer._search_filter_date_to = str(f.get("date_to", ""))
            except Exception:
                pass
            try:
                r = ai.get("search_result", {}) or {}
                if isinstance(r, dict):
                    if isinstance(r.get("thumb_size"), int):
                        viewer._search_result_thumb_size = int(r.get("thumb_size", 192))
                    if isinstance(r.get("view_mode"), str):
                        viewer._search_result_view_mode = str(r.get("view_mode", "grid"))
                    if isinstance(r.get("show_score"), bool):
                        viewer._search_show_score = bool(r.get("show_score", True))
                    if isinstance(r.get("show_in_filmstrip"), bool):
                        viewer._search_show_in_filmstrip = bool(r.get("show_in_filmstrip", False))
            except Exception:
                pass
            try:
                viewer._search_bg_prep_enabled = bool(ai.get("search_bg_prep_enabled", False))
            except Exception:
                pass
            # 검색 상위 K (top_k)
            try:
                viewer._search_top_k = int(ai.get("search_top_k", 80))
            except Exception:
                pass
            # HTTP 타임아웃: AIConfig가 있는 경우에만 설정 반영(없으면 무시)
            try:
                if isinstance(ai.get("http_timeout_s"), (int, float)):
                    cfg = getattr(viewer, "_ai_cfg", None)
                    if cfg is not None and hasattr(cfg, "http_timeout_s"):
                        cfg.http_timeout_s = float(ai.get("http_timeout_s"))
            except Exception:
                pass
        # Map/Info 설정
        try:
            mp = cfg.get("map", {}) if isinstance(cfg, dict) else {}
        except Exception:
            mp = {}
        if isinstance(mp, dict):
            try:
                prov = str(mp.get("static_provider", "auto"))
                viewer._map_static_provider = prov
            except Exception:
                pass
        # Info 요약 설정
        try:
            info = cfg.get("info", {}) if isinstance(cfg, dict) else {}
        except Exception:
            info = {}
        if isinstance(info, dict):
            show = info.get("show", {}) if isinstance(info.get("show"), dict) else {}
            try:
                viewer._info_show_dt = bool(show.get("dt", True))
                viewer._info_show_file = bool(show.get("file", True))
                viewer._info_show_dir = bool(show.get("dir", True))
                viewer._info_show_cam = bool(show.get("cam", True))
                viewer._info_show_size = bool(show.get("size", True))
                viewer._info_show_res = bool(show.get("res", True))
                viewer._info_show_mp = bool(show.get("mp", True))
                viewer._info_show_iso = bool(show.get("iso", True))
                viewer._info_show_focal = bool(show.get("focal", True))
                viewer._info_show_aperture = bool(show.get("aperture", True))
                viewer._info_show_shutter = bool(show.get("shutter", True))
                viewer._info_show_gps = bool(show.get("gps", True))
            except Exception:
                pass
            try:
                viewer._info_max_lines = int(info.get("max_lines", 50))
            except Exception:
                pass
            try:
                unit = str(info.get("shutter_unit", "auto") or "auto").lower()
                if unit not in ("auto", "sec", "frac"):
                    unit = "auto"
                viewer._info_shutter_unit = unit
            except Exception:
                pass
            try:
                size = str(mp.get("preview_size", "medium"))
                viewer._info_map_size_mode = size
            except Exception:
                pass
            try:
                viewer._info_map_default_zoom = int(mp.get("default_zoom", 12))
            except Exception:
                pass
            try:
                viewer._map_cache_max_mb = int(mp.get("cache_max_mb", 128))
                viewer._map_cache_max_days = int(mp.get("cache_max_days", 30))
            except Exception:
                pass
            try:
                ak = mp.get("api_keys", {}) if isinstance(mp.get("api_keys"), dict) else {}
                viewer._map_kakao_api_key = str(ak.get("kakao", getattr(viewer, "_map_kakao_api_key", "")))
                viewer._map_google_api_key = str(ak.get("google", getattr(viewer, "_map_google_api_key", "")))
            except Exception:
                pass
    except Exception:
        pass


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
        viewer.recent_files = viewer.settings.value("recent/files", [], list)
        viewer.recent_folders = viewer.settings.value("recent/folders", [], list)
        if not isinstance(viewer.recent_files, list):
            viewer.recent_files = []
        if not isinstance(viewer.recent_folders, list):
            viewer.recent_folders = []
        viewer.last_open_dir = viewer.settings.value("recent/last_open_dir", "", str)
        if not isinstance(viewer.last_open_dir, str):
            viewer.last_open_dir = ""
        # 회전/저장 정책 관련 기본값 로드
        policy = viewer.settings.value("edit/save_policy", "discard", str)
        # 'ask'도 기본적으로 무시하고 'discard'로 동작하도록 강제
        if policy in ("overwrite", "save_as"):
            viewer._save_policy = policy
        else:
            viewer._save_policy = "discard"
        try:
            viewer._jpeg_quality = int(viewer.settings.value("edit/jpeg_quality", 95))
        except Exception:
            viewer._jpeg_quality = 95
        # UI 환경 설정 로드
        # UI 관련 QSettings 제거: 고정값으로 설정
        viewer._theme = "dark"
        viewer._ui_margins = (5, 5, 5, 5)
        viewer._ui_spacing = 6
        try:
            dvm = str(viewer.settings.value("ui/default_view_mode", "fit", str))
            if dvm not in ("fit", "fit_width", "fit_height", "actual"):
                dvm = "fit"
            viewer._default_view_mode = dvm
        except Exception:
            viewer._default_view_mode = "fit"
        viewer._remember_last_view_mode = True
        try:
            viewer._statusbar_show_profile_details = bool(viewer.settings.value("status/show_profile_details", False, bool))
        except Exception:
            viewer._statusbar_show_profile_details = False
        # 색상 관리 기본값 로드
        try:
            viewer._icc_ignore_embedded = bool(viewer.settings.value("color/icc_ignore_embedded", False, bool))
        except Exception:
            viewer._icc_ignore_embedded = False
        try:
            viewer._assumed_colorspace = str(viewer.settings.value("color/assumed_colorspace", "sRGB", str))
        except Exception:
            viewer._assumed_colorspace = "sRGB"
        try:
            viewer._preview_target = str(viewer.settings.value("color/preview_target", "sRGB", str))
        except Exception:
            viewer._preview_target = "sRGB"
        try:
            viewer._fallback_policy = str(viewer.settings.value("color/fallback_policy", "ignore", str))
        except Exception:
            viewer._fallback_policy = "ignore"
        try:
            viewer._convert_movie_frames_to_srgb = bool(viewer.settings.value("color/anim_convert", True, bool))
        except Exception:
            viewer._convert_movie_frames_to_srgb = True
        try:
            viewer._thumb_convert_to_srgb = bool(viewer.settings.value("color/thumb_convert", True, bool))
        except Exception:
            viewer._thumb_convert_to_srgb = True
        # Open/Animation/Dir/TIFF 옵션 기본값 로드
        try:
            viewer._open_scan_dir_after_open = bool(viewer.settings.value("open/scan_dir_after_open", True, bool))
        except Exception:
            viewer._open_scan_dir_after_open = True
        try:
            viewer._remember_last_open_dir = bool(viewer.settings.value("open/remember_last_dir", True, bool))
        except Exception:
            viewer._remember_last_open_dir = True
        # 최근/세션 옵션 로드
        try:
            viewer._startup_restore_policy = str(viewer.settings.value("session/startup_restore_policy", "always", str))
        except Exception:
            viewer._startup_restore_policy = "always"
        try:
            viewer._recent_max_items = int(viewer.settings.value("recent/max_items", 10))
        except Exception:
            viewer._recent_max_items = 10
        # 제외 규칙 삭제됨
        try:
            viewer._recent_auto_prune_missing = bool(viewer.settings.value("recent/auto_prune_missing", True, bool))
        except Exception:
            viewer._recent_auto_prune_missing = True
        try:
            viewer._anim_autoplay = bool(viewer.settings.value("anim/autoplay", True, bool))
        except Exception:
            viewer._anim_autoplay = True
        try:
            viewer._anim_loop = bool(viewer.settings.value("anim/loop", True, bool))
        except Exception:
            viewer._anim_loop = True
        try:
            viewer._anim_keep_state_on_switch = bool(viewer.settings.value("anim/keep_state_on_switch", False, bool))
        except Exception:
            viewer._anim_keep_state_on_switch = False
        try:
            viewer._anim_pause_on_unfocus = bool(viewer.settings.value("anim/pause_on_unfocus", False, bool))
        except Exception:
            viewer._anim_pause_on_unfocus = False
        try:
            viewer._anim_click_toggle = bool(viewer.settings.value("anim/click_toggle", False, bool))
        except Exception:
            viewer._anim_click_toggle = False
        try:
            viewer._anim_overlay_enabled = bool(viewer.settings.value("anim/overlay/enabled", False, bool))
        except Exception:
            viewer._anim_overlay_enabled = False
        try:
            viewer._anim_overlay_show_index = bool(viewer.settings.value("anim/overlay/show_index", True, bool))
        except Exception:
            viewer._anim_overlay_show_index = True
        try:
            viewer._anim_overlay_position = str(viewer.settings.value("anim/overlay/position", "top-right", str))
        except Exception:
            viewer._anim_overlay_position = "top-right"
        try:
            viewer._anim_overlay_opacity = float(viewer.settings.value("anim/overlay/opacity", 0.6))
        except Exception:
            viewer._anim_overlay_opacity = 0.6
        try:
            viewer._dir_sort_mode = viewer.settings.value("dir/sort_mode", "metadata", str)
            if viewer._dir_sort_mode not in ("metadata", "name"):
                viewer._dir_sort_mode = "metadata"
        except Exception:
            viewer._dir_sort_mode = "metadata"
        try:
            viewer._dir_natural_sort = bool(viewer.settings.value("dir/natural_sort", True, bool))
        except Exception:
            viewer._dir_natural_sort = True
        try:
            viewer._dir_exclude_hidden_system = bool(viewer.settings.value("dir/exclude_hidden_system", True, bool))
        except Exception:
            viewer._dir_exclude_hidden_system = True
        try:
            viewer._tiff_open_first_page_only = bool(viewer.settings.value("tiff/open_first_page_only", True, bool))
        except Exception:
            viewer._tiff_open_first_page_only = True
        # Drag & Drop / 목록 정책
        try:
            viewer._drop_allow_folder = bool(viewer.settings.value("drop/allow_folder_drop", False, bool))
        except Exception:
            viewer._drop_allow_folder = False
        try:
            viewer._drop_use_parent_scan = bool(viewer.settings.value("drop/use_parent_scan", True, bool))
        except Exception:
            viewer._drop_use_parent_scan = True
        try:
            viewer._drop_show_overlay = bool(viewer.settings.value("drop/show_progress_overlay", True, bool))
        except Exception:
            viewer._drop_show_overlay = True
        try:
            viewer._drop_confirm_over_threshold = bool(viewer.settings.value("drop/confirm_over_threshold", True, bool))
        except Exception:
            viewer._drop_confirm_over_threshold = True
        try:
            viewer._drop_large_threshold = int(viewer.settings.value("drop/large_drop_threshold", 500))
        except Exception:
            viewer._drop_large_threshold = 500
        # Prefetch/성능
        try:
            viewer._enable_thumb_prefetch = bool(viewer.settings.value("prefetch/thumbs_enabled", True, bool))
        except Exception:
            viewer._enable_thumb_prefetch = True
        try:
            viewer._preload_radius = int(viewer.settings.value("prefetch/preload_radius", 2))
        except Exception:
            viewer._preload_radius = 2
        try:
            viewer._enable_map_prefetch = bool(viewer.settings.value("prefetch/map_enabled", True, bool))
        except Exception:
            viewer._enable_map_prefetch = True
        try:
            viewer._preload_direction = str(viewer.settings.value("prefetch/preload_direction", "both", str))
            if viewer._preload_direction not in ("both", "forward", "backward"):
                viewer._preload_direction = "both"
        except Exception:
            viewer._preload_direction = "both"
        try:
            viewer._preload_priority = int(viewer.settings.value("prefetch/preload_priority", -1))
        except Exception:
            viewer._preload_priority = -1
        try:
            viewer._preload_max_concurrency = int(viewer.settings.value("prefetch/preload_max_concurrency", 0))
        except Exception:
            viewer._preload_max_concurrency = 0
        try:
            viewer._preload_retry_count = int(viewer.settings.value("prefetch/preload_retry_count", 0))
        except Exception:
            viewer._preload_retry_count = 0
        try:
            viewer._preload_retry_delay_ms = int(viewer.settings.value("prefetch/preload_retry_delay_ms", 0))
        except Exception:
            viewer._preload_retry_delay_ms = 0
        try:
            viewer._preload_only_when_idle = bool(viewer.settings.value("prefetch/only_when_idle", False, bool))
        except Exception:
            viewer._preload_only_when_idle = False
        try:
            viewer._prefetch_on_dir_enter = int(viewer.settings.value("prefetch/prefetch_on_dir_enter", 0))
        except Exception:
            viewer._prefetch_on_dir_enter = 0
        try:
            viewer._slideshow_prefetch_count = int(viewer.settings.value("prefetch/slideshow_prefetch_count", 0))
        except Exception:
            viewer._slideshow_prefetch_count = 0
        try:
            viewer._preload_direction = str(viewer.settings.value("prefetch/preload_direction", "both", str))
            if viewer._preload_direction not in ("both", "forward", "backward"):
                viewer._preload_direction = "both"
        except Exception:
            viewer._preload_direction = "both"
        try:
            viewer._preload_priority = int(viewer.settings.value("prefetch/preload_priority", -1))
        except Exception:
            viewer._preload_priority = -1
        try:
            viewer._preload_max_concurrency = int(viewer.settings.value("prefetch/preload_max_concurrency", 0))
        except Exception:
            viewer._preload_max_concurrency = 0
        # 자동화
        try:
            viewer._auto_ai_on_open = bool(viewer.settings.value("ai/auto_on_open", False, bool))
        except Exception:
            viewer._auto_ai_on_open = False
        try:
            viewer._auto_ai_on_drop = bool(viewer.settings.value("ai/auto_on_drop", False, bool))
        except Exception:
            viewer._auto_ai_on_drop = False
        try:
            viewer._auto_ai_on_nav = bool(viewer.settings.value("ai/auto_on_nav", False, bool))
        except Exception:
            viewer._auto_ai_on_nav = False
        try:
            viewer._auto_ai_delay_ms = int(viewer.settings.value("ai/auto_delay_ms", 0))
        except Exception:
            viewer._auto_ai_delay_ms = 0
        try:
            viewer._ai_skip_if_cached = bool(viewer.settings.value("ai/skip_if_cached", False, bool))
        except Exception:
            viewer._ai_skip_if_cached = False
        # AI 기본값
        try:
            viewer._ai_language = str(viewer.settings.value("ai/language", "ko", str))
        except Exception:
            viewer._ai_language = "ko"
        try:
            viewer._ai_tone = str(viewer.settings.value("ai/tone", "중립", str))
        except Exception:
            viewer._ai_tone = "중립"
        try:
            viewer._ai_purpose = str(viewer.settings.value("ai/purpose", "archive", str))
        except Exception:
            viewer._ai_purpose = "archive"
        try:
            viewer._ai_short_words = int(viewer.settings.value("ai/short_words", 16))
        except Exception:
            viewer._ai_short_words = 16
        try:
            viewer._ai_long_chars = int(viewer.settings.value("ai/long_chars", 120))
        except Exception:
            viewer._ai_long_chars = 120
        try:
            viewer._ai_fast_mode = bool(viewer.settings.value("ai/fast_mode", False, bool))
        except Exception:
            viewer._ai_fast_mode = False
        try:
            viewer._ai_exif_level = str(viewer.settings.value("ai/exif_level", "full", str))
        except Exception:
            viewer._ai_exif_level = "full"
        try:
            viewer._ai_retry_count = int(viewer.settings.value("ai/retry_count", 2))
        except Exception:
            viewer._ai_retry_count = 2
        try:
            viewer._ai_retry_delay_ms = int(viewer.settings.value("ai/retry_delay_ms", 800))
        except Exception:
            viewer._ai_retry_delay_ms = 800
        # OpenAI Key는 QSettings를 무시하고 YAML만 사용
        try:
            pass
        except Exception:
            pass
        # 확장 로드(QSettings)
        try:
            viewer._ai_conf_threshold_pct = int(viewer.settings.value("ai/conf_threshold_pct", 80))
        except Exception:
            viewer._ai_conf_threshold_pct = 80
        try:
            viewer._ai_apply_policy = str(viewer.settings.value("ai/apply_policy", "보류", str))
        except Exception:
            viewer._ai_apply_policy = "보류"
        try:
            viewer._ai_batch_workers = int(viewer.settings.value("ai/batch_workers", 4))
        except Exception:
            viewer._ai_batch_workers = 4
        try:
            viewer._ai_batch_delay_ms = int(viewer.settings.value("ai/batch_delay_ms", 0))
        except Exception:
            viewer._ai_batch_delay_ms = 0
        try:
            viewer._ai_batch_retry_count = int(viewer.settings.value("ai/batch_retry_count", 0))
        except Exception:
            viewer._ai_batch_retry_count = 0
        try:
            viewer._ai_batch_retry_delay_ms = int(viewer.settings.value("ai/batch_retry_delay_ms", 0))
        except Exception:
            viewer._ai_batch_retry_delay_ms = 0
        try:
            viewer._search_verify_mode_default = str(viewer.settings.value("ai/search_verify_mode_default", "strict", str))
        except Exception:
            viewer._search_verify_mode_default = "strict"
        try:
            viewer._search_verify_topn_default = int(viewer.settings.value("ai/search_verify_topn_default", 20))
        except Exception:
            viewer._search_verify_topn_default = 20
        # 자연어 검색 정렬/필터/표시/백그라운드
        try:
            viewer._search_sort_order = str(viewer.settings.value("ai/search_sort_order", "similarity", str))
        except Exception:
            viewer._search_sort_order = "similarity"
        try:
            viewer._search_filter_min_rating = int(viewer.settings.value("ai/search_filter/min_rating", 0))
        except Exception:
            viewer._search_filter_min_rating = 0
        try:
            viewer._search_filter_flag_mode = str(viewer.settings.value("ai/search_filter/flag_mode", "any", str))
        except Exception:
            viewer._search_filter_flag_mode = "any"
        try:
            viewer._search_filter_keywords = str(viewer.settings.value("ai/search_filter/keywords", "", str))
        except Exception:
            viewer._search_filter_keywords = ""
        try:
            viewer._search_filter_date_from = str(viewer.settings.value("ai/search_filter/date_from", "", str))
        except Exception:
            viewer._search_filter_date_from = ""
        try:
            viewer._search_filter_date_to = str(viewer.settings.value("ai/search_filter/date_to", "", str))
        except Exception:
            viewer._search_filter_date_to = ""
        try:
            viewer._search_result_thumb_size = int(viewer.settings.value("ai/search_result/thumb_size", 192))
        except Exception:
            viewer._search_result_thumb_size = 192
        try:
            viewer._search_result_view_mode = str(viewer.settings.value("ai/search_result/view_mode", "grid", str))
        except Exception:
            viewer._search_result_view_mode = "grid"
        try:
            viewer._search_show_score = bool(viewer.settings.value("ai/search_result/show_score", True, bool))
        except Exception:
            viewer._search_show_score = True
        try:
            viewer._search_show_in_filmstrip = bool(viewer.settings.value("ai/search_result/show_in_filmstrip", False, bool))
        except Exception:
            viewer._search_show_in_filmstrip = False
        try:
            viewer._search_bg_prep_enabled = bool(viewer.settings.value("ai/search/bg_prep_enabled", False, bool))
        except Exception:
            viewer._search_bg_prep_enabled = False
        # 임베딩/검증 고급 옵션
        try:
            viewer._search_use_embedding = bool(viewer.settings.value("ai/search/use_embedding", True, bool))
        except Exception:
            viewer._search_use_embedding = True
        try:
            viewer._search_verify_strict_only = bool(viewer.settings.value("ai/search/verify_strict_only", True, bool))
        except Exception:
            viewer._search_verify_strict_only = True
        try:
            viewer._search_verify_max_candidates = int(viewer.settings.value("ai/search/verify_max_candidates", 200))
        except Exception:
            viewer._search_verify_max_candidates = 200
        try:
            viewer._search_verify_workers = int(viewer.settings.value("ai/search/verify_workers", 16))
        except Exception:
            viewer._search_verify_workers = 16
        try:
            viewer._search_blend_alpha = float(viewer.settings.value("ai/search/blend_alpha", 0.7))
        except Exception:
            viewer._search_blend_alpha = 0.7
        try:
            viewer._embed_batch_size = int(viewer.settings.value("ai/embed/batch_size", 64))
        except Exception:
            viewer._embed_batch_size = 64
        try:
            viewer._embed_model = str(viewer.settings.value("ai/embed/model", "text-embedding-3-small", str))
        except Exception:
            viewer._embed_model = "text-embedding-3-small"
        try:
            viewer._search_tag_weight = int(viewer.settings.value("ai/search_tag_weight", 2))
        except Exception:
            viewer._search_tag_weight = 2
        try:
            viewer._bg_index_max = int(viewer.settings.value("ai/bg_index_max", 200))
        except Exception:
            viewer._bg_index_max = 200
        try:
            viewer._privacy_hide_location = bool(viewer.settings.value("ai/privacy_hide_location", False, bool))
        except Exception:
            viewer._privacy_hide_location = False
        try:
            viewer._offline_mode = False
        except Exception:
            viewer._offline_mode = False
        # Navigation/Filmstrip/Zoom 정책 추가
        try:
            viewer._nav_wrap_ends = bool(viewer.settings.value("nav/wrap_ends", False, bool))
        except Exception:
            viewer._nav_wrap_ends = False
        try:
            viewer._nav_min_interval_ms = int(viewer.settings.value("nav/min_interval_ms", 100))
        except Exception:
            viewer._nav_min_interval_ms = 100
        try:
            viewer._filmstrip_auto_center = bool(viewer.settings.value("ui/filmstrip_auto_center", True, bool))
        except Exception:
            viewer._filmstrip_auto_center = True
        # 우선 정렬 키는 제거(메타데이터/파일명만 유지)
        try:
            viewer._zoom_policy = str(viewer.settings.value("view/zoom_policy", "mode", str))
            if viewer._zoom_policy not in ("reset", "mode", "scale"):
                viewer._zoom_policy = "mode"
        except Exception:
            viewer._zoom_policy = "mode"
        # 전체화면/오버레이 관련
        try:
            viewer._fs_auto_hide_ms = int(viewer.settings.value("fullscreen/auto_hide_ms", 1500))
        except Exception:
            viewer._fs_auto_hide_ms = 1500
        try:
            viewer._fs_auto_hide_cursor_ms = int(viewer.settings.value("fullscreen/auto_hide_cursor_ms", 1200))
        except Exception:
            viewer._fs_auto_hide_cursor_ms = 1200
        try:
            viewer._fs_enter_view_mode = str(viewer.settings.value("fullscreen/enter_view_mode", "keep", str))
            if viewer._fs_enter_view_mode not in ("keep", "fit", "fit_width", "fit_height", "actual"):
                viewer._fs_enter_view_mode = "keep"
        except Exception:
            viewer._fs_enter_view_mode = "keep"
        try:
            viewer._fs_show_filmstrip_overlay = bool(viewer.settings.value("fullscreen/show_filmstrip_overlay", False, bool))
        except Exception:
            viewer._fs_show_filmstrip_overlay = False
        try:
            viewer._fs_safe_exit = bool(viewer.settings.value("fullscreen/safe_exit_rule", True, bool))
        except Exception:
            viewer._fs_safe_exit = True
        try:
            viewer._overlay_enabled_default = bool(viewer.settings.value("overlay/enabled_default", False, bool))
        except Exception:
            viewer._overlay_enabled_default = False
        try:
            viewer._smooth_transform = bool(viewer.settings.value("view/smooth_transform", True, bool))
        except Exception:
            viewer._smooth_transform = True
        # 보기/줌 고급 옵션
        # 보기 공유 옵션 제거됨
        try:
            viewer._refit_on_transform = bool(viewer.settings.value("view/refit_on_transform", True, bool))
        except Exception:
            viewer._refit_on_transform = True
        # 회전 시 화면 중심 앵커 유지 옵션(기본 True)
        try:
            viewer._anchor_preserve_on_transform = bool(viewer.settings.value("view/anchor_preserve_on_transform", True, bool))
        except Exception:
            viewer._anchor_preserve_on_transform = True
        try:
            viewer._fit_margin_pct = int(viewer.settings.value("view/fit_margin_pct", 0))
        except Exception:
            viewer._fit_margin_pct = 0
        try:
            viewer._wheel_zoom_requires_ctrl = bool(viewer.settings.value("view/wheel_zoom_requires_ctrl", True, bool))
        except Exception:
            viewer._wheel_zoom_requires_ctrl = True
        try:
            viewer._wheel_zoom_alt_precise = bool(viewer.settings.value("view/wheel_zoom_alt_precise", True, bool))
        except Exception:
            viewer._wheel_zoom_alt_precise = True
        try:
            viewer._use_fixed_zoom_steps = bool(viewer.settings.value("view/use_fixed_zoom_steps", False, bool))
        except Exception:
            viewer._use_fixed_zoom_steps = False
        try:
            viewer._zoom_step_factor = float(viewer.settings.value("view/zoom_step_factor", 1.25))
        except Exception:
            viewer._zoom_step_factor = 1.25
        try:
            viewer._precise_zoom_step_factor = float(viewer.settings.value("view/precise_zoom_step_factor", 1.1))
        except Exception:
            viewer._precise_zoom_step_factor = 1.1
        try:
            viewer._double_click_action = str(viewer.settings.value("view/double_click_action", "toggle", str))
        except Exception:
            viewer._double_click_action = 'toggle'
        try:
            viewer._middle_click_action = str(viewer.settings.value("view/middle_click_action", "none", str))
        except Exception:
            viewer._middle_click_action = 'none'
        try:
            viewer._preserve_visual_size_on_dpr_change = bool(viewer.settings.value("view/preserve_visual_size_on_dpr_change", False, bool))
        except Exception:
            pass
        # 지능형 스케일 프리젠
        try:
            viewer._pregen_scales_enabled = bool(viewer.settings.value("view/pregen_scales_enabled", False, bool))
        except Exception:
            viewer._pregen_scales_enabled = False
        try:
            raw = str(viewer.settings.value("view/pregen_scales", "0.25,0.5,1.0,2.0", str))
            arr: list[float] = []
            for p in [t.strip() for t in raw.split(',') if t.strip()]:
                try:
                    arr.append(float(p))
                except Exception:
                    pass
            viewer._pregen_scales = arr if arr else [0.25, 0.5, 1.0, 2.0]
        except Exception:
            viewer._pregen_scales = [0.25, 0.5, 1.0, 2.0]
        # 고급 캐시 로드
        try:
            viewer._img_cache_max_bytes = int(viewer.settings.value("advanced/image_cache_max_bytes", 256*1024*1024))
        except Exception:
            viewer._img_cache_max_bytes = 256*1024*1024
        try:
            viewer._scaled_cache_max_bytes = int(viewer.settings.value("advanced/scaled_cache_max_bytes", 384*1024*1024))
        except Exception:
            viewer._scaled_cache_max_bytes = 384*1024*1024
        try:
            viewer._cache_auto_shrink_pct = int(viewer.settings.value("advanced/cache_auto_shrink_pct", 50))
        except Exception:
            viewer._cache_auto_shrink_pct = 50
        try:
            viewer._cache_gc_interval_s = int(viewer.settings.value("advanced/cache_gc_interval_s", 0))
        except Exception:
            viewer._cache_gc_interval_s = 0
        # 썸네일 캐시 설정 로드
        try:
            viewer._thumb_cache_quality = int(viewer.settings.value("thumb_cache/quality", 85))
        except Exception:
            viewer._thumb_cache_quality = 85
        try:
            viewer._thumb_cache_dir = str(viewer.settings.value("thumb_cache/dir", "", str))
        except Exception:
            viewer._thumb_cache_dir = ""
        # YAML 구성 최종 적용: QSettings에서 읽은 값 위에 덮어써서 사용자가 명시한 config.yaml이 우선
        try:
            if _cfg_yaml_cached:
                _apply_yaml_settings(viewer, _cfg_yaml_cached)
        except Exception:
            pass
        # Info 요약(QSettings)
        try:
            viewer._info_show_dt = bool(viewer.settings.value("info/show_dt", getattr(viewer, "_info_show_dt", True), bool))
            viewer._info_show_file = bool(viewer.settings.value("info/show_file", getattr(viewer, "_info_show_file", True), bool))
            viewer._info_show_dir = bool(viewer.settings.value("info/show_dir", getattr(viewer, "_info_show_dir", True), bool))
            viewer._info_show_cam = bool(viewer.settings.value("info/show_cam", getattr(viewer, "_info_show_cam", True), bool))
            viewer._info_show_size = bool(viewer.settings.value("info/show_size", getattr(viewer, "_info_show_size", True), bool))
            viewer._info_show_res = bool(viewer.settings.value("info/show_res", getattr(viewer, "_info_show_res", True), bool))
            viewer._info_show_mp = bool(viewer.settings.value("info/show_mp", getattr(viewer, "_info_show_mp", True), bool))
            viewer._info_show_iso = bool(viewer.settings.value("info/show_iso", getattr(viewer, "_info_show_iso", True), bool))
            viewer._info_show_focal = bool(viewer.settings.value("info/show_focal", getattr(viewer, "_info_show_focal", True), bool))
            viewer._info_show_aperture = bool(viewer.settings.value("info/show_aperture", getattr(viewer, "_info_show_aperture", True), bool))
            viewer._info_show_shutter = bool(viewer.settings.value("info/show_shutter", getattr(viewer, "_info_show_shutter", True), bool))
            viewer._info_show_gps = bool(viewer.settings.value("info/show_gps", getattr(viewer, "_info_show_gps", True), bool))
            viewer._info_max_lines = int(viewer.settings.value("info/max_lines", getattr(viewer, "_info_max_lines", 50)))
            viewer._info_shutter_unit = str(viewer.settings.value("info/shutter_unit", getattr(viewer, "_info_shutter_unit", "auto"), str))
        except Exception:
            pass
        # Map/Info: QSettings → 뷰어/환경 반영
        try:
            viewer._map_static_provider = str(viewer.settings.value("map/static_provider", getattr(viewer, "_map_static_provider", "auto"), str))
        except Exception:
            viewer._map_static_provider = getattr(viewer, "_map_static_provider", "auto")
        try:
            viewer._info_map_size_mode = str(viewer.settings.value("map/preview_size", getattr(viewer, "_info_map_size_mode", "medium"), str))
        except Exception:
            viewer._info_map_size_mode = getattr(viewer, "_info_map_size_mode", "medium")
        try:
            viewer._info_map_default_zoom = int(viewer.settings.value("map/default_zoom", getattr(viewer, "_info_map_default_zoom", 12)))
        except Exception:
            viewer._info_map_default_zoom = getattr(viewer, "_info_map_default_zoom", 12)
        try:
            viewer._map_cache_max_mb = int(viewer.settings.value("map/cache_max_mb", getattr(viewer, "_map_cache_max_mb", 128)))
        except Exception:
            viewer._map_cache_max_mb = getattr(viewer, "_map_cache_max_mb", 128)
        try:
            viewer._map_cache_max_days = int(viewer.settings.value("map/cache_max_days", getattr(viewer, "_map_cache_max_days", 30)))
        except Exception:
            viewer._map_cache_max_days = getattr(viewer, "_map_cache_max_days", 30)
        # API 키(QSettings; 평문 저장)
        try:
            viewer._map_kakao_api_key = str(viewer.settings.value("map/kakao_api_key", "", str))
        except Exception:
            viewer._map_kakao_api_key = ""
        try:
            viewer._map_google_api_key = str(viewer.settings.value("map/google_api_key", "", str))
        except Exception:
            viewer._map_google_api_key = ""
        # 지도 API 키(config.yaml에서 우선)
        try:
            mp = _load_yaml_configs().get("map", {}) if isinstance(_load_yaml_configs(), dict) else {}
        except Exception:
            mp = {}
        try:
            ak = mp.get("api_keys", {}) if isinstance(mp.get("api_keys"), dict) else {}
        except Exception:
            ak = {}
        try:
            viewer._map_kakao_api_key = str(ak.get("kakao", str(getattr(viewer, "_map_kakao_api_key", ""))))
        except Exception:
            viewer._map_kakao_api_key = str(getattr(viewer, "_map_kakao_api_key", ""))
        try:
            viewer._map_google_api_key = str(ak.get("google", str(getattr(viewer, "_map_google_api_key", ""))))
        except Exception:
            viewer._map_google_api_key = str(getattr(viewer, "_map_google_api_key", ""))
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
        # QSettings 저장
        viewer.settings.setValue("recent/files", viewer.recent_files)
        viewer.settings.setValue("recent/folders", viewer.recent_folders)
        viewer.settings.setValue("recent/last_open_dir", viewer.last_open_dir)
        # 최근/세션 옵션 저장
        viewer.settings.setValue("session/startup_restore_policy", str(getattr(viewer, "_startup_restore_policy", "always")))
        viewer.settings.setValue("recent/max_items", int(getattr(viewer, "_recent_max_items", 10)))
        # 제외 규칙 저장 없음
        viewer.settings.setValue("recent/auto_prune_missing", bool(getattr(viewer, "_recent_auto_prune_missing", True)))
        viewer.settings.setValue("edit/save_policy", getattr(viewer, "_save_policy", "discard"))
        viewer.settings.setValue("edit/jpeg_quality", int(getattr(viewer, "_jpeg_quality", 95)))
        # Open/Animation/Dir/TIFF 옵션 저장
        viewer.settings.setValue("open/scan_dir_after_open", bool(getattr(viewer, "_open_scan_dir_after_open", True)))
        viewer.settings.setValue("open/remember_last_dir", bool(getattr(viewer, "_remember_last_open_dir", True)))
        viewer.settings.setValue("anim/autoplay", bool(getattr(viewer, "_anim_autoplay", True)))
        viewer.settings.setValue("anim/loop", bool(getattr(viewer, "_anim_loop", True)))
        viewer.settings.setValue("anim/keep_state_on_switch", bool(getattr(viewer, "_anim_keep_state_on_switch", False)))
        viewer.settings.setValue("anim/pause_on_unfocus", bool(getattr(viewer, "_anim_pause_on_unfocus", False)))
        viewer.settings.setValue("anim/click_toggle", bool(getattr(viewer, "_anim_click_toggle", False)))
        viewer.settings.setValue("anim/overlay/enabled", bool(getattr(viewer, "_anim_overlay_enabled", False)))
        viewer.settings.setValue("anim/overlay/show_index", bool(getattr(viewer, "_anim_overlay_show_index", True)))
        viewer.settings.setValue("anim/overlay/position", str(getattr(viewer, "_anim_overlay_position", "top-right")))
        viewer.settings.setValue("anim/overlay/opacity", float(getattr(viewer, "_anim_overlay_opacity", 0.6)))
        viewer.settings.setValue("dir/sort_mode", str(getattr(viewer, "_dir_sort_mode", "metadata")))
        viewer.settings.setValue("dir/natural_sort", bool(getattr(viewer, "_dir_natural_sort", True)))
        viewer.settings.setValue("dir/exclude_hidden_system", bool(getattr(viewer, "_dir_exclude_hidden_system", True)))
        viewer.settings.setValue("tiff/open_first_page_only", bool(getattr(viewer, "_tiff_open_first_page_only", True)))
        # Navigation/Filmstrip/Zoom 정책 저장
        viewer.settings.setValue("nav/wrap_ends", bool(getattr(viewer, "_nav_wrap_ends", False)))
        viewer.settings.setValue("nav/min_interval_ms", int(getattr(viewer, "_nav_min_interval_ms", 100)))
        viewer.settings.setValue("ui/filmstrip_auto_center", bool(getattr(viewer, "_filmstrip_auto_center", True)))
        # dir/sort_primary 저장 제거
        viewer.settings.setValue("view/zoom_policy", str(getattr(viewer, "_zoom_policy", "mode")))
        # 전체화면/오버레이 관련 저장
        viewer.settings.setValue("fullscreen/auto_hide_ms", int(getattr(viewer, "_fs_auto_hide_ms", 1500)))
        viewer.settings.setValue("fullscreen/auto_hide_cursor_ms", int(getattr(viewer, "_fs_auto_hide_cursor_ms", 1200)))
        viewer.settings.setValue("fullscreen/enter_view_mode", str(getattr(viewer, "_fs_enter_view_mode", "keep")))
        viewer.settings.setValue("fullscreen/show_filmstrip_overlay", bool(getattr(viewer, "_fs_show_filmstrip_overlay", False)))
        viewer.settings.setValue("fullscreen/safe_exit_rule", bool(getattr(viewer, "_fs_safe_exit", True)))
        viewer.settings.setValue("overlay/enabled_default", bool(getattr(viewer, "_overlay_enabled_default", False)))
        # 기본 보기 모드 및 스무딩 저장
        viewer.settings.setValue("ui/default_view_mode", str(getattr(viewer, "_default_view_mode", "fit")))
        viewer.settings.setValue("view/smooth_transform", bool(getattr(viewer, "_smooth_transform", True)))
        # 보기/줌 고급 옵션 저장
        # 보기 공유 저장 제거
        viewer.settings.setValue("view/refit_on_transform", bool(getattr(viewer, "_refit_on_transform", True)))
        viewer.settings.setValue("view/anchor_preserve_on_transform", bool(getattr(viewer, "_anchor_preserve_on_transform", True)))
        viewer.settings.setValue("view/fit_margin_pct", int(getattr(viewer, "_fit_margin_pct", 0)))
        viewer.settings.setValue("view/wheel_zoom_requires_ctrl", bool(getattr(viewer, "_wheel_zoom_requires_ctrl", True)))
        viewer.settings.setValue("view/wheel_zoom_alt_precise", bool(getattr(viewer, "_wheel_zoom_alt_precise", True)))
        viewer.settings.setValue("view/use_fixed_zoom_steps", bool(getattr(viewer, "_use_fixed_zoom_steps", False)))
        viewer.settings.setValue("view/zoom_step_factor", float(getattr(viewer, "_zoom_step_factor", 1.25)))
        viewer.settings.setValue("view/precise_zoom_step_factor", float(getattr(viewer, "_precise_zoom_step_factor", 1.1)))
        viewer.settings.setValue("view/double_click_action", str(getattr(viewer, "_double_click_action", 'toggle')))
        viewer.settings.setValue("view/middle_click_action", str(getattr(viewer, "_middle_click_action", 'none')))
        viewer.settings.setValue("view/preserve_visual_size_on_dpr_change", bool(getattr(viewer, "_preserve_visual_size_on_dpr_change", False)))
        # 색상 관리 저장
        viewer.settings.setValue("color/icc_ignore_embedded", bool(getattr(viewer, "_icc_ignore_embedded", False)))
        viewer.settings.setValue("color/assumed_colorspace", str(getattr(viewer, "_assumed_colorspace", "sRGB")))
        viewer.settings.setValue("color/preview_target", str(getattr(viewer, "_preview_target", "sRGB")))
        viewer.settings.setValue("color/fallback_policy", str(getattr(viewer, "_fallback_policy", "ignore")))
        viewer.settings.setValue("color/anim_convert", bool(getattr(viewer, "_convert_movie_frames_to_srgb", True)))
        viewer.settings.setValue("color/thumb_convert", bool(getattr(viewer, "_thumb_convert_to_srgb", True)))
        # 지능형 스케일 프리젠 저장
        viewer.settings.setValue("view/pregen_scales_enabled", bool(getattr(viewer, "_pregen_scales_enabled", False)))
        try:
            txt = ",".join([str(x) for x in (getattr(viewer, "_pregen_scales", [0.25,0.5,1.0,2.0]))])
        except Exception:
            txt = "0.25,0.5,1.0,2.0"
        viewer.settings.setValue("view/pregen_scales", txt)
        # Drag & Drop / 목록 정책 저장
        viewer.settings.setValue("drop/allow_folder_drop", bool(getattr(viewer, "_drop_allow_folder", False)))
        viewer.settings.setValue("drop/use_parent_scan", bool(getattr(viewer, "_drop_use_parent_scan", True)))
        viewer.settings.setValue("drop/show_progress_overlay", bool(getattr(viewer, "_drop_show_overlay", True)))
        viewer.settings.setValue("drop/confirm_over_threshold", bool(getattr(viewer, "_drop_confirm_over_threshold", True)))
        viewer.settings.setValue("drop/large_drop_threshold", int(getattr(viewer, "_drop_large_threshold", 500)))
        # Prefetch/성능 저장
        viewer.settings.setValue("prefetch/thumbs_enabled", bool(getattr(viewer, "_enable_thumb_prefetch", True)))
        viewer.settings.setValue("prefetch/preload_radius", int(getattr(viewer, "_preload_radius", 2)))
        viewer.settings.setValue("prefetch/map_enabled", bool(getattr(viewer, "_enable_map_prefetch", True)))
        viewer.settings.setValue("prefetch/preload_direction", str(getattr(viewer, "_preload_direction", "both")))
        viewer.settings.setValue("prefetch/preload_priority", int(getattr(viewer, "_preload_priority", -1)))
        viewer.settings.setValue("prefetch/preload_max_concurrency", int(getattr(viewer, "_preload_max_concurrency", 0)))
        viewer.settings.setValue("prefetch/preload_retry_count", int(getattr(viewer, "_preload_retry_count", 0)))
        viewer.settings.setValue("prefetch/preload_retry_delay_ms", int(getattr(viewer, "_preload_retry_delay_ms", 0)))
        viewer.settings.setValue("prefetch/only_when_idle", bool(getattr(viewer, "_preload_only_when_idle", False)))
        viewer.settings.setValue("prefetch/prefetch_on_dir_enter", int(getattr(viewer, "_prefetch_on_dir_enter", 0)))
        viewer.settings.setValue("prefetch/slideshow_prefetch_count", int(getattr(viewer, "_slideshow_prefetch_count", 0)))
        # 고급 캐시 상한 저장
        viewer.settings.setValue("advanced/image_cache_max_bytes", int(getattr(viewer, "_img_cache_max_bytes", 256*1024*1024)))
        viewer.settings.setValue("advanced/scaled_cache_max_bytes", int(getattr(viewer, "_scaled_cache_max_bytes", 384*1024*1024)))
        viewer.settings.setValue("advanced/cache_auto_shrink_pct", int(getattr(viewer, "_cache_auto_shrink_pct", 50)))
        viewer.settings.setValue("advanced/cache_gc_interval_s", int(getattr(viewer, "_cache_gc_interval_s", 0)))
        # 썸네일 캐시 저장
        viewer.settings.setValue("thumb_cache/quality", int(getattr(viewer, "_thumb_cache_quality", 85)))
        viewer.settings.setValue("thumb_cache/dir", str(getattr(viewer, "_thumb_cache_dir", "")))
        viewer.settings.setValue("prefetch/preload_direction", str(getattr(viewer, "_preload_direction", "both")))
        viewer.settings.setValue("prefetch/preload_priority", int(getattr(viewer, "_preload_priority", -1)))
        viewer.settings.setValue("prefetch/preload_max_concurrency", int(getattr(viewer, "_preload_max_concurrency", 0)))
        # 자동화 저장
        viewer.settings.setValue("ai/auto_on_open", bool(getattr(viewer, "_auto_ai_on_open", False)))
        viewer.settings.setValue("ai/auto_on_drop", bool(getattr(viewer, "_auto_ai_on_drop", False)))
        viewer.settings.setValue("ai/auto_on_nav", bool(getattr(viewer, "_auto_ai_on_nav", False)))
        viewer.settings.setValue("ai/auto_delay_ms", int(getattr(viewer, "_auto_ai_delay_ms", 0)))
        viewer.settings.setValue("ai/skip_if_cached", bool(getattr(viewer, "_ai_skip_if_cached", False)))
        # AI 기본값 저장
        viewer.settings.setValue("ai/language", str(getattr(viewer, "_ai_language", "ko")))
        viewer.settings.setValue("ai/tone", str(getattr(viewer, "_ai_tone", "중립")))
        viewer.settings.setValue("ai/purpose", str(getattr(viewer, "_ai_purpose", "archive")))
        viewer.settings.setValue("ai/short_words", int(getattr(viewer, "_ai_short_words", 16)))
        viewer.settings.setValue("ai/long_chars", int(getattr(viewer, "_ai_long_chars", 120)))
        viewer.settings.setValue("ai/fast_mode", bool(getattr(viewer, "_ai_fast_mode", False)))
        viewer.settings.setValue("ai/exif_level", str(getattr(viewer, "_ai_exif_level", "full")))
        viewer.settings.setValue("ai/retry_count", int(getattr(viewer, "_ai_retry_count", 2)))
        viewer.settings.setValue("ai/retry_delay_ms", int(getattr(viewer, "_ai_retry_delay_ms", 800)))
        # OpenAI Key는 QSettings에 저장하지 않음(최상단 config.yaml만 사용)
        # Map/Info 저장
        viewer.settings.setValue("map/static_provider", str(getattr(viewer, "_map_static_provider", "auto")))
        viewer.settings.setValue("map/preview_size", str(getattr(viewer, "_info_map_size_mode", "medium")))
        viewer.settings.setValue("map/default_zoom", int(getattr(viewer, "_info_map_default_zoom", 12)))
        viewer.settings.setValue("map/cache_max_mb", int(getattr(viewer, "_map_cache_max_mb", 128)))
        viewer.settings.setValue("map/cache_max_days", int(getattr(viewer, "_map_cache_max_days", 30)))
        # API 키(QSettings; 평문 저장)
        viewer.settings.setValue("map/kakao_api_key", str(getattr(viewer, "_map_kakao_api_key", "")))
        viewer.settings.setValue("map/google_api_key", str(getattr(viewer, "_map_google_api_key", "")))
        # Info 요약 저장
        viewer.settings.setValue("info/show_dt", bool(getattr(viewer, "_info_show_dt", True)))
        viewer.settings.setValue("info/show_file", bool(getattr(viewer, "_info_show_file", True)))
        viewer.settings.setValue("info/show_dir", bool(getattr(viewer, "_info_show_dir", True)))
        viewer.settings.setValue("info/show_cam", bool(getattr(viewer, "_info_show_cam", True)))
        viewer.settings.setValue("info/show_size", bool(getattr(viewer, "_info_show_size", True)))
        viewer.settings.setValue("info/show_res", bool(getattr(viewer, "_info_show_res", True)))
        viewer.settings.setValue("info/show_mp", bool(getattr(viewer, "_info_show_mp", True)))
        viewer.settings.setValue("info/show_iso", bool(getattr(viewer, "_info_show_iso", True)))
        viewer.settings.setValue("info/show_focal", bool(getattr(viewer, "_info_show_focal", True)))
        viewer.settings.setValue("info/show_aperture", bool(getattr(viewer, "_info_show_aperture", True)))
        viewer.settings.setValue("info/show_shutter", bool(getattr(viewer, "_info_show_shutter", True)))
        viewer.settings.setValue("info/show_gps", bool(getattr(viewer, "_info_show_gps", True)))
        viewer.settings.setValue("info/max_lines", int(getattr(viewer, "_info_max_lines", 50)))
        viewer.settings.setValue("info/shutter_unit", str(getattr(viewer, "_info_shutter_unit", "auto")))
        # 환경변수 동기화는 하지 않습니다. 모든 동작은 config.yaml/메모리 설정 기준으로 수행합니다.
        # 확장 저장(QSettings)
        viewer.settings.setValue("ai/conf_threshold_pct", int(getattr(viewer, "_ai_conf_threshold_pct", 80)))
        viewer.settings.setValue("ai/apply_policy", str(getattr(viewer, "_ai_apply_policy", "보류")))
        viewer.settings.setValue("ai/batch_workers", int(getattr(viewer, "_ai_batch_workers", 4)))
        viewer.settings.setValue("ai/batch_delay_ms", int(getattr(viewer, "_ai_batch_delay_ms", 0)))
        viewer.settings.setValue("ai/batch_retry_count", int(getattr(viewer, "_ai_batch_retry_count", 0)))
        viewer.settings.setValue("ai/batch_retry_delay_ms", int(getattr(viewer, "_ai_batch_retry_delay_ms", 0)))
        viewer.settings.setValue("ai/search_verify_mode_default", str(getattr(viewer, "_search_verify_mode_default", "strict")))
        viewer.settings.setValue("ai/search_verify_topn_default", int(getattr(viewer, "_search_verify_topn_default", 20)))
        viewer.settings.setValue("ai/search_tag_weight", int(getattr(viewer, "_search_tag_weight", 2)))
        viewer.settings.setValue("ai/bg_index_max", int(getattr(viewer, "_bg_index_max", 200)))
        viewer.settings.setValue("ai/privacy_hide_location", bool(getattr(viewer, "_privacy_hide_location", False)))
        # 자연어 검색 정렬/필터/표시/백그라운드 저장
        viewer.settings.setValue("ai/search_sort_order", str(getattr(viewer, "_search_sort_order", "similarity")))
        viewer.settings.setValue("ai/search_filter/min_rating", int(getattr(viewer, "_search_filter_min_rating", 0)))
        viewer.settings.setValue("ai/search_filter/flag_mode", str(getattr(viewer, "_search_filter_flag_mode", "any")))
        viewer.settings.setValue("ai/search_filter/keywords", str(getattr(viewer, "_search_filter_keywords", "")))
        viewer.settings.setValue("ai/search_filter/date_from", str(getattr(viewer, "_search_filter_date_from", "")))
        viewer.settings.setValue("ai/search_filter/date_to", str(getattr(viewer, "_search_filter_date_to", "")))
        viewer.settings.setValue("ai/search_result/thumb_size", int(getattr(viewer, "_search_result_thumb_size", 192)))
        viewer.settings.setValue("ai/search_result/view_mode", str(getattr(viewer, "_search_result_view_mode", "grid")))
        viewer.settings.setValue("ai/search_result/show_score", bool(getattr(viewer, "_search_show_score", True)))
        viewer.settings.setValue("ai/search_result/show_in_filmstrip", bool(getattr(viewer, "_search_show_in_filmstrip", False)))
        viewer.settings.setValue("ai/search/bg_prep_enabled", bool(getattr(viewer, "_search_bg_prep_enabled", False)))
        # 임베딩/검증 고급 옵션 저장
        viewer.settings.setValue("ai/search/use_embedding", bool(getattr(viewer, "_search_use_embedding", True)))
        viewer.settings.setValue("ai/search/verify_strict_only", bool(getattr(viewer, "_search_verify_strict_only", True)))
        viewer.settings.setValue("ai/search/verify_max_candidates", int(getattr(viewer, "_search_verify_max_candidates", 200)))
        viewer.settings.setValue("ai/search/verify_workers", int(getattr(viewer, "_search_verify_workers", 16)))
        viewer.settings.setValue("ai/search/blend_alpha", float(getattr(viewer, "_search_blend_alpha", 0.7)))
        viewer.settings.setValue("ai/embed/batch_size", int(getattr(viewer, "_embed_batch_size", 64)))
        viewer.settings.setValue("ai/embed/model", str(getattr(viewer, "_embed_model", "text-embedding-3-small")))
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


