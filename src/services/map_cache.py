from __future__ import annotations

import os, hashlib, threading, time
from typing import Optional, Tuple
from ..storage.settings_store import _load_yaml_configs  # type: ignore
from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool
from PyQt6.QtGui import QPixmap

_cache_mem: dict[str, QPixmap] = {}
_lock = threading.Lock()

# 디스크 캐시 정책(환경변수로 제어)
def _get_limits() -> Tuple[int, int]:
    try:
        cfg = _load_yaml_configs().get('map', {})  # type: ignore
    except Exception:
        cfg = {}
    try:
        max_mb = int(cfg.get('cache_max_mb', 128))
    except Exception:
        max_mb = 128
    try:
        max_days = int(cfg.get('cache_max_days', 30))
    except Exception:
        max_days = 30
    return max(8, max_mb), max(1, max_days)

def _cache_dir() -> str:
    base = os.path.join(os.path.expanduser("~"), ".jusawi", "maps")
    os.makedirs(base, exist_ok=True)
    return base

def _key(lat: float, lon: float, w: int, h: int, zoom: int) -> str:
    s = f"{lat:.6f},{lon:.6f}|{w}x{h}|z{zoom}"
    return hashlib.sha1(s.encode()).hexdigest()

def get_cached(lat: float, lon: float, w: int, h: int, zoom: int) -> Optional[QPixmap]:
    k = _key(lat, lon, w, h, zoom)
    with _lock:
        pm = _cache_mem.get(k)
        if pm and not pm.isNull():
            return pm
    p = os.path.join(_cache_dir(), f"{k}.png")
    if os.path.exists(p):
        pm = QPixmap(p)
        if not pm.isNull():
            with _lock:
                _cache_mem[k] = pm
            return pm
    return None

def put_cached(lat: float, lon: float, w: int, h: int, zoom: int, pm: QPixmap) -> None:
    if pm.isNull():
        return
    k = _key(lat, lon, w, h, zoom)
    with _lock:
        _cache_mem[k] = pm
    p = os.path.join(_cache_dir(), f"{k}.png")
    try:
        pm.save(p, "PNG")
    except Exception:
        pass
    try:
        _enforce_limits()
    except Exception:
        pass

def _enforce_limits() -> None:
    base = _cache_dir()
    try:
        max_mb, max_days = _get_limits()
        now = time.time()
        files = []
        total = 0
        for name in os.listdir(base):
            if not name.endswith('.png'):
                continue
            fp = os.path.join(base, name)
            try:
                st = os.stat(fp)
                age_days = (now - float(st.st_mtime)) / 86400.0
                if age_days > max_days:
                    os.remove(fp)
                    continue
                size = int(st.st_size)
                total += size
                files.append((fp, st.st_mtime, size))
            except Exception:
                continue
        limit_bytes = max_mb * 1024 * 1024
        if total > limit_bytes:
            # 오래된 파일부터 제거
            files.sort(key=lambda x: x[1])
            i = 0
            while total > limit_bytes and i < len(files):
                fp, _mt, sz = files[i]
                try:
                    os.remove(fp)
                    total -= sz
                except Exception:
                    pass
                i += 1
    except Exception:
        pass

def clear_disk_cache() -> None:
    try:
        base = _cache_dir()
        if os.path.isdir(base):
            for name in os.listdir(base):
                fp = os.path.join(base, name)
                try:
                    if os.path.isfile(fp):
                        os.remove(fp)
                except Exception:
                    pass
    except Exception:
        pass

class MapFetchTask(QRunnable):
    def __init__(self, lat: float, lon: float, w: int, h: int, zoom: int, token: int, receiver: QObject, signal_name: str):
        super().__init__()
        self.lat, self.lon, self.w, self.h, self.zoom, self.token = lat, lon, w, h, zoom, token
        self.receiver, self.signal_name = receiver, signal_name

    def run(self):
        pm = get_cached(self.lat, self.lon, self.w, self.h, self.zoom)
        if pm is None:
            try:
                from .geocoding import get_static_map_png  # type: ignore
                # 제공자 자동 선택(google→kakao→osm 폴백). provider 인자는 생략.
                data = get_static_map_png(self.lat, self.lon, width=self.w, height=self.h, zoom=self.zoom)
                if data:
                    pm2 = QPixmap()
                    if pm2.loadFromData(bytes(data)):
                        put_cached(self.lat, self.lon, self.w, self.h, self.zoom, pm2)
                        pm = pm2
            except Exception:
                pm = None
        try:
            sig = getattr(self.receiver, self.signal_name)
            sig.emit(self.token, pm if pm else QPixmap())
        except Exception:
            pass

def submit_fetch(lat: float, lon: float, w: int, h: int, zoom: int, token: int, receiver: QObject, signal_name: str):
    QThreadPool.globalInstance().start(MapFetchTask(lat, lon, w, h, zoom, token, receiver, signal_name))


