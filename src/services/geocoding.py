from __future__ import annotations

"""
GPS 좌표를 주소로 변환하는 Geocoding 모듈
한국: Kakao Map API 사용 (도로명주소 우선, 지번주소 대체)
해외: Google Maps Geocoding API 사용
"""

import logging
from typing import Optional, Dict

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

logger = logging.getLogger(__name__)

from ..storage.settings_store import _load_yaml_configs  # type: ignore

PROVINCE_MAPPING = {
    '서울특별시': '서울특별시',  # 2024 개정명칭 대응(없으면 원본 유지)
    '서울시': '서울특별시',
    '서울': '서울특별시',
    '부산광역시': '부산광역시',
    '부산시': '부산광역시',
    '부산': '부산광역시',
    '대구광역시': '대구광역시',
    '대구시': '대구광역시',
    '대구': '대구광역시',
    '인천광역시': '인천광역시',
    '인천시': '인천광역시',
    '인천': '인천광역시',
    '광주광역시': '광주광역시',
    '광주시': '광주광역시',
    '광주': '광주광역시',
    '대전광역시': '대전광역시',
    '대전시': '대전광역시',
    '대전': '대전광역시',
    '울산광역시': '울산광역시',
    '울산시': '울산광역시',
    '울산': '울산광역시',
    '세종특별자치시': '세종특별자치시',
    '세종시': '세종특별자치시',
    '세종': '세종특별자치시',
    '경기': '경기도',
    '경기도': '경기도',
    '강원': '강원특별자치도',
    '강원도': '강원특별자치도',
    '강원특별자치도': '강원특별자치도',
    '충북': '충청북도',
    '충청북도': '충청북도',
    '충남': '충청남도',
    '충청남도': '충청남도',
    '전북': '전북특별자치도',
    '전라북도': '전북특별자치도',
    '전북특별자치도': '전북특별자치도',
    '전남': '전라남도',
    '전라남도': '전라남도',
    '경북': '경상북도',
    '경상북도': '경상북도',
    '경남': '경상남도',
    '경상남도': '경상남도',
    '제주': '제주특별자치도',
    '제주도': '제주특별자치도',
    '제주특별자치도': '제주특별자치도'
}


def standardize_province_name(address: str) -> str:
    if not address:
        return address
    # 긴 키부터 치환
    for short_name, full_name in sorted(PROVINCE_MAPPING.items(), key=lambda x: len(x[0]), reverse=True):
        if address.startswith(short_name + ' '):
            return address.replace(short_name + ' ', full_name + ' ', 1)
        if address.startswith(short_name):
            return address.replace(short_name, full_name, 1)
    return address


class GeocodingService:
    def __init__(self):
        cfg = {}
        try:
            cfg = _load_yaml_configs().get('map', {})  # type: ignore
        except Exception:
            cfg = {}
        ak = cfg.get('api_keys', {}) if isinstance(cfg.get('api_keys'), dict) else {}
        self.kakao_api_key = str(ak.get('kakao', '') or '')
        self.google_api_key = str(ak.get('google', '') or '')
        if not self.kakao_api_key:
            logger.info("Kakao API 키 없음 — 한국 주소 변환 비활성화")
        if not self.google_api_key:
            logger.info("Google Maps API 키 없음 — 해외 주소 변환 비활성화")

    def _refresh_api_keys(self) -> None:
        try:
            cfg = _load_yaml_configs().get('map', {})  # type: ignore
        except Exception:
            cfg = {}
        ak = cfg.get('api_keys', {}) if isinstance(cfg.get('api_keys'), dict) else {}
        try:
            self.kakao_api_key = str(ak.get('kakao', self.kakao_api_key) or self.kakao_api_key)
        except Exception:
            pass
        try:
            self.google_api_key = str(ak.get('google', self.google_api_key) or self.google_api_key)
        except Exception:
            pass

    def is_korea_coordinate(self, latitude: float, longitude: float) -> bool:
        return (33.0 <= float(latitude) <= 38.6) and (124.0 <= float(longitude) <= 132.0)

    def get_address_from_coordinates(self, latitude: float, longitude: float, language: str | None = None) -> Optional[Dict]:
        try:
            # 최신 키 반영
            self._refresh_api_keys()
            if requests is None:
                return None
            if self.is_korea_coordinate(latitude, longitude):
                return self._get_korea_address(latitude, longitude)
            # 국제 주소: 언어 코드 전달(기본 ko)
            lang = (language or "ko").strip() or "ko"
            return self._get_international_address(latitude, longitude, language=lang)
        except Exception:
            return None

    def _get_korea_address(self, latitude: float, longitude: float) -> Optional[Dict]:
        if not self.kakao_api_key or requests is None:
            try:
                logger.info("geocode.kr.skip | reason=%s", "no_key" if not self.kakao_api_key else "no_requests")
            except Exception:
                pass
            return None
        try:
            url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
            headers = {"Authorization": f"KakaoAK {self.kakao_api_key}"}
            params = {"x": longitude, "y": latitude, "input_coord": "WGS84"}
            resp = requests.get(url, headers=headers, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            docs = data.get('documents', [])
            if not docs:
                try:
                    logger.warning("geocode.kr.empty | status=%s", data.get('status', ''))
                except Exception:
                    pass
                return None
            doc = docs[0]
            road = (doc.get('road_address') or {})
            addr = (doc.get('address') or {})
            if road:
                full = road.get('address_name', '')
                a_type = '도로명'
            elif addr:
                full = addr.get('address_name', '')
                a_type = '지번'
            else:
                return None
            full = standardize_province_name(full)
            return {
                'country': '대한민국',
                'full_address': full,
                'address_type': a_type,
                'coordinates': f"{latitude}, {longitude}",
                'formatted': f"{full} ({a_type})",
            }
        except Exception as e:
            try:
                logger.exception("geocode.kr.error | err=%s", str(e))
            except Exception:
                pass
            return None

    def _get_international_address(self, latitude: float, longitude: float, language: str = "ko") -> Optional[Dict]:
        if not self.google_api_key or requests is None:
            try:
                logger.info("geocode.intl.skip | reason=%s", "no_key" if not self.google_api_key else "no_requests")
            except Exception:
                pass
            return None
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {"latlng": f"{latitude},{longitude}", "key": self.google_api_key, "language": str(language or "ko")}
            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            if data.get('status') != 'OK':
                try:
                    logger.warning("geocode.intl.status | status=%s | error_message=%s", data.get('status'), data.get('error_message'))
                except Exception:
                    pass
                return None
            results = data.get('results', [])
            if not results:
                return None
            best = results[0]
            full = best.get('formatted_address', '')
            return {
                'country': '',
                'full_address': full,
                'address_type': '해외주소',
                'coordinates': f"{latitude}, {longitude}",
                'formatted': full,
            }
        except Exception as e:
            try:
                logger.exception("geocode.intl.error | err=%s", str(e))
            except Exception:
                pass
            return None


"""전역 인스턴스"""
geocoding_service = GeocodingService()


def _get_google_static_map_png(latitude: float, longitude: float, width: int = 640, height: int = 400, zoom: int = 15) -> Optional[bytes]:
    try:
        if requests is None:
            return None
        # config.yaml에서 키 로드
        try:
            ak = _load_yaml_configs().get('map', {}).get('api_keys', {})  # type: ignore
        except Exception:
            ak = {}
        api_key = str(ak.get('google', '') or '')
        if not api_key:
            return None
        w = max(64, min(640, int(width)))
        h = max(64, min(640, int(height)))
        z = max(1, min(20, int(zoom)))
        url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{float(latitude)},{float(longitude)}",
            "zoom": str(z),
            "size": f"{w}x{h}",
            "maptype": "roadmap",
            "markers": f"color:red|{float(latitude)},{float(longitude)}",
            "key": api_key,
            "scale": "3",
        }
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code != 200:
            try:
                logger.warning("map.static.google.http | code=%s | reason=%s", resp.status_code, getattr(resp, 'reason', ''))
            except Exception:
                pass
            return None
        if "image" not in (resp.headers.get("Content-Type", "") or ""):
            try:
                logger.warning("map.static.google.content_type | ct=%s", resp.headers.get("Content-Type", ""))
            except Exception:
                pass
            return None
        return resp.content
    except Exception:
        return None


def _get_osm_static_map_png(latitude: float, longitude: float, width: int = 640, height: int = 400, zoom: int = 15) -> Optional[bytes]:
    """OSM 공개 정적 지도 서비스 사용(키 불필요). 정책에 따라 트래픽을 과도하게 발생시키지 않도록 주의."""
    try:
        if requests is None:
            return None
        w = max(64, min(1024, int(width)))
        h = max(64, min(1024, int(height)))
        z = max(1, min(20, int(zoom)))
        url = "https://staticmap.openstreetmap.de/staticmap.php"
        params = {
            "center": f"{float(latitude)},{float(longitude)}",
            "zoom": str(z),
            "size": f"{w}x{h}",
            "markers": f"{float(latitude)},{float(longitude)},red-pushpin",
        }
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code != 200:
            try:
                logger.warning("map.static.osm.http | code=%s | reason=%s", resp.status_code, getattr(resp, 'reason', ''))
            except Exception:
                pass
            return None
        if "image" not in (resp.headers.get("Content-Type", "") or ""):
            try:
                logger.warning("map.static.osm.content_type | ct=%s", resp.headers.get("Content-Type", ""))
            except Exception:
                pass
            return None
        return resp.content
    except Exception:
        return None


def _get_kakao_static_map_png(latitude: float, longitude: float, width: int = 640, height: int = 400, zoom: int = 15) -> Optional[bytes]:
    """Kakao Static Map REST API 사용. Kakao REST API 키 필요."""
    try:
        if requests is None:
            return None
        try:
            ak = _load_yaml_configs().get('map', {}).get('api_keys', {})  # type: ignore
        except Exception:
            ak = {}
        api_key = str(ak.get('kakao', '') or '')
        if not api_key:
            return None
        w = max(64, min(1024, int(width)))
        h = max(64, min(1024, int(height)))
        # Kakao level: 1(가까움) ~ 14(멀어짐). Google/OSM(1~20)와 다르므로 대략 매핑
        try:
            level = int(max(1, min(14, round(21 - int(zoom)))))
        except Exception:
            level = 6
        url = "https://dapi.kakao.com/v2/maps/staticmap"
        params = {
            "center": f"{float(longitude)},{float(latitude)}",
            "level": str(level),
            "w": str(w),
            "h": str(h),
            "markers": f"color:red|{float(longitude)},{float(latitude)}",
        }
        headers = {"Authorization": f"KakaoAK {api_key}"}
        resp = requests.get(url, headers=headers, params=params, timeout=6)
        if resp.status_code != 200:
            try:
                logger.warning("map.static.kakao.http | code=%s | reason=%s", resp.status_code, getattr(resp, 'reason', ''))
            except Exception:
                pass
            return None
        if "image" not in (resp.headers.get("Content-Type", "") or ""):
            try:
                logger.warning("map.static.kakao.content_type | ct=%s", resp.headers.get("Content-Type", ""))
            except Exception:
                pass
            return None
        return resp.content
    except Exception:
        return None


def get_static_map_png(latitude: float, longitude: float, width: int = 640, height: int = 400, zoom: int = 15, provider: str | None = None) -> Optional[bytes]:
    """정적 지도 PNG 바이트를 반환한다.

    우선순위:
    - provider가 'google'이면 Google 시도, 실패 시 OSM 폴백
    - provider가 'kakao'이면 Kakao 시도, 실패 시 OSM 폴백
    - provider가 None 또는 'auto'이면:
        - 한국 좌표 & Kakao 키가 있으면 Kakao → 실패 시 Google → 실패 시 OSM
        - 그 외에는 Google → 실패 시 Kakao → 실패 시 OSM
    """
    prov = (provider or "auto").strip().lower()
    # 최신 키 로드
    try:
        ak = _load_yaml_configs().get('map', {}).get('api_keys', {})  # type: ignore
    except Exception:
        ak = {}
    kakao_key = str(ak.get('kakao', '') or '')
    google_key = str(ak.get('google', '') or '')
    try:
        logger.info("map.static | prov=%s | google_key=%s | kakao_key=%s | lat=%.6f | lon=%.6f | w=%d | h=%d | z=%d",
                    prov, "set" if bool(google_key) else "", "set" if bool(kakao_key) else "",
                    float(latitude), float(longitude), int(width), int(height), int(zoom))
    except Exception:
        pass

    def _try_google() -> Optional[bytes]:
        return _get_google_static_map_png(latitude, longitude, width, height, zoom)

    def _try_kakao() -> Optional[bytes]:
        return _get_kakao_static_map_png(latitude, longitude, width, height, zoom)

    def _try_osm() -> Optional[bytes]:
        return _get_osm_static_map_png(latitude, longitude, width, height, zoom)

    # 강제 지정 처리
    if prov == 'google':
        data = _try_google()
        if not data:
            try:
                logger.warning("map.static.google_failed -> fallback=osm")
            except Exception:
                pass
        return data or _try_osm()
    if prov == 'kakao':
        data = _try_kakao()
        if not data:
            try:
                logger.warning("map.static.kakao_failed -> fallback=osm")
            except Exception:
                pass
        return data or _try_osm()

    # 자동 선택
    if GeocodingService().is_korea_coordinate(latitude, longitude) and kakao_key:
        data = _try_kakao() or _try_google()
        if not data:
            try:
                logger.warning("map.static.auto_korea_failed -> fallback=osm")
            except Exception:
                pass
        return data or _try_osm()
    # 비한국 좌표 또는 카카오 키 없음
    if google_key:
        data = _try_google() or _try_kakao()
        if not data:
            try:
                logger.warning("map.static.auto_global_failed -> fallback=osm")
            except Exception:
                pass
        return data or _try_osm()
    if kakao_key:
        data = _try_kakao()
        if not data:
            try:
                logger.warning("map.static.auto_kakao_only_failed -> fallback=osm")
            except Exception:
                pass
        return data or _try_osm()
    return _try_osm()


