from __future__ import annotations

import json
import os
import urllib.parse
from typing import List, Tuple

from PyQt6.QtCore import QUrl  # type: ignore[import]
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel  # type: ignore[import]


def _to_file_uri(path: str) -> str:
	try:
		# 표준 file:// URI로 변환 (Windows 경로 지원)
		# urllib.parse.quote를 이용해 공백/특수문자 이스케이프
		abspath = os.path.abspath(path)
		# 앞의 드라이브 문자와 콜론 처리
		if os.name == "nt":
			# /C:/ 형태로 맞춤
			return "file:///" + urllib.parse.quote(abspath.replace("\\", "/"), safe="/:")
		return "file://" + urllib.parse.quote(abspath, safe="/:")
	except Exception:
		return ""


def _make_thumb_data_uri(path: str, max_w: int = 320, max_h: int = 240) -> str:
	# 브라우저/웹엔진의 file:// 접근 제한을 우회하기 위해 Data URI로 미리보기 생성
	try:
		from PIL import Image  # type: ignore
		import io, base64
	except Exception:
		return ""
	try:
		if not os.path.isfile(path):
			return ""
		with Image.open(path) as im:
			try:
				im = im.convert("RGB")
			except Exception:
				pass
			try:
				im.thumbnail((int(max_w), int(max_h)))
			except Exception:
				pass
			buf = io.BytesIO()
			try:
				im.save(buf, format="JPEG", quality=80, optimize=True)
			except Exception:
				im.save(buf, format="JPEG")
			import base64 as _b64  # shadow-safe
			data = _b64.b64encode(buf.getvalue()).decode("ascii")
			return "data:image/jpeg;base64," + data
	except Exception:
		return ""


def _collect_points(paths: List[str]) -> Tuple[list, int]:
	points: list[dict] = []
	cur_index = -1
	try:
		from ..services.exif_utils import extract_with_pillow  # type: ignore
	except Exception:
		extract_with_pillow = None  # type: ignore
	if not extract_with_pillow:
		return points, cur_index
	for i, p in enumerate(paths):
		try:
			meta = extract_with_pillow(p) or {}
			lat = meta.get("lat")
			lon = meta.get("lon")
			if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
				points.append({
					"lat": float(lat),
					"lon": float(lon),
					"name": os.path.basename(p),
					"path": p,
					"fileUri": _to_file_uri(p),
					"thumbDataUri": _make_thumb_data_uri(p, max_w=320, max_h=240),
				})
		except Exception:
			continue
	return points, cur_index


class FolderMapDialog(QDialog):
	def __init__(self, owner, image_paths: List[str], current_path: str | None = None):
		super().__init__(owner)
		self.setWindowTitle("폴더 지도 보기")
		try:
			self.resize(960, 640)
		except Exception:
			pass

		# WebEngine 가용성 확인
		try:
			# 일부 환경(관리자 권한/가상화/보안정책)에서 샌드박스/GPU로 인한 로딩 실패를 완화
			os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
			os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
			from PyQt6.QtWebEngineWidgets import QWebEngineView  # type: ignore
			from PyQt6.QtWebEngineCore import QWebEnginePage  # type: ignore
			from PyQt6.QtCore import QUrlQuery  # type: ignore
		except Exception:
			# 폴백: 시스템 기본 브라우저로 HTML 지도를 연다.
			_err = None
			try:
				import traceback
				_err = traceback.format_exc()
			except Exception:
				_err = None
			points, _ = _collect_points(image_paths or [])
			if not points:
				lay = QVBoxLayout(self)
				msg = "웹 엔진(QtWebEngine)을 초기화할 수 없어 지도를 표시할 수 없습니다.\n또한 이 폴더에는 GPS 위치가 포함된 사진이 없습니다."
				try:
					if _err:
						msg += f"\n\n원인(참고):\n{_err}"
				except Exception:
					pass
				lay.addWidget(QLabel(msg, self))
				return
			js_points = json.dumps(points, ensure_ascii=False)
			html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    html, body, #map {{
      height: 100%;
      margin: 0;
      padding: 0;
    }}
    .popup-box {{
      text-align: center;
      max-width: 360px;
    }}
    .popup-box img {{
      display: block;
      max-width: 340px;
      width: 100%;
      height: auto;
      max-height: 260px;
      border-radius: 4px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    }}
    .popup-title {{
      font-weight: 600;
      margin-bottom: 6px;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script>
    const data = {js_points};
    const map = L.map('map');
    const group = L.featureGroup();
    const POPUP_MAX_W = 380; // Leaflet 기본 300 제한 완화
    const tile = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }});
    tile.addTo(map);
    function makePopupHtml(p) {{
      const src = (p.thumbDataUri && p.thumbDataUri.startsWith('data:')) ? p.thumbDataUri : (p.fileUri || '');
      const img = `<a href="${{p.fileUri}}"><img src="${{src}}" alt="${{p.name}}"></a>`;
      const title = `<div class="popup-title">${{p.name || ''}}</div>`;
      return `<div class="popup-box">${{title}}${{img}}</div>`;
    }}
    (data || []).forEach(p => {{
      if (typeof p.lat === 'number' && typeof p.lon === 'number') {{
        const m = L.marker([p.lat, p.lon]);
        m.bindPopup(makePopupHtml(p), {{ maxWidth: POPUP_MAX_W, closeButton: true, autoPan: true }});
        group.addLayer(m);
      }}
    }});
    if (group.getLayers().length > 0) {{
      group.addTo(map);
      try {{
        map.fitBounds(group.getBounds().pad(0.2));
      }} catch (e) {{
        map.setView([data[0].lat, data[0].lon], 12);
      }}
    }} else {{
      map.setView([37.5665, 126.9780], 6);
    }}
  </script>
</body>
</html>"""
			try:
				import tempfile
				from PyQt6.QtGui import QDesktopServices  # type: ignore
				tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
				tmp_path = tmp.name
				try:
					tmp.write(html.encode("utf-8"))
				finally:
					tmp.close()
				QDesktopServices.openUrl(QUrl.fromLocalFile(tmp_path))
			except Exception:
				pass
			lay = QVBoxLayout(self)
			msg = "웹 엔진(QtWebEngine)을 초기화할 수 없어 기본 브라우저로 지도를 열었습니다."
			try:
				if _err:
					msg += f"\n\n원인(참고):\n{_err}"
			except Exception:
				pass
			lay.addWidget(QLabel(msg, self))
			return

		class _BridgePage(QWebEnginePage):
			def acceptNavigationRequest(self_inner, url: QUrl, nav_type, is_main_frame: bool) -> bool:  # type: ignore[override]
				try:
					if url.scheme().lower() == "open":
						try:
							q = QUrlQuery(url)
							enc = q.queryItemValue("p")
						except Exception:
							enc = ""
						if enc:
							try:
								target_path = urllib.parse.unquote(enc)
							except Exception:
								target_path = enc
							try:
								if os.path.isfile(target_path):
									# 뷰어에서 해당 파일 열기
									getattr(owner, "load_image", lambda p, source=None: None)(target_path, source="map")
									# 대화상자는 유지(필요 시 닫고 싶다면 아래 주석 해제)
									# self.close()
							except Exception:
								pass
						return False
				except Exception:
					return False
				return super().acceptNavigationRequest(url, nav_type, is_main_frame)

		view = QWebEngineView(self)
		page = _BridgePage(view)
		view.setPage(page)

		points, _ = _collect_points(image_paths or [])

		if not points:
			lay = QVBoxLayout(self)
			lay.addWidget(QLabel("이 폴더에는 GPS 위치가 포함된 사진이 없습니다.", self))
			return

		# JS에 안전하게 넘기기 위해 JSON 직렬화
		js_points = json.dumps(points, ensure_ascii=False)

		# Leaflet 기반 HTML 생성
		# CDN 사용 (오프라인 환경에서는 로드 실패할 수 있음)
		html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    html, body, #map {{
      height: 100%;
      margin: 0;
      padding: 0;
    }}
    .popup-box {{
      text-align: center;
      max-width: 360px;
    }}
    .popup-box img {{
      display: block;
      max-width: 340px;
      width: 100%;
      height: auto;
      max-height: 260px;
      border-radius: 4px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    }}
    .popup-title {{
      font-weight: 600;
      margin-bottom: 6px;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script>
    const data = {js_points};
    const map = L.map('map');
    const group = L.featureGroup();
    const POPUP_MAX_W = 380; // Leaflet 기본 300 제한 완화

    const tile = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }});
    tile.addTo(map);

    function makePopupHtml(p) {{
      const enc = encodeURIComponent(p.path || '');
      const src = (p.thumbDataUri && p.thumbDataUri.startsWith('data:')) ? p.thumbDataUri : (p.fileUri || '');
      const img = `<a href="open://?p=${{enc}}"><img src="${{src}}" alt="${{p.name}}"></a>`;
      const title = `<div class="popup-title">${{p.name || ''}}</div>`;
      return `<div class="popup-box">${{title}}${{img}}</div>`;
    }}

    (data || []).forEach(p => {{
      if (typeof p.lat === 'number' && typeof p.lon === 'number') {{
        const m = L.marker([p.lat, p.lon]);
        m.bindPopup(makePopupHtml(p), {{ maxWidth: POPUP_MAX_W, closeButton: true, autoPan: true }});
        group.addLayer(m);
      }}
    }});

    if (group.getLayers().length > 0) {{
      group.addTo(map);
      try {{
        map.fitBounds(group.getBounds().pad(0.2));
      }} catch (e) {{
        map.setView([data[0].lat, data[0].lon], 12);
      }}
    }} else {{
      map.setView([37.5665, 126.9780], 6);
    }}
  </script>
</body>
</html>"""

		view.setHtml(html, baseUrl=QUrl("about:blank"))

		lay = QVBoxLayout(self)
		lay.setContentsMargins(0, 0, 0, 0)
		lay.addWidget(view)


