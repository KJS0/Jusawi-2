from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import threading

import torch  # type: ignore
from PIL import Image  # type: ignore
import numpy as np  # type: ignore
import os
import time
import sqlite3
import json
from ..utils.logging_setup import get_logger  # type: ignore
from ultralytics import YOLO  # type: ignore


@dataclass
class Det:
    bbox: Tuple[int, int, int, int]
    label: str
    score: float


class Yolo11Detector:
    _init_lock = threading.Lock()
    _instance: Optional["Yolo11Detector"] = None

    def __init__(self, model_name: str = "yolo11s"):
        self.model_name = model_name
        self.log = get_logger("svc.object_detection")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights_path, use_pretrained = self._resolve_weights(model_name)
        # Ultralytics YOLO 사용: 로컬 가중치 우선, 없으면 <model_name>.pt 자동 다운로드
        try:
            if weights_path and os.path.isfile(weights_path):
                self.model = YOLO(weights_path)
            else:
                default_pt = f"{model_name}.pt" if not model_name.endswith(".pt") else model_name
                self.model = YOLO(default_pt)
        except Exception:
            # 폴백: 가장 작은 모델
            self.model = YOLO("yolo11n.pt")
        # 클래스 이름
        try:
            self.id2label = {int(i): str(n) for i, n in enumerate(getattr(self.model, 'names', []) or [])}
        except Exception:
            self.id2label = {}
        try:
            src = weights_path if (weights_path and os.path.isfile(weights_path)) else f"hub:{model_name}"
            self.log.info("detector_init | device=%s | src=%s | labels=%d", self.device, src, len(self.id2label))
        except Exception:
            pass
        # 캐시 DB 준비
        try:
            base_dir = os.path.join(os.path.expanduser("~"), ".jusawi")
            os.makedirs(base_dir, exist_ok=True)
            self._cache_path = os.path.join(base_dir, "detect_cache.sqlite3")
            self._ensure_cache_db()
        except Exception:
            self._cache_path = ""

    # ---- 캐시 유틸 ----
    def _ensure_cache_db(self) -> None:
        if not self._cache_path:
            return
        con = sqlite3.connect(self._cache_path)
        try:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS det_cache (
                    path TEXT,
                    mtime INTEGER,
                    model TEXT,
                    params TEXT,
                    dets TEXT,
                    PRIMARY KEY(path, mtime, model, params)
                )
                """
            )
            con.commit()
        finally:
            con.close()

    def _sig(self, params: Dict[str, Any]) -> str:
        # 모델/추론 파라미터/최종 필터 임계값을 모두 포함한 시그니처
        parts = [f"model={self.model_name}"]
        for k in sorted(params.keys()):
            v = params[k]
            parts.append(f"{k}={int(v) if isinstance(v, bool) else v}")
        return "|".join(parts)

    def _load_cache(self, path: str, mtime: int, model_sig: str, params_sig: str) -> Optional[List[Det]]:
        if not self._cache_path:
            return None
        con = sqlite3.connect(self._cache_path)
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT dets FROM det_cache WHERE path=? AND mtime=? AND model=? AND params=?",
                (path, int(mtime), model_sig, params_sig),
            )
            row = cur.fetchone()
            if not row:
                return None
            try:
                data = json.loads(str(row[0]))
                out: List[Det] = []
                for d in data.get("dets", []):
                    bbox = tuple(int(x) for x in d.get("bbox", [0, 0, 0, 0]))
                    label = str(d.get("label", ""))
                    score = float(d.get("score", 0.0))
                    out.append(Det(bbox=bbox, label=label, score=score))
                return out
            except Exception:
                return None
        except Exception:
            return None
        finally:
            con.close()

    def _save_cache(self, path: str, mtime: int, model_sig: str, params_sig: str, dets: List[Det]) -> None:
        if not self._cache_path or not dets:
            return
        payload = {
            "dets": [
                {"bbox": list(d.bbox), "label": d.label, "score": float(d.score)}
                for d in dets
            ]
        }
        con = sqlite3.connect(self._cache_path)
        try:
            cur = con.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO det_cache(path, mtime, model, params, dets) VALUES(?,?,?,?,?)",
                (path, int(mtime), model_sig, params_sig, json.dumps(payload, ensure_ascii=False)),
            )
            con.commit()
        except Exception:
            pass
        finally:
            con.close()

    def _resolve_weights(self, model_name: str) -> Tuple[Optional[str], bool]:
        """
        로컬 가중치 우선 경로를 찾는다.
        - 환경변수 JUSAWI_YOLO_WEIGHTS 우선
        - ./models/<model_name>.pt
        반환: (weights_path or None, use_pretrained_bool)
        """
        # env
        p_env = os.getenv("JUSAWI_YOLO_WEIGHTS")
        if p_env and os.path.isfile(p_env):
            return p_env, False
        # ./models/<name>.pt
        try:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
            base_dir = os.path.normpath(base_dir)
        except Exception:
            base_dir = "models"
        local_pt = os.path.join(base_dir, f"{model_name}.pt")
        if os.path.isfile(local_pt):
            return local_pt, False
        # no local weights -> use pretrained from hub
        return None, True

    @classmethod
    def get(cls) -> "Yolo11Detector":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = Yolo11Detector()
        return cls._instance

    @torch.inference_mode()
    def detect_pil(
        self,
        image: Image.Image,
        score_thr: float = 0.5,
        max_results: Optional[int] = None,
    ) -> List[Det]:
        t0 = time.perf_counter()
        try:
            w, h = image.size
            self.log.info("detect_begin | device=%s | thr=%.3f | size=%dx%d", self.device, float(score_thr), w, h)
        except Exception:
            pass
        def _run_infer(device: str) -> List[Det]:
            # PIL -> numpy RGB
            np_img = np.array(image.convert("RGB"))
            # inference (imgsz default 640)
            results = self.model.predict(
                source=np_img,
                imgsz=1536,          # 소형/원거리 객체에 유리한 고해상도
                device=device,
                conf=0.001,          # 내부 선필터 최소화(최종 score_thr로 필터)
                iou=0.45,            # 혼잡 장면에서 과한 억제 방지
                augment=True,        # TTA
                agnostic_nms=True,   # 클래스 비종속 NMS
                verbose=False
            )
            dets_: List[Det] = []
            if not results:
                return dets_
            r = results[0]
            boxes = getattr(r, "boxes", None)
            names = getattr(self.model, "names", {})
            if boxes is None:
                return dets_
            try:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                conf = boxes.conf.detach().cpu().numpy()
                cls_ = boxes.cls.detach().cpu().numpy()
            except Exception:
                return dets_
            for (x1, y1, x2, y2), sc, ci in zip(xyxy, conf, cls_):
                try:
                    if float(sc) < float(score_thr):
                        continue
                    cls_i = int(ci)
                    if isinstance(names, dict) and cls_i in names:
                        label = str(names[cls_i])
                    elif isinstance(names, list) and 0 <= cls_i < len(names):
                        label = str(names[cls_i])
                    else:
                        label = str(cls_i)
                    dets_.append(
                        Det(
                            bbox=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                            label=label,
                            score=float(sc),
                        )
                    )
                except Exception:
                    continue
            return dets_

        # 1차: 현재 디바이스, 2차: 실패 시 CPU 폴백
        try:
            dets = _run_infer(self.device)
        except Exception:
            try:
                self.log.warning("detect_error_device | device=%s | fallback=cpu", self.device)
                dets = _run_infer("cpu")
            except Exception:
                self.log.exception("detect_fail")
                raise
        dt_ms = int((time.perf_counter() - t0) * 1000)
        try:
            self.log.info("detect_end | count=%d | ms=%d", len(dets), dt_ms)
            if dets:
                top = sorted(dets, key=lambda d: d.score, reverse=True)[:5]
                for i, d in enumerate(top):
                    self.log.debug("detect_top | rank=%d | score=%.3f | label=%s | bbox=%s", i + 1, d.score, d.label, d.bbox)
        except Exception:
            pass
        if isinstance(max_results, int) and max_results > 0 and len(dets) > max_results:
            dets.sort(key=lambda d: d.score, reverse=True)
            dets = dets[:max_results]
        return dets

    def detect_file(
        self,
        image_path: str,
        score_thr: float = 0.5,
        max_results: Optional[int] = None,
    ) -> List[Det]:
        # 캐시 키 구성
        try:
            mtime = int(os.path.getmtime(image_path))
        except Exception:
            mtime = 0
        # 모델 시그니처와 파라미터 시그니처(추론/최종 필터)를 분리
        model_sig = self.model_name
        params_sig = self._sig({
            "imgsz": 1536,
            "conf": 0.001,
            "iou": 0.45,
            "augment": True,
            "agnostic_nms": True,
            "thr": float(score_thr),
        })
        # 캐시 조회
        cached = self._load_cache(image_path, mtime, model_sig, params_sig)
        if cached is not None:
            return cached[: max_results] if isinstance(max_results, int) and max_results and len(cached) > max_results else cached
        # 추론
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            dets = self.detect_pil(im, score_thr=score_thr, max_results=max_results)
        # 저장
        try:
            self._save_cache(image_path, mtime, model_sig, params_sig, dets)
        except Exception:
            pass
        return dets


