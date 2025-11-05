from __future__ import annotations

import os
import json
import hashlib
from typing import Dict, Any, List, Tuple

from ..utils.logging_setup import get_logger
try:
    import threading as _threading
except Exception:
    _threading = None  # type: ignore

_log = get_logger("svc.Verifier")


def _sha1(b: bytes) -> str:
    try:
        return hashlib.sha1(b).hexdigest()
    except Exception:
        return ""


class VerifierService:
    """GPT 비전을 활용해 후보 이미지가 질의 장면과 일치하는지 재검증.

    결과 스키마(JSON): { match: bool, confidence: float, reasons: string }
    캐시 키: sha1(image_bytes_1024_jpeg) + '|' + sha1(prompt)
    """

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 timeout_s: float | None = None, top_p: float | None = None,
                 n: int | None = None, agg: str | None = None):
        self._api_key = api_key or None
        # 기본 모델은 gpt-4o 계열 권장(프로젝트 기본값 유지)
        self._model = (model or "gpt-5-nano")
        # 타임아웃 상향(요청 반영)
        self._timeout_s = float(timeout_s if timeout_s is not None else 45.0)
        self._top_p = float(top_p if top_p is not None else 1.0)
        try:
            self._n = int(n if n is not None else 1)
        except Exception:
            self._n = 1
        self._agg = (agg or "max").lower()
        # 동시 호출 폭주 방지: 가변 동시성 제어용 락/카운터
        try:
            self._lock = _threading.Lock() if _threading is not None else None  # type: ignore
            self._inflight = 0
            self._max_parallel = 8
        except Exception:
            self._lock = None  # type: ignore
            self._inflight = 0
            self._max_parallel = 8
        base_dir = os.path.join(os.path.expanduser("~"), ".jusawi")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            pass
        self._cache_path = os.path.join(base_dir, "verify_cache.jsonl")
        self._cache = {}
        try:
            if os.path.exists(self._cache_path):
                with open(self._cache_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            obj = json.loads(line)
                            self._cache[obj.get("key", "?")] = obj.get("value")
                        except Exception:
                            pass
        except Exception:
            pass

        # OpenAI 클라이언트 재사용(keep-alive)
        try:
            from openai import OpenAI  # type: ignore
            import httpx  # type: ignore
            # 환경 프록시 자동 감지 (httpx 0.28+ API)
            http_client = None
            try:
                hp = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
                sp = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
                ap = os.getenv("ALL_PROXY") or os.getenv("all_proxy")
                if ap:
                    http_client = httpx.Client(proxy=ap, timeout=self._timeout_s)
                elif hp or sp:
                    mounts = {}
                    if hp:
                        mounts["http://"] = httpx.HTTPTransport(proxy=hp)
                    if sp:
                        mounts["https://"] = httpx.HTTPTransport(proxy=sp)
                    if mounts:
                        http_client = httpx.Client(mounts=mounts, timeout=self._timeout_s)
            except Exception:
                http_client = None
            self._client = OpenAI(api_key=self._api_key, http_client=http_client) if http_client is not None else OpenAI(api_key=self._api_key, timeout=self._timeout_s)
        except Exception:
            self._client = None  # type: ignore

    def _save_cache(self, key: str, value: Dict[str, Any]) -> None:
        self._cache[key] = value
        try:
            with open(self._cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _preprocess(self, image_path: str) -> bytes | None:
        try:
            from PIL import Image  # type: ignore
        except Exception:
            return None
        try:
            from .ai_analysis_service import _preprocess_image_for_model  # reuse
        except Exception:
            return None
        max_side = 512
        jpeg_quality = 60
        target_bytes = 200000
        min_side = 320
        return _preprocess_image_for_model(
            image_path,
            max_side=max_side,
            jpeg_quality=jpeg_quality,
            target_bytes=target_bytes,
            min_side=min_side,
        )

    def _acquire_slot(self) -> None:
        if self._lock is None:
            return
        while True:
            try:
                with self._lock:  # type: ignore
                    if self._inflight < self._max_parallel:
                        self._inflight += 1
                        return
            except Exception:
                return
            try:
                import time as _time
                _time.sleep(0.01)
            except Exception:
                return

    def _release_slot(self) -> None:
        if self._lock is None:
            return
        try:
            with self._lock:  # type: ignore
                if self._inflight > 0:
                    self._inflight -= 1
        except Exception:
            pass

    def _tune_parallel(self, ok: bool) -> None:
        if self._lock is None:
            return
        try:
            with self._lock:  # type: ignore
                if ok:
                    if self._max_parallel < 64:
                        self._max_parallel += 1
                else:
                    if self._max_parallel > 2:
                        self._max_parallel = max(2, self._max_parallel // 2)
        except Exception:
            pass

    def verify(self, image_path: str, query_text: str) -> Dict[str, Any]:
        if not self._api_key:
            # 로깅: API 키 없음
            try:
                _log.warning("verify_no_api_key | file=%s", os.path.basename(image_path))
            except Exception:
                pass
            return {"match": False, "confidence": 0.0, "reasons": "API 키 없음"}
        import time
        t0 = time.monotonic()
        try:
            _log.info("verify_start | model=%s | file=%s", str(getattr(self, "_model", "?")), os.path.basename(image_path))
        except Exception:
            pass
        img = self._preprocess(image_path)
        if not img:
            try:
                _log.warning("verify_preprocess_fail | file=%s", os.path.basename(image_path))
            except Exception:
                pass
            return {"match": False, "confidence": 0.0, "reasons": "이미지 전처리 실패"}
        t1 = time.monotonic()
        key = _sha1(img) + "|" + _sha1(query_text.encode("utf-8"))
        if key in self._cache:
            out = dict(self._cache[key])
            try:
                _log.info("verify_cache_hit | file=%s | conf=%.3f | dt_pre=%.3fs", os.path.basename(image_path), float(out.get("confidence", 0.0)), (t1 - t0))
            except Exception:
                pass
            return out

        # 프롬프트 강화: 색상/객관 요소 중시, 허위 추론 금지, JSON 강제
        system_msg = (
            "당신은 사진 검증 어시스턴트다. 사용자 질의와 이미지의 일치 여부를 판단한다.\n"
            "- 반드시 JSON으로만 답한다(여분 텍스트 금지).\n"
            "- 보이는 정보에만 근거한다. 추측/환각 금지.\n"
            "- 질의에 특정 색상/수량/객체가 명시되면 해당 요소 부재 시 match=false로 판정한다.\n"
        )
        instruction = (
            "스키마:\n"
            '{"match": true|false, "confidence": 0.0~1.0, "reasons": "간단 근거"}\n'
            "판정 기준:\n"
            "- 장면/주요 객체/행동/관계/색상을 종합 판단.\n"
            "- 색상 언급이 있는 경우, 해당 색이 핵심 대상에 실제로 보이는지 확인.\n"
            "- 불명확하면 match=false에 가깝게 낮은 confidence로.\n"
        )

        import base64
        from openai import OpenAI  # type: ignore
        import httpx  # type: ignore
        timeout_s = float(getattr(self, "_timeout_s", 10.0))
        top_p = float(getattr(self, "_top_p", 1.0))
        n = max(1, min(8, int(getattr(self, "_n", 1))))
        # 프록시 감지 및 주입
        try:
            hp = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
            sp = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
            ap = os.getenv("ALL_PROXY") or os.getenv("all_proxy")
            http_client = None
            if ap:
                http_client = httpx.Client(proxy=ap, timeout=timeout_s)
            elif hp or sp:
                mounts = {}
                if hp:
                    mounts["http://"] = httpx.HTTPTransport(proxy=hp)
                if sp:
                    mounts["https://"] = httpx.HTTPTransport(proxy=sp)
                if mounts:
                    http_client = httpx.Client(mounts=mounts, timeout=timeout_s)
            client = OpenAI(api_key=self._api_key, http_client=http_client) if http_client is not None else OpenAI(api_key=self._api_key, timeout=timeout_s)
        except Exception:
            client = OpenAI(api_key=self._api_key, timeout=timeout_s)
        b64 = base64.b64encode(img).decode("ascii")
        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction + "\n장면 설명:" + query_text},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                ],
            },
        ]
        try:
            t2 = time.monotonic()
            resp = client.chat.completions.create(
                model=self._model,
                messages=messages,
                top_p=top_p,
                n=n,
            )
            t3 = time.monotonic()
            best_conf = 0.0
            best = {"match": False, "confidence": 0.0, "reasons": ""}
            # 집계 방식: 기본 max
            agg = getattr(self, "_agg", "max")
            confs: List[float] = []
            outs: List[Dict[str, Any]] = []
            for ch in resp.choices:
                txt = (getattr(getattr(ch, "message", None), "content", None) or "{}")
                try:
                    data = json.loads(txt)
                    out = {
                        "match": bool(data.get("match", False)),
                        "confidence": float(data.get("confidence", 0.0)),
                        "reasons": str(data.get("reasons", "")),
                    }
                except Exception:
                    out = {"match": False, "confidence": 0.0, "reasons": "parse_fail"}
                outs.append(out)
                confs.append(float(out.get("confidence", 0.0)))
                # 각 선택지 로깅
                try:
                    _log.info("verify_choice | conf=%.3f | match=%s | reason_len=%d", float(out.get("confidence", 0.0)), bool(out.get("match", False)), len(str(out.get("reasons", ""))))
                except Exception:
                    pass
                if float(out.get("confidence", 0.0)) > best_conf:
                    best_conf = float(out.get("confidence", 0.0))
                    best = out
            if agg == "mean" and confs:
                mean_conf = sum(confs) / float(len(confs))
                # mean에서는 match를 best 기준으로 유지, confidence만 평균
                best = dict(best)
                best["confidence"] = float(mean_conf)
            self._save_cache(key, best)
            try:
                _log.info("verify_ok | file=%s | model=%s | conf=%.3f | n=%d | dt_pre=%.3fs | dt_api=%.3fs", os.path.basename(image_path), str(getattr(self, "_model", "?")), float(best.get("confidence", 0.0)), int(n), (t1 - t0), (t3 - t2))
            except Exception:
                pass
            return best
        except Exception as e:
            try:
                _log.warning("verify_fail | err=%s", str(e))
            except Exception:
                pass
            return {"match": False, "confidence": 0.0, "reasons": "검증 실패"}

    def verify_binary(self, image_path: str, query_text: str) -> Dict[str, Any]:
        """강화 프롬프트 기반 최종 예/아니오 판정.
        스키마: { ok: bool, match: bool, confidence: float, reasons: str }
        """
        if not self._api_key:
            try:
                _log.warning("verify_bin_no_api_key | file=%s", os.path.basename(image_path))
            except Exception:
                pass
            return {"ok": False, "match": False, "confidence": 0.0, "reasons": "no_api_key"}

        img = self._preprocess(image_path)
        if not img:
            try:
                _log.warning("verify_bin_preprocess_fail | file=%s", os.path.basename(image_path))
            except Exception:
                pass
            return {"ok": False, "match": False, "confidence": 0.0, "reasons": "preprocess_fail"}

        key = _sha1(img) + "|bin|" + _sha1(query_text.encode("utf-8"))
        if key in self._cache:
            v = dict(self._cache[key])
            # 구 캐시 스키마 호환
            if "ok" in v and "match" in v:
                return v
            if "ok" in v:
                return {"ok": bool(v.get("ok")), "match": bool(v.get("match", v.get("ok", False))), "confidence": float(v.get("confidence", 0.0)), "reasons": str(v.get("reasons", ""))}
            return {"ok": bool(v.get("match", False)), "match": bool(v.get("match", False)), "confidence": float(v.get("confidence", 0.0)), "reasons": str(v.get("reasons", ""))}

        # 강화 프롬프트: match 판별/근거 포함, JSON 강제
        system_msg = (
            "당신은 사진 검증 어시스턴트다. 사용자 질의와 이미지의 일치 여부를 판단한다.\n"
            "- 반드시 JSON으로만 답한다(여분 텍스트 금지).\n"
            "- 보이는 정보에만 근거한다. 추측/환각 금지.\n"
            "- 질의에 특정 색상/수량/객체가 명시되면 해당 요소 부재 시 match=false로 판정한다.\n"
        )
        instruction = (
            "스키마:\n"
            '{"match": true|false, "confidence": 0.0~1.0, "reasons": "간단 근거"}\n'
            "판정 기준:\n"
            "- 장면/주요 객체/행동/관계/색상을 종합 판단.\n"
            "- 색상 언급이 있는 경우, 해당 색이 핵심 대상에 실제로 보이는지 확인.\n"
            "- 불명확하면 match=false에 가깝게 낮은 confidence로.\n"
            "- 이 사진은 '" + str(query_text) + "' 키워드를 만족하는가? 만족하면 match=true, 만족하지 않으면 match=false로 판단한다.\n"
        )

        import base64
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=self._api_key, timeout=float(getattr(self, "_timeout_s", 10.0)))
        b64 = base64.b64encode(img).decode("ascii")
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            ]},
        ]
        try:
            # 재시도 + 동시성 제한(가변)
            attempts = 0
            last_exc: Exception | None = None
            resp = None
            while attempts < 3:
                attempts += 1
                try:
                    self._acquire_slot()
                    try:
                        client2 = getattr(self, "_client", None)
                        if client2 is None:
                            from openai import OpenAI  # type: ignore
                            client2 = OpenAI(api_key=self._api_key, timeout=float(getattr(self, "_timeout_s", 10.0)))
                        resp = client2.chat.completions.create(
                            model=self._model,
                            messages=messages,
                            top_p=float(getattr(self, "_top_p", 1.0)),
                            n=max(1, min(4, int(getattr(self, "_n", 1)))),
                        )
                    finally:
                        self._release_slot()
                    # 성공 → 동시성 상향 시도
                    self._tune_parallel(True)
                    break
                except Exception as e:
                    last_exc = e
                    # 실패 → 동시성 하향
                    self._tune_parallel(False)
                    try:
                        import time as _time, random as _rand
                        _time.sleep(0.35 * attempts * (1.0 + 0.2 * _rand.random()))
                    except Exception:
                        pass
            if resp is None and last_exc is not None:
                raise last_exc

            best_conf = 0.0
            best_ok = False
            best_match = False
            best_reasons = ""
            for ch in resp.choices:  # type: ignore[attr-defined]
                raw = (getattr(getattr(ch, "message", None), "content", None) or "{}")
                txt = raw
                try:
                    if isinstance(txt, str):
                        s = txt.find("{")
                        e = txt.rfind("}")
                        if s != -1 and e != -1 and e > s:
                            txt = txt[s:e + 1]
                    data = json.loads(txt)
                    match = bool(data.get("match", data.get("ok", False)))
                    ok = bool(match)
                    conf = float(data.get("confidence", 0.0))
                    reasons = str(data.get("reasons", ""))
                except Exception:
                    match = False
                    ok = False
                    conf = 0.0
                    reasons = "parse_fail"
                if conf > best_conf:
                    best_conf = conf
                    best_ok = ok
                    best_match = match
                    best_reasons = reasons
            out = {"ok": bool(best_ok), "match": bool(best_match), "confidence": float(best_conf), "reasons": str(best_reasons)}
            self._save_cache(key, out)
            return out
        except Exception as e:
            try:
                _log.warning("verify_bin_fail | err=%s", str(e))
            except Exception:
                pass
            return {"ok": False, "match": False, "confidence": 0.0, "reasons": "fail"}

    @staticmethod
    def pass_threshold(conf: float, mode: str) -> bool:
        # mode: loose/normal/strict (정확도 우선으로 상향)
        if mode == "strict":
            t = 0.75
        elif mode == "loose":
            t = 0.50
        else:
            t = 0.65
        try:
            return float(conf) >= float(t)
        except Exception:
            return False


