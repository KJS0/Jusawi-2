from __future__ import annotations

import os, json
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import imagehash  # type: ignore
    _HAS_PHASH = True
except Exception:
    _HAS_PHASH = False

try:  # optional ANN
    import hnswlib  # type: ignore
    _HAS_HNSW = True
except Exception:
    _HAS_HNSW = False

# Optional offline CLIP fallback (SentenceTransformers/OpenCLIP via OfflineVerifierService)
try:
    from .offline_verifier import OfflineVerifierService  # type: ignore
    _HAS_OFFLINE = True
except Exception:
    _HAS_OFFLINE = False

from ..utils.logging_setup import get_logger  # type: ignore
_log = get_logger("svc.Similarity")

# Default workers for pHash parallel computation (source-controlled)
_PHASH_WORKERS_DEFAULT = 8

_SUPPORTED = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff','.gif'}

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    # 벡터 모양 강제: (512,), (N,) 형태로 평탄화하여 내적 정렬 오류 방지
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


class SimilarityIndex:
    def __init__(self, phash_workers: int | None = None):
        self._model = None
        self._cache_dir = os.path.join(os.path.expanduser("~"), ".jusawi_sim_cache")
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
        except Exception:
            pass
        self._index: Dict[str, Dict] = {}
        self._offline = None
        try:
            n = int(phash_workers) if phash_workers is not None else int(_PHASH_WORKERS_DEFAULT)
        except Exception:
            n = int(_PHASH_WORKERS_DEFAULT)
        if n <= 0:
            n = int(_PHASH_WORKERS_DEFAULT)
        self._phash_workers = n

    def _ensure_model(self):
        if self._model is None and _HAS_CLIP:
            # 로컬 모델 폴더 우선 사용: <repo_root>/models/clip-ViT-B-32
            try:
                base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                local_dir = os.path.join(base, "models", "clip-ViT-B-32")
            except Exception:
                local_dir = ""
            if local_dir and os.path.isdir(local_dir):
                try:
                    self._model = SentenceTransformer(local_dir)  # type: ignore
                    return
                except Exception:
                    pass
            # 기본: 허브에서 로드(캐시 사용)
            self._model = SentenceTransformer("clip-ViT-B-32")

    def _is_image(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in _SUPPORTED

    def _vec_image(self, path: str) -> Optional[np.ndarray]:
        if Image is None:
            return None
        try:
            if _HAS_CLIP:
                self._ensure_model()
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    # 일부 SentenceTransformer 버전은 images= 키워드를 받지 않으므로 리스트로 첫 인자 전달
                    v = self._model.encode([im], convert_to_numpy=True, normalize_embeddings=True)  # type: ignore
                arr = np.asarray(v, dtype=np.float32)
                # (1, D) -> (D,)로 정규화
                if arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr[0]
                return arr
            elif _HAS_OFFLINE:
                try:
                    if self._offline is None:
                        self._offline = OfflineVerifierService()
                    vec = self._offline._get_or_compute_image_vec(path)  # type: ignore[attr-defined]
                    if vec is not None:
                        arr = np.asarray(vec, dtype=np.float32)
                        if arr.ndim == 2 and 1 in arr.shape:
                            arr = arr.ravel()
                        return arr
                except Exception:
                    pass
            # offline 실패 시 pHash로 폴백 시도
            if _HAS_PHASH:
                with Image.open(path) as im:
                    ph = imagehash.phash(im)
                bits = np.unpackbits(np.array([int(str(ph),16)], dtype='>u8').view('>u1'))
                vec = (bits.astype(np.float32) * 2.0 - 1.0)
                return vec
        except Exception:
            return None
        return None

    def _vec_text(self, text: str) -> Optional[np.ndarray]:
        if not text.strip() or not _HAS_CLIP:
            return None
        try:
            self._ensure_model()
            v = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)  # type: ignore
            return np.asarray(v[0], dtype=np.float32)
        except Exception:
            return None

    def _phash_int(self, path: str, hash_size: int = 8) -> Optional[int]:
        """Perceptual hash as integer. hash_size=8 -> 64bit, 16 -> 256bit."""
        if not _HAS_PHASH or Image is None:
            return None
        try:
            with Image.open(path) as im:
                h = imagehash.phash(im, hash_size=hash_size)
            return int(str(h), 16)
        except Exception:
            return None

    def _phash_pair(self, path: str) -> Tuple[Optional[int], Optional[int]]:
        if not _HAS_PHASH or Image is None:
            return None, None
        try:
            with Image.open(path) as im:
                h8 = imagehash.phash(im, hash_size=8)
                try:
                    h16 = imagehash.phash(im, hash_size=16)
                except Exception:
                    h16 = None
            v64 = int(str(h8), 16) if h8 is not None else None
            v256 = int(str(h16), 16) if h16 is not None else None
            return v64, v256
        except Exception:
            return None, None

    def _hamming(self, a: int, b: int) -> int:
        try:
            return (a ^ b).bit_count()
        except Exception:
            # Py<3.8 호환
            x = a ^ b
            cnt = 0
            while x:
                x &= x - 1
                cnt += 1
            return cnt

    def _sig(self, path: str) -> Tuple[int,int]:
        st = os.stat(path)
        return int(st.st_mtime), int(st.st_size)

    def _cache_path(self, dir_path: str) -> str:
        key = dir_path.replace(":","_").replace("\\","_").replace("/","_")
        return os.path.join(self._cache_dir, f"{key}.json")

    def build_or_load(self, dir_path: str) -> None:
        cp = self._cache_path(dir_path)
        try:
            _log.info("sim_build_or_load_start | dir=%s", os.path.basename(dir_path) or dir_path)
        except Exception:
            pass
        try:
            if os.path.exists(cp):
                with open(cp, "r", encoding="utf-8") as fh:
                    self._index = json.load(fh)
        except Exception:
            self._index = {}
        # 현재 임베딩 엔진 식별자: clip | offline | phash | ''
        cur_engine = ("clip" if _HAS_CLIP else ("offline" if _HAS_OFFLINE else ("phash" if _HAS_PHASH else "")))
        to_compute_phash: List[str] = []
        for name in os.listdir(dir_path):
            p = os.path.join(dir_path, name)
            if not os.path.isfile(p) or not self._is_image(p):
                continue
            try:
                m, s = self._sig(p)
                rec = self._index.get(p)
                if rec and rec.get("mtime") == m and rec.get("size") == s and rec.get("engine") == cur_engine:
                    continue
                vec = self._vec_image(p)
                if vec is None:
                    continue
                rec: Dict[str, object] = {"vec": vec.tolist(), "mtime": m, "size": s, "engine": cur_engine, "dim": int(vec.shape[0])}
                self._index[p] = rec
                if _HAS_PHASH:
                    to_compute_phash.append(p)
            except Exception:
                pass
        self._index = {k:v for k,v in self._index.items() if os.path.exists(k)}
        # Batch-compute pHash in parallel to speed up folder-wide preparation
        if _HAS_PHASH and to_compute_phash:
            try:
                _log.info("sim_phash_batch_start | dir=%s | files=%d | workers=%d", os.path.basename(dir_path) or dir_path, len(to_compute_phash), int(self._phash_workers))
            except Exception:
                pass
            try:
                with ThreadPoolExecutor(max_workers=int(self._phash_workers)) as ex:
                    futs = {ex.submit(self._phash_pair, p): p for p in to_compute_phash}
                    for fut in as_completed(futs):
                        p = futs.get(fut)
                        try:
                            ph64, ph256 = fut.result()
                        except Exception:
                            ph64, ph256 = None, None
                        rec = self._index.get(p or "")
                        if not isinstance(rec, dict):
                            continue
                        if ph64 is not None:
                            rec["phash"] = ph64
                            rec["phash_bits"] = 64
                        if ph256 is not None:
                            rec["phash256"] = ph256
                            rec["phash256_bits"] = 256
            except Exception:
                pass
            try:
                _log.info("sim_phash_batch_done | dir=%s | files=%d", os.path.basename(dir_path) or dir_path, len(to_compute_phash))
            except Exception:
                pass
        try:
            with open(cp, "w", encoding="utf-8") as fh:
                json.dump(self._index, fh)
        except Exception:
            pass
        try:
            _log.info("sim_build_or_load_done | dir=%s | count=%d | engine=%s", os.path.basename(dir_path) or dir_path, len(self._index), cur_engine)
        except Exception:
            pass

    def similar(self, anchor_path: str, dir_path: str, query_text: str = "", alpha: float = 0.7, top_k: int = 50) -> List[Tuple[str, float]]:
        alpha = float(max(0.0, min(1.0, alpha)))
        self.build_or_load(dir_path)
        a = self._vec_image(anchor_path)
        if a is None:
            return []
        t = self._vec_text(query_text) if query_text else None
        if t is None or a.shape[0] != t.shape[0]:
            q = a
        else:
            q = alpha * a + (1.0 - alpha) * t
        out: List[Tuple[str,float]] = []
        for p, rec in self._index.items():
            if os.path.normcase(p) == os.path.normcase(anchor_path):
                continue
            v = np.asarray(rec.get("vec") or [], dtype=np.float32)
            if v.size == 0:
                continue
            score = _cos(q, v)
            out.append((p, score))
        out.sort(key=lambda x: x[1], reverse=True)
        res = out[:top_k]
        try:
            _log.info("sim_plain | dir=%s | top_k=%d | got=%d", os.path.basename(dir_path) or dir_path, top_k, len(res))
        except Exception:
            pass
        return res

    def similar_fast(self, anchor_path: str, dir_path: str, top_k: int = 50, preselect: int = 300) -> List[Tuple[str, float]]:
        """pHash로 후보 선별 후 CLIP으로 재랭크하는 2단계 검색."""
        self.build_or_load(dir_path)
        # prefer 256-bit phash for better discrimination; fallback to 64-bit
        ah256 = self._phash_int(anchor_path, 16)
        ah64 = self._phash_int(anchor_path, 8)
        cand: List[str] = []
        if ah256 is not None or ah64 is not None:
            tmp: List[Tuple[str, float, str]] = []  # (path, norm_dist, mode)
            for p, rec in self._index.items():
                if os.path.normcase(p) == os.path.normcase(anchor_path):
                    continue
                # try 256-bit first
                ph256 = rec.get("phash256")
                if isinstance(ph256, int) and ah256 is not None:
                    d = self._hamming(ah256, int(ph256))
                    norm = float(d) / 256.0
                    tmp.append((p, norm, "256"))
                    continue
                ph64 = rec.get("phash")
                if isinstance(ph64, int) and ah64 is not None:
                    d = self._hamming(ah64, int(ph64))
                    norm = float(d) / 64.0
                    tmp.append((p, norm, "64"))
            # smaller normalized distance first
            tmp.sort(key=lambda x: x[1])
            cand = [p for (p, _, __) in tmp[:max(preselect, top_k*5)]]
        if not cand:
            cand = [p for p in self._index.keys() if os.path.normcase(p) != os.path.normcase(anchor_path)]

        a = self._vec_image(anchor_path)
        if a is None:
            # 임베딩 실패 시 pHash 기준 근접 이웃만 반환 (비트수 기반 정규화)
            tmp: List[Tuple[str, float]] = []
            for p, rec in self._index.items():
                if os.path.normcase(p) == os.path.normcase(anchor_path):
                    continue
                if ah256 is not None and isinstance(rec.get("phash256"), int):
                    d = self._hamming(ah256, int(rec.get("phash256") or 0))
                    norm = float(min(max(d, 0), 256)) / 256.0
                    tmp.append((p, 1.0 - norm))
                elif ah64 is not None and isinstance(rec.get("phash"), int):
                    d = self._hamming(ah64, int(rec.get("phash") or 0))
                    norm = float(min(max(d, 0), 64)) / 64.0
                    tmp.append((p, 1.0 - norm))
            tmp.sort(key=lambda x: x[1], reverse=True)
            out_ph = tmp[:top_k]
            try:
                _log.info("sim_fast_phash_only | dir=%s | top_k=%d | got=%d", os.path.basename(dir_path) or dir_path, top_k, len(out_ph))
            except Exception:
                pass
            return out_ph
        out: List[Tuple[str,float]] = []
        for p in cand:
            v = np.asarray(self._index[p].get("vec") or [], dtype=np.float32)
            if v.size == 0:
                v2 = self._vec_image(p)
                if v2 is None:
                    continue
                self._index[p]["vec"] = v2.tolist()
                v = v2
            out.append((p, _cos(a, v)))
        out.sort(key=lambda x: x[1], reverse=True)
        res = out[:top_k]
        try:
            _log.info("sim_fast | dir=%s | preselect=%d | top_k=%d | got=%d", os.path.basename(dir_path) or dir_path, preselect, top_k, len(res))
        except Exception:
            pass
        return res

    def similar_hnsw(self, anchor_path: str, dir_path: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """HNSW ANN 기반 근사 최근접. 대용량 폴더에 적합."""
        if not _HAS_HNSW:
            return self.similar_fast(anchor_path, dir_path, top_k=top_k)
        self.build_or_load(dir_path)
        vecs: List[np.ndarray] = []
        paths: List[str] = []
        for p, rec in self._index.items():
            if os.path.normcase(p) == os.path.normcase(anchor_path):
                continue
            v = np.asarray(rec.get("vec") or [], dtype=np.float32)
            if v.ndim > 1:
                v = v.ravel()
            if v.size:
                vecs.append(v)
                paths.append(p)
        if not vecs:
            return []
        X = np.vstack(vecs).astype(np.float32)
        dim = X.shape[1]
        # cosine 유사도를 위해 내적 기반, 벡터 정규화
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        index = hnswlib.Index(space='ip', dim=dim)
        index.init_index(max_elements=Xn.shape[0], ef_construction=200, M=16)
        index.add_items(Xn)
        index.set_ef(min(200, max(50, top_k*10)))
        q = self._vec_image(anchor_path)
        if q is None:
            return []
        if q.ndim > 1:
            q = q.ravel()
        qn = q / (np.linalg.norm(q) + 1e-9)
        labels, dists = index.knn_query(qn, k=min(top_k+5, Xn.shape[0]))
        out: List[Tuple[str,float]] = []
        for lbl, dist in zip(labels[0], dists[0]):
            i = int(lbl)
            score = float(dist)
            if 0 <= i < len(paths):
                out.append((paths[i], score))
        out.sort(key=lambda x: x[1], reverse=True)
        res = out[:top_k]
        try:
            _log.info("sim_hnsw | dir=%s | top_k=%d | got=%d", os.path.basename(dir_path) or dir_path, top_k, len(res))
        except Exception:
            pass
        return res

    def similar_auto(self, anchor_path: str, dir_path: str, top_k: int = 50, mode: str = "auto") -> List[Tuple[str,float]]:
        """auto: 파일 수 기준(hnsw>fast>plain), hnsw: ANN, fast: pHash+CLIP, plain: CLIP only"""
        try:
            n = len(self._index) if self._index else len(os.listdir(dir_path))
        except Exception:
            n = 0
        if mode == 'plain':
            try:
                _log.info("sim_auto_select | dir=%s | mode=%s | n=%d", os.path.basename(dir_path) or dir_path, "plain", n)
            except Exception:
                pass
            return self.similar(anchor_path, dir_path, top_k=top_k)
        if mode == 'fast':
            try:
                _log.info("sim_auto_select | dir=%s | mode=%s | n=%d", os.path.basename(dir_path) or dir_path, "fast", n)
            except Exception:
                pass
            return self.similar_fast(anchor_path, dir_path, top_k=top_k)
        if mode == 'hnsw' or (mode == 'auto' and n >= 1000 and _HAS_HNSW):
            try:
                _log.info("sim_auto_select | dir=%s | mode=%s | n=%d", os.path.basename(dir_path) or dir_path, "hnsw", n)
            except Exception:
                pass
            return self.similar_hnsw(anchor_path, dir_path, top_k=top_k)
        try:
            _log.info("sim_auto_select | dir=%s | mode=%s | n=%d", os.path.basename(dir_path) or dir_path, "fast(auto)", n)
        except Exception:
            pass
        return self.similar_fast(anchor_path, dir_path, top_k=top_k)

    # --- 중복/변형판 탐지: pHash 선별 + CLIP 확증 ---
    def find_near_duplicates(self, dir_path: str, phash_max_hamming: int = 8, clip_min_score: float = 0.96) -> List[Tuple[str, str, float]]:
        """
        디렉터리 내 근접 중복 후보를 반환.
        반환: [(path_a, path_b, score)] with score=CLIP 유사도
        """
        self.build_or_load(dir_path)
        paths = sorted(list(self._index.keys()))
        out: List[Tuple[str, str, float]] = []
        # pHash가 없으면 CLIP만으로는 O(N^2) 비용이라 후보를 줄이기 어려움 → pHash 필수
        if not _HAS_PHASH:
            return out
        # pHash 버킷으로 근접 후보 수집
        phashes: List[Tuple[str, int]] = []
        for p, rec in self._index.items():
            ph = rec.get("phash")
            if isinstance(ph, int):
                phashes.append((p, int(ph)))
        for i in range(len(phashes)):
            pi, hi = phashes[i]
            for j in range(i + 1, len(phashes)):
                pj, hj = phashes[j]
                d = self._hamming(hi, hj)
                if d <= int(phash_max_hamming):
                    # CLIP 유사도로 확증
                    try:
                        ai = self._vec_image(pi)
                        aj = self._vec_image(pj)
                        if ai is None or aj is None:
                            continue
                        sc = _cos(ai, aj)
                        if sc >= float(clip_min_score):
                            out.append((pi, pj, float(sc)))
                    except Exception:
                        pass
        out.sort(key=lambda x: x[2], reverse=True)
        return out


