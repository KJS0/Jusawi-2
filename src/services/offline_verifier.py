from __future__ import annotations

import os
import io
import sqlite3
import json
from typing import List, Tuple, Optional
import inspect
import importlib

try:  # optional dependency
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

SentenceTransformer = None  # lazy import
open_clip = None  # lazy import
torch = None  # lazy import

from ..utils.logging_setup import get_logger

_log = get_logger("svc.OfflineVerifier")


class OfflineVerifierService:
    """
    오프라인 CLIP 임베딩으로 텍스트-이미지 유사도 기반 재검증/재정렬.
    - 의존성: sentence_transformers (선택). 없으면 available=False.
    - 이미지 벡터는 ~/.jusawi/offline_clip.sqlite3 에 캐시.
    """

    def __init__(self):
        # 텍스트/이미지 모델을 분리해 항상 이미지 임베딩이 가능하도록 구성
        self._text_model_name = os.getenv(
            "OFFLINE_CLIP_TEXT_MODEL",
            "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        ).strip()
        self._image_model_name = os.getenv(
            "OFFLINE_CLIP_IMAGE_MODEL",
            "clip-ViT-B-32",
        ).strip()
        self._st_available = True  # determined on demand
        self._st_supports_images = False  # maintained for open_clip fallback logic
        self._text_model: Optional[SentenceTransformer] = None
        self._image_model: Optional[SentenceTransformer] = None
        # open_clip fallback
        self._oc_available = True  # determined on demand
        self._oc_model = None
        self._oc_preprocess = None
        self._oc_tokenizer = None
        self._device = "cpu"
        self.engine = ""  # 'st' | 'open_clip' | ''
        self._force = ""
        # 이미지 모델 실패 캐시 및 HF-CLIP 폴백 보관
        self._image_model_failed = False
        self._hf_clip_model = None
        self._hf_clip_processor = None
        base_dir = os.path.join(os.path.expanduser("~"), ".jusawi")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            pass
        self._db_path = os.path.join(base_dir, "offline_clip.sqlite3")
        self._ensure_db()
        # 더 이상 존재하지 않는 속성 접근을 제거하여 초기화 오류 방지

    @property
    def available(self) -> bool:
        return bool(self._st_available or self._oc_available)

    def is_ready(self) -> bool:
        """엔진이 실제로 임베딩을 계산할 준비가 되었는지 확인."""
        # SentenceTransformers 경로(텍스트/이미지 모델 각각 확인)
        if self._st_available:
            ok_txt = self._ensure_text_model()
            ok_img = self._ensure_image_model()
            if ok_txt and ok_img:
                self.engine = "st"
                return True
        # OpenCLIP 경로
        if self._oc_available:
            if self._oc_model is None:
                self._ensure_open_clip()
            if self._oc_model is not None and self._oc_preprocess is not None:
                self.engine = "open_clip"
                return True
        return False

    def prepare(self) -> tuple[bool, str]:
        """엔진을 로드해 사용할 준비를 한다. (ready, engine|error)"""
        # Auto: Try ST (텍스트/이미지), then OpenCLIP
        ok_txt = self._ensure_text_model()
        ok_img = self._ensure_image_model()
        if ok_txt and ok_img:
            self.engine = "st"
            try:
                _log.info("offline_ready | engine=st | text=%s | image=%s", self._text_model_name, self._image_model_name)
            except Exception:
                pass
            return True, self.engine
        if self._ensure_open_clip():
            self.engine = "open_clip"
            try:
                _log.info("offline_ready | engine=open_clip")
            except Exception:
                pass
            return True, self.engine
        return False, "no_engine"

    def _ensure_text_model(self) -> bool:
        if not self._st_available:
            return False
        if self._text_model is not None:
            return True
        try:
            # Lazy import here
            global SentenceTransformer
            if SentenceTransformer is None:
                try:
                    from sentence_transformers import SentenceTransformer as _ST  # type: ignore
                    SentenceTransformer = _ST
                except Exception as ie:
                    try:
                        _log.warning("st_import_fail | err=%s", str(ie))
                    except Exception:
                        pass
                    self._st_available = False
                    return False
            candidates = []
            # 환경변수 경로 우선
            env_txt = os.getenv("JUSAWI_CLIP_TEXT_MODEL", "").strip()
            if env_txt and os.path.isdir(env_txt):
                candidates.append(env_txt)
            # 로컬 레포 모델 폴더
            try:
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                local_txt = os.path.join(repo_root, "models", "clip-ViT-B-32-multilingual-v1")
                if os.path.isdir(local_txt):
                    candidates.append(local_txt)
            except Exception:
                pass
            # 허브 이름들
            candidates.append(self._text_model_name)
            # 레포 접두어 누락 시 보정 후보 추가
            if "/" not in self._text_model_name:
                candidates.append("sentence-transformers/" + self._text_model_name)
            # 일반 단일언어 폴백(텍스트)
            candidates.append("sentence-transformers/clip-ViT-B-32")
            last_err = None
            for name in candidates:
                try:
                    self._text_model = SentenceTransformer(name)
                    return True
                except Exception as e:
                    last_err = e
                    continue
            raise last_err or RuntimeError("model_load_failed")
        except Exception as e:
            try:
                _log.warning("offline_text_model_load_fail | model=%s | err=%s", self._text_model_name, str(e))
            except Exception:
                pass
            self._st_available = False
            return False

    def _ensure_image_model(self) -> bool:
        if not self._st_available:
            return False
        if self._image_model_failed:
            # 이전에 이미지 모델 로딩이 실패했으면 재시도하지 않음
            return bool(self._image_model is not None or self._hf_clip_model is not None)
        if self._image_model is not None:
            return True
        try:
            # Lazy import here
            global SentenceTransformer
            if SentenceTransformer is None:
                try:
                    from sentence_transformers import SentenceTransformer as _ST  # type: ignore
                    SentenceTransformer = _ST
                except Exception as ie:
                    try:
                        _log.warning("st_import_fail | err=%s", str(ie))
                    except Exception:
                        pass
                    self._st_available = False
                    return False
            candidates = []
            # 환경변수 경로 우선
            env_img = os.getenv("JUSAWI_CLIP_IMAGE_MODEL", "").strip()
            if env_img and os.path.isdir(env_img):
                candidates.append(env_img)
            # 로컬 레포 모델 폴더
            try:
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                local_img_a = os.path.join(repo_root, "models", "clip-ViT-B-32")
                local_img_b = os.path.join(repo_root, "models", "clip-ViT-B-32-laion2B-s34B-b79K")
                if os.path.isdir(local_img_a):
                    candidates.append(local_img_a)
                if os.path.isdir(local_img_b):
                    candidates.append(local_img_b)
            except Exception:
                pass
            # 허브 이름들
            candidates.append(self._image_model_name)
            if "/" not in self._image_model_name:
                candidates.append("sentence-transformers/" + self._image_model_name)
            # 이미지 임베딩용 기본 CLIP
            candidates.append("sentence-transformers/clip-ViT-B-32")
            last_err = None
            for name in candidates:
                try:
                    m = SentenceTransformer(name)
                    # 이미지 인코딩 지원 여부 확인
                    # 일부 버전은 encode 시그니처에 images가 명시되지 않으나 **kwargs로 지원 → 런타임 소형 테스트
                    supports_images = False
                    try:
                        from PIL import Image as _PIL  # type: ignore
                        test = _PIL.new("RGB", (1, 1), (255, 0, 0))
                        _ = m.encode(images=[test], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
                        supports_images = True
                    except Exception:
                        supports_images = False
                    if supports_images and m is not None:
                        self._image_model = m
                        self._st_supports_images = True
                        return True
                except Exception as e:
                    last_err = e
                    continue
            # 여기까지 오면 ST 경로 실패 → HuggingFace Transformers CLIP 폴백
            try:
                from transformers import CLIPModel, CLIPProcessor  # type: ignore
            except Exception as te:
                try:
                    _log.warning("hf_clip_import_fail | err=%s", str(te))
                except Exception:
                    pass
                self._image_model_failed = True
                raise last_err or RuntimeError("image_model_load_failed")
            # 모델 이름 매핑
            hf_name = "openai/clip-vit-base-patch32"
            if "laion" in self._image_model_name.lower():
                # laion 변형도 기본 openai 가중치로 대체 가능(차이 미미)
                hf_name = "openai/clip-vit-base-patch32"
            try:
                self._hf_clip_model = CLIPModel.from_pretrained(hf_name)
                self._hf_clip_processor = CLIPProcessor.from_pretrained(hf_name)
                try:
                    _log.info("offline_ready | engine=hf-clip | model=%s", hf_name)
                except Exception:
                    pass
                return True
            except Exception as he:
                try:
                    _log.warning("hf_clip_model_load_fail | model=%s | err=%s", hf_name, str(he))
                except Exception:
                    pass
                self._image_model_failed = True
            raise last_err or RuntimeError("image_model_load_failed")
        except Exception as e:
            try:
                _log.warning("offline_image_model_load_fail | model=%s | err=%s", self._image_model_name, str(e))
            except Exception:
                pass
            return False

    def _ensure_open_clip(self) -> bool:
        if not self._oc_available:
            return False
        if self._oc_model is not None and self._oc_preprocess is not None and self._oc_tokenizer is not None:
            return True
        try:
            # Lazy import here
            global open_clip, torch
            if open_clip is None or torch is None:
                try:
                    import open_clip as _OC  # type: ignore
                    import torch as _TH  # type: ignore
                    open_clip = _OC
                    torch = _TH
                except Exception as ie:
                    try:
                        _log.warning("open_clip_import_fail | err=%s", str(ie))
                    except Exception:
                        pass
                    self._oc_available = False
                    return False
            oc_model_name = os.getenv("OPENCLIP_MODEL", "ViT-B-32").strip() or "ViT-B-32"
            oc_pretrained = os.getenv("OPENCLIP_PRETRAINED", "laion2b_s34b_b79k").strip() or "laion2b_s34b_b79k"
            m, preprocess, tokenizer = open_clip.create_model_and_transforms(oc_model_name, pretrained=oc_pretrained, device=self._device)
            m.eval()
            self._oc_model = m
            self._oc_preprocess = preprocess
            self._oc_tokenizer = tokenizer
            return True
        except Exception as e:
            try:
                _log.warning("open_clip_load_fail | err=%s", str(e))
            except Exception:
                pass
            self._oc_available = False
            return False

    def _ensure_db(self) -> None:
        con = sqlite3.connect(self._db_path)
        try:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    path TEXT PRIMARY KEY,
                    mtime INTEGER,
                    model TEXT,
                    dim INTEGER,
                    vec BLOB
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vectors_path ON vectors(path)")
            con.commit()
        finally:
            con.close()

    def _get_mtime(self, path: str) -> int:
        try:
            return int(os.path.getmtime(path))
        except Exception:
            return 0

    def _read_vec(self, b: bytes) -> List[float]:
        try:
            s = b.decode("utf-8")
            return [float(x) for x in s.split(",") if x]
        except Exception:
            return []

    def _write_vec(self, vec: List[float]) -> bytes:
        return (",".join(f"{x:.7f}" for x in vec)).encode("utf-8")

    def _embed_image(self, path: str) -> Optional[List[float]]:
        # 모델 준비(필요 시 1회)
        self._ensure_image_model()

        # 1) SentenceTransformers 이미지 모델 우선
        if self._image_model is not None and Image is not None:
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    vec = self._image_model.encode(
                        images=[im],
                        batch_size=1,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )
                    return list(vec[0].tolist())
            except Exception as e:
                try:
                    _log.warning("offline_img_embed_fail | file=%s | err=%s", os.path.basename(path), str(e))
                except Exception:
                    pass

        # 2) Fallback to HF-CLIP (transformers)
        if self._hf_clip_model is not None and self._hf_clip_processor is not None and Image is not None:
            try:
                try:
                    import torch as _TH  # type: ignore
                except Exception:
                    _TH = None  # type: ignore
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    inputs = self._hf_clip_processor(images=im, return_tensors="pt")
                    if _TH is not None:
                        with _TH.no_grad():
                            feat = self._hf_clip_model.get_image_features(**inputs)
                            feat = feat / feat.norm(dim=-1, keepdim=True)
                        return list(feat[0].cpu().tolist())
                    # torch 임포트 실패 시에도 시도(성능 저하 허용)
                    feat = self._hf_clip_model.get_image_features(**inputs)
                    # no normalization without torch; convert via numpy if possible
                    try:
                        import numpy as _NP  # type: ignore
                        arr = feat[0].detach().cpu().numpy() if hasattr(feat, "detach") else _NP.array(feat[0])
                        n = _NP.linalg.norm(arr) + 1e-9
                        arr = arr / n
                        return list(arr.tolist())
                    except Exception:
                        return None
            except Exception as e:
                try:
                    _log.warning("offline_img_embed_fail | hf_clip | file=%s | err=%s", os.path.basename(path), str(e))
                except Exception:
                    pass
                return None

        # 3) Fallback to open_clip
        if self._ensure_open_clip() and Image is not None:
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    img = self._oc_preprocess(im).unsqueeze(0)
                    with torch.no_grad():
                        feat = self._oc_model.encode_image(img)
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                    return list(feat[0].cpu().tolist())
            except Exception as e:
                try:
                    _log.warning("offline_img_embed_fail | file=%s | err=%s", os.path.basename(path), str(e))
                except Exception:
                    pass
                return None

    def _embed_text(self, text: str) -> Optional[List[float]]:
        if self._ensure_text_model():
            try:
                vec = self._text_model.encode(
                    sentences=[text or " "],
                    batch_size=1,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                return list(vec[0].tolist())
            except Exception as e:
                try:
                    _log.warning("offline_txt_embed_fail | err=%s", str(e))
                except Exception:
                    pass
        # Fallback to open_clip
        if self._ensure_open_clip():
            try:
                tok = self._oc_tokenizer([text or " "])
                with torch.no_grad():
                    feat = self._oc_model.encode_text(tok)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                return list(feat[0].cpu().tolist())
            except Exception as e:
                try:
                    _log.warning("offline_txt_embed_fail | oc | err=%s", str(e))
                except Exception:
                    pass
            return None

    def _get_or_compute_image_vec(self, path: str) -> Optional[List[float]]:
        mtime = self._get_mtime(path)
        con = sqlite3.connect(self._db_path)
        try:
            cur = con.cursor()
            cur.execute("SELECT mtime, dim, vec FROM vectors WHERE path=? AND model=?", (path, self._image_model_name))
            row = cur.fetchone()
            if row and int(row[0]) == mtime:
                return self._read_vec(row[2])
        except Exception:
            pass
        finally:
            con.close()
        vec = self._embed_image(path)
        if not vec:
            return None
        con = sqlite3.connect(self._db_path)
        try:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO vectors(path, mtime, model, dim, vec) VALUES(?,?,?,?,?)"
                " ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, model=excluded.model, dim=excluded.dim, vec=excluded.vec",
                (path, mtime, self._image_model_name, len(vec), self._write_vec(vec)),
            )
            con.commit()
        except Exception:
            pass
        finally:
            con.close()
        return vec

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i in range(len(a)):
            va = a[i]
            vb = b[i]
            dot += va * vb
            na += va * va
            nb += vb * vb
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        import math
        c = dot / (math.sqrt(na) * math.sqrt(nb))
        # 수치 안정화: [-1, 1]로 클램프
        if c > 1.0:
            c = 1.0
        elif c < -1.0:
            c = -1.0
        return c

    @staticmethod
    def threshold_for(mode: str) -> float:
        # 경험적 초기값. 데이터에 맞게 조정 가능
        if mode == "loose":
            return 0.24
        if mode == "strict":
            return 0.30
        return 0.27

    def rerank_offline(self, results: List[Tuple[str, float]], query_text: str, mode: str = "normal") -> List[Tuple[str, float]]:
        if not self.available:
            return results
        qvec = self._embed_text(query_text)
        if not qvec:
            return results
        th = self.threshold_for(mode)
        kept: List[Tuple[str, float]] = []
        for path, sim in results:
            ivec = self._get_or_compute_image_vec(path)
            if not ivec:
                continue
            c = self._cosine(qvec, ivec)
            # 최종 점수는 [0,1]로 제한
            score = float(max(0.0, min(1.0, c)))
            if score >= th:
                kept.append((path, float(score)))
        kept.sort(key=lambda x: x[1], reverse=True)
        return kept

    def search_offline(self, image_paths: List[str], query_text: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """완전 오프라인 검색: 모든 후보의 이미지 임베딩과 쿼리 임베딩(CLIP)으로 유사도를 계산해 상위 반환."""
        if not self.available:
            return []
        qvec = self._embed_text(query_text)
        if not qvec:
            return []
        # 배치 임베딩 선계산(없거나 오래된 항목만) — 기본 16장씩
        try:
            self.precompute_images(image_paths, batch_size=16)
        except Exception:
            pass
        scored: List[Tuple[str, float]] = []
        for p in image_paths:
            ivec = self._get_or_compute_image_vec(p)
            if not ivec:
                continue
            c = self._cosine(qvec, ivec)
            # 최종 점수는 [0,1]로 제한
            scored.append((p, float(max(0.0, min(1.0, c)))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, int(top_k))]

    # --- 배치 임베딩 사전계산(최적화) ---
    def precompute_images(self, image_paths: List[str], batch_size: int = 16) -> int:
        if not image_paths:
            return 0
        # ST 이미지 모델 또는 HF-CLIP 준비
        has_st = self._ensure_image_model() and (self._image_model is not None)
        has_hf = (self._hf_clip_model is not None and self._hf_clip_processor is not None)
        if not has_st and not has_hf:
            return 0
        # 캐시에 없는 대상 수집
        to_proc: List[Tuple[str, int]] = []
        con = sqlite3.connect(self._db_path)
        try:
            cur = con.cursor()
            for p in image_paths:
                try:
                    mt = int(os.path.getmtime(p))
                except Exception:
                    mt = 0
                try:
                    cur.execute("SELECT mtime FROM vectors WHERE path=? AND model=?", (p, self._image_model_name))
                    row = cur.fetchone()
                except Exception:
                    row = None
                if not row or int(row[0]) != mt:
                    to_proc.append((p, mt))
        finally:
            con.close()
        if not to_proc:
            return 0
        done = 0
        # 배치 처리
        i = 0
        n = len(to_proc)
        while i < n:
            chunk = to_proc[i:i+int(max(1, batch_size))]
            paths = [p for p, _ in chunk]
            mtimes = [mt for _, mt in chunk]
            vecs: List[Optional[List[float]]] = [None] * len(paths)
            # 1) ST 모델 경로
            if has_st and Image is not None:
                try:
                    ims = []
                    for p in paths:
                        try:
                            with Image.open(p) as im:
                                ims.append(im.convert("RGB"))
                        except Exception:
                            ims.append(None)
                    ims2 = [im for im in ims if im is not None]
                    if ims2:
                        arr = self._image_model.encode(images=ims2, batch_size=int(max(1, batch_size)), convert_to_numpy=True, normalize_embeddings=True)
                        # encode한 순서 ↔ paths 매핑
                        k = 0
                        for j, im in enumerate(ims):
                            if im is None:
                                continue
                            try:
                                vecs[j] = list(arr[k].tolist())
                            except Exception:
                                vecs[j] = None
                            k += 1
                except Exception:
                    pass
            # 2) HF-CLIP 경로
            if not any(v is not None for v in vecs) and has_hf and Image is not None:
                try:
                    import torch as _TH  # type: ignore
                except Exception:
                    _TH = None  # type: ignore
                try:
                    ims = []
                    for p in paths:
                        try:
                            with Image.open(p) as im:
                                ims.append(im.convert("RGB"))
                        except Exception:
                            ims.append(None)
                    ims2 = [im for im in ims if im is not None]
                    if ims2:
                        inputs = self._hf_clip_processor(images=ims2, return_tensors="pt")
                        if _TH is not None:
                            with _TH.no_grad():
                                feat = self._hf_clip_model.get_image_features(**inputs)
                                feat = feat / feat.norm(dim=-1, keepdim=True)
                            arr = feat.cpu().numpy()
                        else:
                            feat = self._hf_clip_model.get_image_features(**inputs)
                            try:
                                import numpy as _NP  # type: ignore
                                arr = feat.detach().cpu().numpy() if hasattr(feat, "detach") else _NP.array(feat)
                                nrm = (_NP.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
                                arr = arr / nrm
                            except Exception:
                                arr = None
                        if arr is not None:
                            k = 0
                            for j, im in enumerate(ims):
                                if im is None:
                                    continue
                                try:
                                    vecs[j] = list(arr[k].tolist())
                                except Exception:
                                    vecs[j] = None
                                k += 1
                except Exception:
                    pass
            # 저장
            con = sqlite3.connect(self._db_path)
            try:
                cur = con.cursor()
                for j, p in enumerate(paths):
                    v = vecs[j]
                    if not v:
                        continue
                    try:
                        cur.execute(
                            "INSERT INTO vectors(path, mtime, model, dim, vec) VALUES(?,?,?,?,?) "
                            "ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, model=excluded.model, dim=excluded.dim, vec=excluded.vec",
                            (p, int(mtimes[j]), self._image_model_name, len(v), self._write_vec(v)),
                        )
                        done += 1
                    except Exception:
                        pass
                con.commit()
            finally:
                con.close()
            i += int(max(1, batch_size))
        return done

