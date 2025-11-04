from __future__ import annotations

import os
import sqlite3
import json
import math
from typing import List, Tuple, Optional, Dict, Any

from ..utils.logging_setup import get_logger

_log = get_logger("svc.OnlineSearch")

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_ST = False

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore


def _read_vec(b: bytes) -> List[float]:
    try:
        s = b.decode("utf-8")
        return [float(x) for x in s.split(",") if x]
    except Exception:
        return []


def _write_vec(vec: List[float]) -> bytes:
    return (",".join(f"{x:.7f}" for x in vec)).encode("utf-8")


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
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _build_doc_for_image(path: str) -> str:
    """이미지의 파일명/폴더/가벼운 EXIF 요약을 이용해 텍스트 문서를 구성."""
    try:
        import os as _os
        base = _os.path.basename(path)
        parent = _os.path.basename(_os.path.dirname(path))
        parts = [f"file: {base}", f"folder: {parent}"]
    except Exception:
        parts = [f"file: {path}"]
    # EXIF 요약(가벼움)
    try:
        from .ai_analysis_service import _extract_exif_summary  # type: ignore
        ex = _extract_exif_summary(path) or {}
        if ex:
            ex_lines = []
            for k in [
                "make",
                "model",
                "lens",
                "aperture",
                "shutter",
                "iso",
                "focal_length_mm",
                "focal_length_35mm_eq_mm",
                "datetime_original",
            ]:
                v = ex.get(k)
                if v is not None and str(v).strip() != "":
                    ex_lines.append(f"{k}: {v}")
            if ex_lines:
                parts.append("exif: " + "; ".join(ex_lines))
    except Exception:
        pass
    return " | ".join(parts)


class OnlineEmbeddingIndex:
    """M-CLIP 기반 자연어 → 이미지 검색 인덱스."""

    _DEFAULT_MODEL_NAME = "M-CLIP/XLM-Roberta-Large-Vit-B-32"

    def __init__(
        self,
        model_path: str | None = None,
        tag_weight: int | None = None,
        image_batch: int | None = None,
        text_batch: int | None = None,
    ) -> None:
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers 패키지를 설치해야 자연어 검색을 사용할 수 있습니다.")

        base_dir = os.path.join(os.path.expanduser("~"), ".jusawi")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            pass

        self._db_path = os.path.join(base_dir, "online_embed.sqlite3")
        self._requested_model_path = model_path
        self._model: SentenceTransformer | None = None
        self._model_source: Optional[str] = None
        self._model_signature = f"mclip::{self._DEFAULT_MODEL_NAME}"

        try:
            self._tag_weight = float(tag_weight if tag_weight is not None else 2.0)
        except Exception:
            self._tag_weight = 2.0
        if self._tag_weight < 0:
            self._tag_weight = 0.0

        try:
            self._image_batch_size = max(1, int(image_batch if image_batch is not None else 24))
        except Exception:
            self._image_batch_size = 24

        try:
            self._text_batch_size = max(1, int(text_batch if text_batch is not None else 64))
        except Exception:
            self._text_batch_size = 64

        self._ensure_db()

    # ------------------------------------------------------------------
    # 모델 및 저장소 관리
    # ------------------------------------------------------------------
    def _resolve_model_path(self) -> str:
        # 1) 호출 시 지정된 경로
        if self._requested_model_path and os.path.isdir(self._requested_model_path):
            return self._requested_model_path

        # 2) 환경 변수
        env_path = os.getenv("JUSAWI_MCLIP_MODEL")
        if env_path and os.path.isdir(env_path):
            self._requested_model_path = env_path
            return env_path

        # 3) 리포지토리 내 기본 경로 후보
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidates = [
            os.path.join(repo_root, "models", "M-CLIP-XLMR-L-ViT-B-32"),
            os.path.join(repo_root, "models", "M-CLIP-XLM-Roberta-Large-Vit-B-32"),
            os.path.join(repo_root, "models", "M-CLIP"),
        ]
        for cand in candidates:
            if os.path.isdir(cand):
                self._requested_model_path = cand
                return cand

        # 4) 모델 이름(허브에서 자동 다운로드)
        return self._DEFAULT_MODEL_NAME

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers 패키지를 찾을 수 없습니다.")
        source = self._resolve_model_path()
        self._model = SentenceTransformer(source)  # type: ignore[arg-type]
        self._model_source = source
        if os.path.isdir(source):
            base = os.path.basename(source.rstrip(os.sep))
            self._model_signature = f"mclip::{base}"
        else:
            self._model_signature = f"mclip::{source}"

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
                    vec BLOB,
                    meta TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tags (
                    path TEXT PRIMARY KEY,
                    tags TEXT,
                    subjects TEXT,
                    short_caption TEXT,
                    long_caption TEXT
                )
                """
            )
            try:
                cur.execute("ALTER TABLE tags ADD COLUMN short_caption TEXT")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE tags ADD COLUMN long_caption TEXT")
            except Exception:
                pass
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vectors_model ON vectors(model)")
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------------
    # 임베딩 유틸리티
    # ------------------------------------------------------------------
    def _get_mtime(self, path: str) -> int:
        try:
            return int(os.path.getmtime(path))
        except Exception:
            return 0

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._ensure_model()
        try:
            batch = min(self._text_batch_size, max(1, len(texts)))
        except Exception:
            batch = max(1, len(texts))
        arr = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [[float(x) for x in vec.tolist()] for vec in arr]

    def _embed_images(self, paths: List[str]) -> List[Optional[List[float]]]:
        outputs: List[Optional[List[float]]] = [None] * len(paths)
        if Image is None:
            return outputs
        if not paths:
            return outputs

        self._ensure_model()
        loaded_images = []
        indices = []
        for idx, path in enumerate(paths):
            try:
                with Image.open(path) as im:  # type: ignore[arg-type]
                    img = im.convert("RGB").copy()
                loaded_images.append(img)
                indices.append(idx)
            except Exception:
                continue

        if not loaded_images:
            return outputs

        try:
            batch = min(self._image_batch_size, max(1, len(loaded_images)))
        except Exception:
            batch = max(1, len(loaded_images))

        try:
            arr = self._model.encode(  # type: ignore[union-attr]
                images=loaded_images,
                batch_size=batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for idx, vec in zip(indices, arr):
                outputs[idx] = [float(x) for x in vec.tolist()]
        finally:
            for img in loaded_images:
                try:
                    img.close()
                except Exception:
                    pass

        return outputs

    def _combine_vectors(self, base: Optional[List[float]], meta: Optional[List[float]]) -> List[float]:
        if not base:
            return []
        if not meta or len(base) != len(meta):
            return [float(x) for x in base]
        # 태그 가중치를 0~1 범위로 변환(기본 0.5)
        weight = max(0.0, min(1.0, 0.25 * self._tag_weight))
        if weight <= 0.0:
            return [float(x) for x in base]
        combined = [float((1.0 - weight) * a + weight * b) for a, b in zip(base, meta)]
        norm = math.sqrt(sum(x * x for x in combined))
        if norm > 0:
            return [float(x / norm) for x in combined]
        return [float(x) for x in base]

    # ------------------------------------------------------------------
    # 색인 & 검색
    # ------------------------------------------------------------------
    def ensure_index(
        self,
        image_paths: List[str],
        progress_cb=None,
        batch_size: int | None = None,
        is_cancelled=None,
    ) -> int:
        pending: List[Tuple[str, str, int]] = []
        con = sqlite3.connect(self._db_path)
        try:
            cur = con.cursor()
            for path in image_paths:
                mt = self._get_mtime(path)
                try:
                    cur.execute("SELECT mtime, model FROM vectors WHERE path=?", (path,))
                    row = cur.fetchone()
                except Exception:
                    row = None
                need_update = True
                if row is not None:
                    try:
                        prev_mtime = int(row[0])
                        prev_model = str(row[1] or "")
                        if prev_mtime == mt and prev_model == self._model_signature:
                            need_update = False
                    except Exception:
                        need_update = True
                if need_update:
                    pending.append((path, self._build_doc_with_tags(path), mt))
        finally:
            con.close()

        if not pending:
            return 0

        created = 0
        try:
            batch = int(batch_size if batch_size is not None else self._image_batch_size)
        except Exception:
            batch = self._image_batch_size

        total = len(pending)
        i = 0
        while i < total:
            if callable(is_cancelled) and is_cancelled():
                break
            chunk = pending[i : i + batch]
            paths = [item[0] for item in chunk]
            docs = [item[1] for item in chunk]
            if progress_cb:
                try:
                    progress_cb(
                        min(90, int(10 + 60 * (i / max(1, total)))),
                        f"이미지 임베딩 {i + 1}-{min(i + batch, total)}/{total}",
                    )
                except Exception:
                    pass

            img_vecs = self._embed_images(paths)
            text_vecs = self._embed_texts(docs) if self._tag_weight > 0 else [None] * len(chunk)

            con = sqlite3.connect(self._db_path)
            try:
                cur = con.cursor()
                for idx, (path, doc, mt) in enumerate(chunk):
                    ivec = img_vecs[idx] if idx < len(img_vecs) else None
                    if not ivec:
                        continue
                    tvec = text_vecs[idx] if idx < len(text_vecs) else None
                    final_vec = self._combine_vectors(ivec, tvec)
                    if not final_vec:
                        continue
                    try:
                        cur.execute(
                            "INSERT INTO vectors(path, mtime, model, dim, vec, meta) VALUES(?,?,?,?,?,?) "
                            "ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, model=excluded.model, dim=excluded.dim, vec=excluded.vec, meta=excluded.meta",
                            (
                                path,
                                mt,
                                self._model_signature,
                                len(final_vec),
                                _write_vec(final_vec),
                                json.dumps({"doc": doc}, ensure_ascii=False),
                            ),
                        )
                        created += 1
                    except Exception:
                        pass
                con.commit()
            finally:
                con.close()
            i += batch

        return created

    def _build_doc_with_tags(self, path: str) -> str:
        base = _build_doc_for_image(path)
        try:
            con = sqlite3.connect(self._db_path)
            try:
                cur = con.cursor()
                cur.execute("SELECT tags, subjects, short_caption, long_caption FROM tags WHERE path=?", (path,))
                row = cur.fetchone()
                if row is not None:
                    t = str(row[0] or "").strip()
                    s = str(row[1] or "").strip()
                    sc = str(row[2] or "").strip()
                    lc = str(row[3] or "").strip()
                    parts = [base]
                    if t:
                        parts.append("tags: " + t)
                    if s:
                        parts.append("subjects: " + s)
                    if sc:
                        parts.append("short_caption: " + sc)
                    if lc:
                        parts.append("long_caption: " + lc)
                    return " | ".join(parts)
            finally:
                con.close()
        except Exception:
            pass
        return base

    def upsert_tags_subjects(
        self,
        path: str,
        tags: List[str] | None,
        subjects: List[str] | None,
        short_caption: Optional[str] = None,
        long_caption: Optional[str] = None,
    ) -> None:
        if not path:
            return
        t = ",".join([str(x) for x in (tags or []) if str(x).strip()])
        s = ",".join([str(x) for x in (subjects or []) if str(x).strip()])
        con = sqlite3.connect(self._db_path)
        try:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO tags(path, tags, subjects, short_caption, long_caption) VALUES(?,?,?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET tags=excluded.tags, subjects=excluded.subjects, short_caption=excluded.short_caption, long_caption=excluded.long_caption",
                (path, t, s, (short_caption or ""), (long_caption or "")),
            )
            con.commit()
        finally:
            con.close()

    def _load_all_vectors(self, image_paths: List[str]) -> List[Tuple[str, List[float]]]:
        con = sqlite3.connect(self._db_path)
        out: List[Tuple[str, List[float]]] = []
        try:
            cur = con.cursor()
            if not image_paths:
                return []
            qmarks = ",".join(["?"] * len(image_paths))
            cur.execute(
                f"SELECT path, dim, vec FROM vectors WHERE path IN ({qmarks}) AND model=?",
                (*image_paths, self._model_signature),
            )
            for row in cur.fetchall():
                path = str(row[0])
                vec = _read_vec(row[2])
                if vec:
                    out.append((path, vec))
        except Exception:
            pass
        finally:
            con.close()
        return out

    def search(
        self,
        image_paths: List[str],
        query_text: str,
        top_k: int | None = None,
        progress_cb=None,
        is_cancelled=None,
    ) -> List[Tuple[str, float]]:
        if not query_text.strip():
            return []

        if progress_cb:
            try:
                progress_cb(5, "색인 확인")
            except Exception:
                pass

        self.ensure_index(image_paths, progress_cb=progress_cb, is_cancelled=is_cancelled)

        if callable(is_cancelled) and is_cancelled():
            return []

        if progress_cb:
            try:
                progress_cb(35, "질의 임베딩")
            except Exception:
                pass

        qvecs = self._embed_texts([query_text])
        if not qvecs:
            return []
        qvec = qvecs[0]

        if callable(is_cancelled) and is_cancelled():
            return []

        if progress_cb:
            try:
                progress_cb(55, "벡터 로드")
            except Exception:
                pass

        vec_map = {p: v for p, v in self._load_all_vectors(image_paths)}

        if progress_cb:
            try:
                progress_cb(70, "유사도 계산")
            except Exception:
                pass

        scored: List[Tuple[str, float]] = []
        for path in image_paths:
            vec = vec_map.get(path)
            score = _cosine(qvec, vec) if vec else 0.0
            scored.append((path, float(max(0.0, score))))

        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None and top_k > 0:
            scored = scored[:min(top_k, len(scored))]

        if progress_cb:
            try:
                progress_cb(100, "완료")
            except Exception:
                pass

        return scored


