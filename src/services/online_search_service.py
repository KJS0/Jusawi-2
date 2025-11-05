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
    """OpenAI 임베딩 기반 자연어 → 이미지 검색 인덱스."""

    # 기본값: 호환성이 높은 멀티링구얼 CLIP (SentenceTransformers 포맷)
    # 텍스트/이미지 인코더를 분리: 텍스트는 멀티링구얼, 이미지는 CLIP ViT-B/32
    _DEFAULT_TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    _DEFAULT_IMAGE_MODEL = "clip-ViT-B-32"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        tag_weight: int | None = None,
        verify_model: str | None = None,
        embed_batch: int | None = None,
        # 호환 파라미터(무시): 이전 M-CLIP 경로 인자들
        model_path: str | None = None,
        text_model_path: str | None = None,
        image_model_path: str | None = None,
        image_batch: int | None = None,
        text_batch: int | None = None,
    ) -> None:

        base_dir = os.path.join(os.path.expanduser("~"), ".jusawi")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            pass

        self._db_path = os.path.join(base_dir, "online_embed.sqlite3")
        self._model = (model or "text-embedding-3-small").strip() or "text-embedding-3-small"
        self._api_key = api_key or None
        try:
            self._tag_weight = float(tag_weight if tag_weight is not None else 2.0)
        except Exception:
            self._tag_weight = 2.0
        if self._tag_weight < 0:
            self._tag_weight = 0.0
        self._verify_model = str(verify_model or "gpt-5-nano")
        try:
            self._text_batch_size = max(1, int(embed_batch if embed_batch is not None else 64))
        except Exception:
            self._text_batch_size = 64
        # 모델 시그니처(캐시 구분)
        self._model_signature = f"openai::{self._model}|tagw={int(self._tag_weight)}"

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
    # 저장소 관리
    # ------------------------------------------------------------------
    def _resolve_text_model_paths(self) -> List[str]:
        paths: List[str] = []
        # 명시 경로 우선
        if self._requested_text_model_path and os.path.isdir(self._requested_text_model_path):
            paths.append(self._requested_text_model_path)
        # 이전 단일 경로 호환
        if self._requested_model_path and os.path.isdir(self._requested_model_path):
            paths.append(self._requested_model_path)
        # 환경변수
        env_txt = os.getenv("JUSAWI_CLIP_TEXT_MODEL")
        if env_txt and os.path.isdir(env_txt):
            paths.append(env_txt)
        # 로컬 후보
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        for name in [
            "clip-ViT-B-32-multilingual-v1",
            "M-CLIP-XLMR-L-ViT-B-32",
            "M-CLIP-XLM-Roberta-Large-Vit-B-32",
        ]:
            p = os.path.join(repo_root, "models", name)
            if os.path.isdir(p):
                paths.append(p)
        # 허브 이름(우선순위)
        paths.extend([
            self._DEFAULT_TEXT_MODEL,
            "M-CLIP/XLM-Roberta-Large-Vit-B-32",
        ])
        return paths

    def _resolve_image_model_paths(self) -> List[str]:
        paths: List[str] = []
        if self._requested_image_model_path and os.path.isdir(self._requested_image_model_path):
            paths.append(self._requested_image_model_path)
        if self._requested_model_path and os.path.isdir(self._requested_model_path):
            paths.append(self._requested_model_path)
        env_img = os.getenv("JUSAWI_CLIP_IMAGE_MODEL")
        if env_img and os.path.isdir(env_img):
            paths.append(env_img)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        for name in [
            "clip-ViT-B-32",
            "clip-ViT-B-32-laion2B-s34B-b79K",  # 일부 배포 변형
        ]:
            p = os.path.join(repo_root, "models", name)
            if os.path.isdir(p):
                paths.append(p)
        paths.extend([
            self._DEFAULT_IMAGE_MODEL,
            "sentence-transformers/clip-ViT-B-32",
        ])
        return paths

    def _ensure_models(self) -> None:
        if self._text_model is not None and self._image_model is not None:
            return
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers 패키지를 찾을 수 없습니다.")

        last_err: Exception | None = None
        # 텍스트 모델 로드
        if self._text_model is None:
            for src in self._resolve_text_model_paths():
                try:
                    self._text_model = SentenceTransformer(src)  # type: ignore[arg-type]
                    self._text_model_source = src
                    break
                except Exception as e:
                    last_err = e
                    continue
        # 이미지 모델 로드
        if self._image_model is None:
            for src in self._resolve_image_model_paths():
                try:
                    self._image_model = SentenceTransformer(src)  # type: ignore[arg-type]
                    self._image_model_source = src
                    break
                except Exception as e:
                    last_err = e
                    continue

        if self._text_model is None or self._image_model is None:
            raise RuntimeError(f"모델 로드 실패: {last_err}")

        # 시그니처 구성(캐시 무효화 기준)
        tname = self._text_model_source if self._text_model_source else self._DEFAULT_TEXT_MODEL
        iname = self._image_model_source if self._image_model_source else self._DEFAULT_IMAGE_MODEL
        def _base(n: str) -> str:
            return os.path.basename(n.rstrip(os.sep)) if os.path.isdir(n) else n
        self._model_signature = f"mclip::{_base(iname)}|{_base(tname)}"

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
    # 임베딩/색인/검색
    # ------------------------------------------------------------------
    def _get_mtime(self, path: str) -> int:
        try:
            return int(os.path.getmtime(path))
        except Exception:
            return 0

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY 없음")
        # proxies 환경(HTTP(S)_PROXY/ALL_PROXY) 자동 감지 → httpx 클라이언트로 주입
        try:
            from openai import OpenAI  # type: ignore
            import httpx  # type: ignore
        except Exception as e:
            raise RuntimeError(f"openai/httpx 로드 실패: {e}")

        # httpx 0.28+: proxies 인자 제거됨 → proxy 또는 mounts 사용
        http_client = None
        try:
            hp = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
            sp = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
            ap = os.getenv("ALL_PROXY") or os.getenv("all_proxy")
            if ap:
                http_client = httpx.Client(proxy=ap, timeout=30.0)
            elif hp or sp:
                mounts = {}
                if hp:
                    mounts["http://"] = httpx.HTTPTransport(proxy=hp)
                if sp:
                    mounts["https://"] = httpx.HTTPTransport(proxy=sp)
                if mounts:
                    http_client = httpx.Client(mounts=mounts, timeout=30.0)
        except Exception:
            http_client = None

        client = OpenAI(api_key=self._api_key, http_client=http_client) if http_client is not None else OpenAI(api_key=self._api_key, timeout=30.0)
        resp = client.embeddings.create(model=self._model, input=texts)
        out: List[List[float]] = []
        for item in resp.data:
            out.append(list(item.embedding))
        return out

    # 이미지 임베딩/블렌딩은 OpenAI 텍스트 방식에선 사용하지 않음

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
            batch = int(batch_size if batch_size is not None else self._text_batch_size)
        except Exception:
            batch = self._text_batch_size

        total = len(pending)
        i = 0
        while i < total:
            if callable(is_cancelled) and is_cancelled():
                break
            chunk = pending[i : i + batch]
            docs = [item[1] for item in chunk]
            if progress_cb:
                try:
                    progress_cb(min(90, int(10 + 60 * (i / max(1, total)))), f"문서 임베딩 {i + 1}-{min(i + batch, total)}/{total}")
                except Exception:
                    pass

            text_vecs = self._embed_texts(docs)

            con = sqlite3.connect(self._db_path)
            try:
                cur = con.cursor()
                for idx, (path, doc, mt) in enumerate(chunk):
                    tvec = text_vecs[idx] if idx < len(text_vecs) else None
                    final_vec = tvec or []
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

        # 2차: GPT 비전 바이너리 재검증(전체 후보)
        try:
            from .verifier_service import VerifierService  # type: ignore
        except Exception:
            VerifierService = None  # type: ignore

        if VerifierService is None or not self._api_key:
            if top_k is not None and top_k > 0:
                scored = scored[:min(top_k, len(scored))]
            if progress_cb:
                try:
                    progress_cb(100, "완료")
                except Exception:
                    pass
            return scored

        if progress_cb:
            try:
                progress_cb(75, "최종 필터링")
            except Exception:
                pass

        try:
            import concurrent.futures as _fut
        except Exception:
            _fut = None  # type: ignore
        workers = 32
        final: List[Tuple[str, float]] = []
        tasks: List[Tuple[int, str]] = [(i, scored[i][0]) for i in range(len(scored))]
        verifier = VerifierService(api_key=self._api_key, model=self._verify_model)
        if _fut is not None and workers > 1:
            with _fut.ThreadPoolExecutor(max_workers=workers) as ex:
                fut_to_idx = {ex.submit(verifier.verify_binary, path, query_text): idx for idx, path in tasks}
                done = 0
                for fut in _fut.as_completed(fut_to_idx):
                    idx = fut_to_idx[fut]
                    path = scored[idx][0]
                    try:
                        r = fut.result()
                        if bool(r.get("match", r.get("ok", False))):
                            conf = float(r.get("confidence", 0.0))
                            final.append((path, conf))
                    except Exception:
                        pass
                    done += 1
                    if progress_cb:
                        try:
                            base = 75
                            span = 20
                            progress_cb(base + int(span * (done / max(1, len(scored)))), "최종 필터링")
                        except Exception:
                            pass
        else:
            for i, path in tasks:
                r = verifier.verify_binary(path, query_text)
                if bool(r.get("match", r.get("ok", False))):
                    conf = float(r.get("confidence", 0.0))
                    final.append((path, conf))
                if progress_cb:
                    try:
                        base = 75
                        span = 20
                        progress_cb(base + int(span * ((i + 1) / max(1, len(scored)))), "최종 필터링")
                    except Exception:
                        pass
        final.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None and top_k > 0:
            final = final[:min(top_k, len(final))]
        if progress_cb:
            try:
                progress_cb(100, "완료")
            except Exception:
                pass
        return final


