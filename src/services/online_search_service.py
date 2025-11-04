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
    """OpenAI 임베딩으로 로컬 이미지 컬렉션을 색인/검색.

    - 임베딩 모델: env EMBED_MODEL (기본: text-embedding-3-small)
    - DB: ~/.jusawi/online_embed.sqlite3
    - 벡터 저장 형식: CSV float 문자열(bytes)
    """

    def __init__(self, model: str | None = None, api_key: str | None = None, tag_weight: int | None = None, verify_model: str | None = None):
        base_dir = os.path.join(os.path.expanduser("~"), ".jusawi")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            pass
        self._db_path = os.path.join(base_dir, "online_embed.sqlite3")
        self._model = (model or "text-embedding-3-small").strip() or "text-embedding-3-small"
        self._api_key = api_key or None
        try:
            self._tag_weight = int(tag_weight if tag_weight is not None else 2)
        except Exception:
            self._tag_weight = 2
        try:
            self._verify_model = str(verify_model or "gpt-4o-mini")
        except Exception:
            self._verify_model = "gpt-4o-mini"
        self._ensure_db()

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
            # AI 태그/주제 저장 테이블(+캡션 컬럼 포함)
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
            # 마이그레이션: 기존 DB에 누락 컬럼 추가(이미 있으면 무시)
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

    def _get_mtime(self, path: str) -> int:
        try:
            return int(os.path.getmtime(path))
        except Exception:
            return 0

    def _embed_text_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY 없음")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(f"openai SDK 로드 실패: {e}")
        # 임베딩 호출 타임아웃 상향(네트워크 지연 대응)
        client = OpenAI(api_key=self._api_key, timeout=30.0)
        # OpenAI Embeddings API: 최대 입력 길이에 주의(안전하게 batch로 처리)
        resp = client.embeddings.create(model=self._model, input=texts)
        out: List[List[float]] = []
        for item in resp.data:
            out.append(list(item.embedding))
        return out

    def ensure_index(self, image_paths: List[str], progress_cb=None, batch_size: int | None = None, is_cancelled=None) -> int:
        """경로 목록에 대해 누락/구버전 임베딩을 생성해 저장. 반환: 새로 생성한 개수."""
        pending: List[Tuple[str, str, int]] = []  # (path, doc, mtime)
        con = sqlite3.connect(self._db_path)
        try:
            cur = con.cursor()
            for p in image_paths:
                mt = self._get_mtime(p)
                try:
                    cur.execute("SELECT mtime, model FROM vectors WHERE path=?", (p,))
                    row = cur.fetchone()
                except Exception:
                    row = None
                need = True
                if row is not None:
                    try:
                        prev_mtime = int(row[0])
                        prev_model = str(row[1] or "")
                        if prev_mtime == mt and prev_model == self._model:
                            need = False
                        else:
                            need = True
                    except Exception:
                        need = True
                if need:
                    doc = self._build_doc_with_tags(p)
                    pending.append((p, doc, mt))
        finally:
            con.close()

        created = 0
        if not pending:
            return created
        # 배치 단위로 임베딩
        B = int(batch_size if batch_size is not None else int(os.getenv("EMBED_BATCH", "64") or 64))
        i = 0
        while i < len(pending):
            if callable(is_cancelled) and is_cancelled():
                break
            chunk = pending[i : i + B]
            texts = [d for (_, d, __) in chunk]
            if progress_cb:
                try:
                    progress_cb(min(90, int(100 * (i / max(1, len(pending))))), f"임베딩 {i+1}-{min(i+B, len(pending))}/{len(pending)}")
                except Exception:
                    pass
            vecs = self._embed_text_batch(texts)
            con = sqlite3.connect(self._db_path)
            try:
                cur = con.cursor()
                for (path, doc, mt), vec in zip(chunk, vecs):
                    try:
                        cur.execute(
                            "INSERT INTO vectors(path, mtime, model, dim, vec, meta) VALUES(?,?,?,?,?,?) "
                            "ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, model=excluded.model, dim=excluded.dim, vec=excluded.vec, meta=excluded.meta",
                            (path, mt, self._model, len(vec), _write_vec(vec), json.dumps({"doc": doc}, ensure_ascii=False)),
                        )
                        created += 1
                    except Exception:
                        pass
                con.commit()
            finally:
                con.close()
            i += B
        return created

    def _build_doc_with_tags(self, path: str) -> str:
        base = _build_doc_for_image(path)
        # tags 테이블에서 불러와 결합
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
                        w = max(1, min(5, int(getattr(self, "_tag_weight", 2))))
                        parts.append("tags: " + ",".join([t] * w))
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

    def upsert_tags_subjects(self, path: str, tags: List[str] | None, subjects: List[str] | None,
                              short_caption: Optional[str] = None, long_caption: Optional[str] = None) -> None:
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
            qmarks = ",".join(["?"] * len(image_paths)) if image_paths else ""
            if not qmarks:
                return []
            cur.execute(f"SELECT path, dim, vec FROM vectors WHERE path IN ({qmarks}) AND model=?", (*image_paths, self._model))
            for row in cur.fetchall():
                path = str(row[0])
                # dim = int(row[1])  # 미사용
                vec = _read_vec(row[2])
                if vec:
                    out.append((path, vec))
        except Exception:
            pass
        finally:
            con.close()
        return out

    def search(self,
               image_paths: List[str],
               query_text: str,
               top_k: int = 50,
               verify_top_n: int = 20,
               verify_mode: str = "normal",
               progress_cb=None,
               use_embedding: bool | None = None,
               strict_only_opt: bool | None = None,
               verify_max_candidates: int | None = None,
               verify_workers_opt: int | None = None,
               blend_alpha_opt: float | None = None,
               embed_batch_size: int | None = None,
               is_cancelled=None) -> List[Tuple[str, float]]:
        if not query_text.strip():
            return []
        if progress_cb:
            try:
                progress_cb(5, "색인 확인")
            except Exception:
                pass
        # 임베딩 사용 여부: 기본 사용(True). 매개변수로만 제어
        no_embed = (False if use_embedding is None else (not bool(use_embedding)))
        strict_only = True if strict_only_opt is None else bool(strict_only_opt)
        # 검증 모드 기본값: strict
        vm = (verify_mode or "").strip().lower()
        if vm not in ("loose", "normal", "strict"):
            vm = "strict"
        if no_embed:
            if progress_cb:
                try:
                    progress_cb(20, "임베딩 생략: 후보 수집")
                except Exception:
                    pass
            try:
                verify_cap = int(verify_max_candidates if verify_max_candidates is not None else 200)
            except Exception:
                verify_cap = 200
            verify_cap = max(1, verify_cap)
            cands = list(image_paths)[:verify_cap]
            scored: List[Tuple[str, float]] = [(p, 0.0) for p in cands]
            top_k = len(scored)
            verify_top_n = len(scored)
        else:
            # 캐시를 사용하지 않고, 매 검색마다 전체 파일 임베딩을 새로 생성하여 코사인 유사도 계산
            if callable(is_cancelled) and is_cancelled():
                return []
            if progress_cb:
                try:
                    progress_cb(20, "질의 임베딩")
                except Exception:
                    pass
            qvec = self._embed_text_batch([query_text])[0]
            if callable(is_cancelled) and is_cancelled():
                return []
            # 이미지 문서 생성 → 배치 임베딩(신규 생성, DB 미저장)
            try:
                B = int(embed_batch_size if embed_batch_size is not None else 128)
            except Exception:
                B = 128
            docs: List[Tuple[str, str]] = [(p, self._build_doc_with_tags(p)) for p in image_paths]
            embed_map_vec: Dict[str, List[float]] = {}
            i = 0
            total = len(docs)
            while i < total:
                if callable(is_cancelled) and is_cancelled():
                    return []
                chunk = docs[i : i + B]
                texts = [d for (_, d) in chunk]
                if progress_cb:
                    try:
                        progress_cb(30 + int(25 * (i / max(1, total))), f"이미지 임베딩 {i+1}-{min(i+B, total)}/{total}")
                    except Exception:
                        pass
                # 임베딩 재시도 로직(부분 실패 시 배치 분할 → 단건 폴백)
                def _embed_with_retry(in_texts: List[str]) -> List[List[float]]:
                    attempts = 0
                    cur_texts = in_texts
                    cur_batch = len(cur_texts)
                    while attempts < 3:
                        attempts += 1
                        try:
                            out_vecs = self._embed_text_batch(cur_texts)
                            if len(out_vecs) == len(cur_texts):
                                return out_vecs
                        except Exception:
                            pass
                        # 배치 반으로 줄여 재시도
                        if cur_batch > 1:
                            cur_batch = max(1, cur_batch // 2)
                            cur_texts = in_texts[:cur_batch]
                        else:
                            break
                    # 단건 폴백
                    out: List[List[float]] = []
                    for t in in_texts:
                        try:
                            v = self._embed_text_batch([t])[0]
                        except Exception:
                            v = []
                        out.append(v)
                    return out

                vecs = _embed_with_retry(texts)
                # vecs 길이 보정(부족분 0-벡터)
                if len(vecs) < len(texts):
                    diff = len(texts) - len(vecs)
                    vecs.extend([[] for _ in range(diff)])
                for (path, _), vec in zip(chunk, vecs):
                    if vec:
                        embed_map_vec[path] = vec
                i += B
            if progress_cb:
                try:
                    progress_cb(60, "코사인 유사도 계산")
                except Exception:
                    pass
            # 코사인 점수 산출
            scored = []
            for p in image_paths:
                v = embed_map_vec.get(p)
                c = _cosine(qvec, v) if v else 0.0
                scored.append((p, float(max(0.0, c))))
            # 전수 재검증(전체 후보)
            verify_top_n = len(scored)
            scored.sort(key=lambda x: x[1], reverse=True)

        # 2차 재검증을 생략하고, 임베딩 이후 바로 3차 바이너리 필터만 수행
        if scored:
            if progress_cb:
                try:
                    progress_cb(75, "최종 필터링")
                except Exception:
                    pass
            try:
                from .verifier_service import VerifierService  # type: ignore
                verifier = VerifierService(api_key=self._api_key, model=getattr(self, "_verify_model", "gpt-5-nano"))
                # 병렬 워커 수 설정
                try:
                    workers = int(verify_workers_opt if verify_workers_opt is not None else 64)
                except Exception:
                    workers = 64
                workers = max(1, min(64, workers))
                try:
                    import concurrent.futures as _fut
                except Exception:
                    _fut = None  # type: ignore

                # 전체 후보에 대해 이미지당 단일 요청으로 바이너리 판정 수행
                n = len(scored)
                final: List[Tuple[str, float]] = []
                tasks: List[Tuple[int, str]] = [(i, scored[i][0]) for i in range(n)]
                if _fut is not None and workers > 1:
                    with _fut.ThreadPoolExecutor(max_workers=workers) as ex:
                        fut_to_idx = {
                            ex.submit(verifier.verify_binary, path, query_text): idx for idx, path in tasks
                        }
                        done = 0
                        for fut in _fut.as_completed(fut_to_idx):
                            if callable(is_cancelled) and is_cancelled():
                                break
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
                                    progress_cb(base + int(span * (done / max(1, n))), "최종 필터링")
                                except Exception:
                                    pass
                else:
                    for i, path in tasks:
                        if callable(is_cancelled) and is_cancelled():
                            break
                        r = verifier.verify_binary(path, query_text)
                        if bool(r.get("match", r.get("ok", False))):
                            conf = float(r.get("confidence", 0.0))
                            final.append((path, conf))
                        if progress_cb:
                            try:
                                base = 75
                                span = 20
                                progress_cb(base + int(span * ((i + 1) / max(1, n))), "최종 필터링")
                            except Exception:
                                pass
                final.sort(key=lambda x: x[1], reverse=True)
                return final
            except Exception as e:
                try:
                    _log.warning("verify_bin_only_fail | err=%s", str(e))
                except Exception:
                    pass
        return scored


