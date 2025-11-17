import sys
import os
from PyQt6.QtCore import QCoreApplication, Qt  # type: ignore[import]
from PyQt6.QtWidgets import QApplication  # type: ignore[import]
from src.ui.main_window import JusawiViewer
from src.utils.logging_setup import setup_logging, get_logger, shutdown_logging

def _mask_args(argv):
    out = []
    for a in argv:
        try:
            if a.startswith('-'):
                out.append(a)
            else:
                out.append(os.path.basename(os.path.abspath(os.path.expanduser(a))))
        except Exception:
            out.append(a)
    return out

if __name__ == "__main__":
    # 로깅 초기화
    try:
        lvl = os.getenv("JUSAWI_LOG_LEVEL", "INFO")
        setup_logging(level=lvl)
        log = get_logger("app")
        log.info("app_start | argv=%s", _mask_args(sys.argv[1:]))
    except Exception:
        log = None  # type: ignore

    # 로컬 모델을 사전 다운로드(models/ 폴더) — 환경변수 미사용
    try:
        repo_root = os.path.abspath(os.path.dirname(__file__))
        models_dir = os.path.join(repo_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        def _ensure_repo(repo_id: str, target_dir_name: str) -> None:
            try:
                tgt = os.path.join(models_dir, target_dir_name)
                if os.path.isdir(tgt) and os.listdir(tgt):
                    if log:
                        log.info("model_present | %s", target_dir_name)
                    return
                try:
                    from huggingface_hub import snapshot_download  # type: ignore
                except Exception as e:
                    if log:
                        log.warning("hf_missing | skip_download | err=%s", str(e))
                    return
                # 로컬 디렉터리에 직접 다운로드(심볼릭링크 비활성화: Windows 호환)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=tgt,
                    local_dir_use_symlinks=False,
                    allow_patterns=None,
                    ignore_patterns=None,
                )
                if log:
                    log.info("model_downloaded | %s -> %s", repo_id, target_dir_name)
            except Exception as e:
                if log:
                    log.warning("model_prep_fail | repo=%s | err=%s", repo_id, str(e))
        # 텍스트(멀티링구얼) / 이미지(CLIP) 모델
        _ensure_repo("sentence-transformers/clip-ViT-B-32-multilingual-v1", "clip-ViT-B-32-multilingual-v1")
        _ensure_repo("sentence-transformers/clip-ViT-B-32", "clip-ViT-B-32")
    except Exception as e:
        try:
            if log:
                log.warning("model_prep_outer_fail | err=%s", str(e))
        except Exception:
            pass

    # QtWebEngine 요구사항: QCoreApplication 생성 전 초기 설정
    try:
        # OpenGL 컨텍스트 공유(필수 경고 회피)
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        # QtWebEngine을 미리 임포트(설치되지 않은 환경은 무시)
        try:
            import PyQt6.QtWebEngineWidgets  # type: ignore
        except Exception:
            pass
        try:
            import PyQt6.QtWebEngineCore  # type: ignore
        except Exception:
            pass
    except Exception:
        pass

    app = QApplication(sys.argv)
    # 명령줄 인자: 파일 또는 폴더 경로 사전 파싱
    args = [a for a in sys.argv[1:] if a and not a.startswith('-')]
    skip_restore = bool(args)
    viewer = JusawiViewer(skip_session_restore=skip_restore)
    # 인자가 있는 경우에만 즉시 열기 시도
    try:
        opened = False  
        if args:
            for arg in args:
                path = os.path.abspath(os.path.expanduser(arg))
                if os.path.isfile(path):
                    # 파일 직접 열기
                    try:
                        viewer.load_image(path, source='open')
                        opened = True
                        try:
                            if log:
                                log.info("open_arg_file | file=%s | ok=%s", os.path.basename(path), True)
                        except Exception:
                            pass
                        break
                    except Exception as e:
                        try:
                            if log:
                                log.exception("open_arg_file_failed | file=%s | err=%s", os.path.basename(path), str(e))
                        except Exception:
                            pass
                elif os.path.isdir(path):
                    # 폴더 인자: 디렉터리 스캔 후 현재 인덱스 이미지 로드
                    try:
                        viewer.scan_directory(path)
                        if 0 <= viewer.current_image_index < len(viewer.image_files_in_dir):
                            viewer.load_image(viewer.image_files_in_dir[viewer.current_image_index], source='open')
                            opened = True
                            try:
                                if log:
                                    log.info("open_arg_dir | dir=%s | count=%d", os.path.basename(path), len(viewer.image_files_in_dir))
                            except Exception:
                                pass
                            break
                    except Exception as e:
                        try:
                            if log:
                                log.exception("open_arg_dir_failed | dir=%s | err=%s", os.path.basename(path), str(e))
                        except Exception:
                            pass
        # 인자가 있었으나 모두 실패한 경우, 마지막 세션 복원 시도
        if args and not opened:
            try:
                viewer.restore_last_session()
                try:
                    if log:
                        log.info("restore_last_session_after_args | ok=True")
                except Exception:
                    pass
            except Exception as e:
                try:
                    if log:
                        log.exception("restore_last_session_after_args_failed | err=%s", str(e))
                except Exception:
                    pass
    except Exception as e:
        try:
            if log:
                log.exception("startup_flow_failed | err=%s", str(e))
        except Exception:
            pass
    viewer.show()
    try:
        app.aboutToQuit.connect(viewer.save_last_session)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        rc = app.exec()
        try:
            if log:
                log.info("app_exit | code=%s", rc)
        except Exception:
            pass
        sys.exit(rc)
    finally:
        try:
            shutdown_logging()
        except Exception:
            pass