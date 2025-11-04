from dataclasses import dataclass
from typing import List, Dict, Callable

from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtCore import Qt


@dataclass(frozen=True)
class Command:
    id: str
    label: str
    desc: str
    category: str
    handler_name: str
    default_keys: List[str]
    lock_key: bool = False  # true면 사용자 재매핑 불가(F1 등)


# 명령 레지스트리: 필요 시 추가/수정
COMMANDS: List[Command] = [
    Command("open_file", "파일 열기", "파일 열기 대화상자", "파일", "open_file", ["Ctrl+O"]),
    Command("open_folder", "폴더 열기", "폴더 선택 후 스캔", "파일", "open_folder", ["Ctrl+Shift+O"]),
    Command("reload_current", "현재 파일 다시 읽기", "디스크에서 다시 로드", "파일", "reload_current_image", ["F5", "Ctrl+R"]),
    Command("toggle_fullscreen", "전체화면 토글", "전체화면 전환", "보기", "toggle_fullscreen", ["F11", "Alt+Enter"]),
    Command("toggle_ui_chrome", "UI 크롬 토글", "툴바/필름스트립/평점바 표시 전환", "보기", "toggle_ui_chrome", ["Tab"]),
    Command("toggle_info_overlay", "정보 오버레이 토글", "파일/해상도 정보 오버레이 표시 전환", "보기", "toggle_info_overlay", ["Ctrl+H"]),
    Command("handle_escape", "나가기/전체화면 종료", "Esc 동작", "시스템", "handle_escape", ["Escape"], lock_key=True),
    Command("delete_current_image", "파일 삭제", "현재 파일을 휴지통으로", "파일", "delete_current_image", ["Delete"]),

    # 최근/세션 관련
    Command("reopen_last_closed", "마지막 닫은 이미지 다시 열기", "직전에 닫은 이미지를 다시 엽니다", "파일", "reopen_last_closed_image", ["Ctrl+Shift+T"]),
    Command("clear_recent", "최근 목록 비우기", "최근 파일/폴더 목록을 지웁니다", "파일", "clear_recent", ["Ctrl+Alt+Shift+R"]),

    # 캐시 비우기
    Command("clear_caches", "캐시 비우기", "이미지/스케일/썸네일 캐시 삭제", "파일", "clear_caches", ["Ctrl+Shift+Delete", "Ctrl+Shift+C"]),

    Command("show_prev_image", "이전 이미지", "이전 파일로 이동", "탐색", "show_prev_image", ["Left", "PageUp"]),
    # Space는 애니메이션 재생/일시정지용으로 사용하므로 기본 매핑에서 제외
    Command("show_next_image", "다음 이미지", "다음 파일로 이동", "탐색", "show_next_image", ["Right", "PageDown"]),
    Command("show_first_image", "첫 이미지", "첫 파일로 이동", "탐색", "show_first_image", ["Home"]),
    Command("show_last_image", "마지막 이미지", "마지막 파일로 이동", "탐색", "show_last_image", ["End"]),

    Command("fit_to_window", "화면 맞춤", "화면에 맞게 보기", "보기", "fit_to_window", ["Ctrl+0"]),
    Command("fit_to_width", "가로 맞춤", "너비에 맞게 보기", "보기", "fit_to_width", ["Ctrl+2"]),
    Command("fit_to_height", "세로 맞춤", "높이에 맞게 보기", "보기", "fit_to_height", ["Ctrl+3"]),
    Command("reset_to_100", "실제 크기(100%)", "배율 100%", "보기", "reset_to_100", ["Ctrl+1"]),
    # Ctrl+Plus/Ctrl+Equal 모두 허용 + 단일 키(=/-)도 허용
    Command("zoom_in", "확대", "점진 확대", "보기", "zoom_in", ["=", "Ctrl++", "Ctrl+="]),
    Command("zoom_out", "축소", "점진 축소", "보기", "zoom_out", ["-", "Ctrl+-", "Ctrl+Minus", "Ctrl+Subtract"]),

    # 원본 업그레이드/프리뷰 제어
    Command("upgrade_fullres_now", "원본 업그레이드 강제", "지연 없이 즉시 원본으로 교체", "보기", "upgrade_fullres_now", ["U"]),
    Command("revert_to_preview", "프리뷰로 되돌리기", "현재 화면 배율에 맞춘 프리뷰로 재전환", "보기", "revert_to_preview", ["Ctrl+U"]),

    # 회전/반전 기본 단축키 업데이트
    Command("rotate_ccw_90", "왼쪽 90° 회전", "반시계 방향 회전", "편집", "rotate_ccw_90", ["["]),
    Command("rotate_cw_90", "오른쪽 90° 회전", "시계 방향 회전", "편집", "rotate_cw_90", ["]"]),
    Command("rotate_cycle", "순환 회전(시계 방향)", "0→90→180→270 순환", "편집", "rotate_cycle", ["Shift+R"]),
    Command("rotate_180", "180° 회전", "180도 회전", "편집", "rotate_180", ["Ctrl+Alt+R"]),
    Command("flip_horizontal", "좌우 뒤집기", "수평 반전", "편집", "flip_horizontal", ["H"]),
    Command("flip_vertical", "상하 뒤집기", "수직 반전", "편집", "flip_vertical", ["V"]),
    Command("reset_transform", "변환 초기화", "회전/반전 상태 초기화", "편집", "reset_transform", ["Backspace"]),

    # 편집 히스토리 제거됨

    # 애니메이션 토글: Space 고정(다른 명령에 할당 금지)
    Command("toggle_animation", "애니메이션 토글", "재생/일시정지 전환", "보기", "anim_toggle_play", ["Space"], lock_key=True),

    # 애니메이션 프레임 탐색/점프
    Command("anim_prev_frame", "이전 프레임", "애니메이션 이전 프레임", "보기", "anim_prev_frame", [","]),
    Command("anim_next_frame", "다음 프레임", "애니메이션 다음 프레임", "보기", "anim_next_frame", ["."]),
    Command("anim_jump_back_10", "10프레임 뒤로", "애니메이션 10프레임 이전", "보기", "anim_jump_back_10", ["Shift+,"]),
    Command("anim_jump_forward_10", "10프레임 앞으로", "애니메이션 10프레임 다음", "보기", "anim_jump_forward_10", ["Shift+."]),

    # 색상 보기 A/B 토글 (원본 vs sRGB/타깃 변환)
    Command("toggle_color_ab", "원본/sRGB 보기 토글", "색상 보기 A/B 전환", "보기", "toggle_color_ab", ["C"]),

    # 정보/패널
    Command("toggle_info_panel", "정보 패널 토글", "우측 정보 패널 표시/숨김", "보기", "toggle_info_panel", ["I", "Ctrl+I"]),

    # 도움말(F1)은 고정
    Command("help_shortcuts", "단축키 도움말", "현재 단축키 표시", "도움말", "open_shortcuts_help", ["F1"], lock_key=True),
    # Natural language search
    Command("open_natural_search", "자연어 검색", "텍스트로 이미지 검색", "검색", "open_natural_search_dialog", ["Ctrl+K"]),
    Command("rerun_natural_search", "자연어 재검색(최근 질의)", "최근 질의로 재검색", "검색", "rerun_last_natural_search", ["Ctrl+Shift+K"]),

    # Similar image search
    Command("open_similar_search", "유사 검색 열기", "현재 사진과 유사한 사진 찾기", "검색", "open_similar_search_dialog", ["Ctrl+Shift+S", "Alt+S"]),

    # AI 분석
    Command("open_ai_analysis", "AI 분석 실행", "현재 사진 분석", "AI", "open_ai_analysis_dialog", ["Ctrl+Shift+A"]),
    Command("chain_ai_analysis", "연쇄 AI 분석 토글", "현재→다음 반복 분석 시작/중지", "AI", "toggle_chain_ai_analysis", ["Ctrl+Alt+A"]),
    Command("toggle_ai_language", "AI 언어 전환", "분석 언어 한국어/영어 토글", "AI", "toggle_ai_language", ["Ctrl+Alt+L"]),

    # 프라이버시/위치 표시 토글
    Command("toggle_privacy_location", "위치 표시 토글", "정보 패널에서 주소/지도 표시 토글", "보기", "toggle_privacy_hide_location", ["Ctrl+L"]),

    # 최근 파일 빠른 실행 (Alt+1..9)
    Command("open_recent_1", "최근 파일 1", "최근 파일 1 열기", "파일", "_open_recent_1", ["Alt+1"]),
    Command("open_recent_2", "최근 파일 2", "최근 파일 2 열기", "파일", "_open_recent_2", ["Alt+2"]),
    Command("open_recent_3", "최근 파일 3", "최근 파일 3 열기", "파일", "_open_recent_3", ["Alt+3"]),
    Command("open_recent_4", "최근 파일 4", "최근 파일 4 열기", "파일", "_open_recent_4", ["Alt+4"]),
    Command("open_recent_5", "최근 파일 5", "최근 파일 5 열기", "파일", "_open_recent_5", ["Alt+5"]),
    Command("open_recent_6", "최근 파일 6", "최근 파일 6 열기", "파일", "_open_recent_6", ["Alt+6"]),
    Command("open_recent_7", "최근 파일 7", "최근 파일 7 열기", "파일", "_open_recent_7", ["Alt+7"]),
    Command("open_recent_8", "최근 파일 8", "최근 파일 8 열기", "파일", "_open_recent_8", ["Alt+8"]),
    Command("open_recent_9", "최근 파일 9", "최근 파일 9 열기", "파일", "_open_recent_9", ["Alt+9"]),

    # Logs
    Command("open_logs_folder", "로그 폴더 열기", "로그 파일 폴더를 엽니다", "도움말", "open_logs_folder", ["Ctrl+Alt+O"]),
]


def _load_custom_keymap(settings) -> Dict[str, List[str]]:
    keymap: Dict[str, List[str]] = {}
    try:
        for cmd in COMMANDS:
            # reset_to_100는 항상 무시(숫자 키는 평점에 사용)
            if cmd.id == "reset_to_100":
                continue
            raw = settings.value(f"keys/custom/{cmd.id}", "", str)
            if raw:
                parts = [p.strip() for p in raw.split(";") if p.strip()]
                if parts:
                    keymap[cmd.id] = parts
    except Exception:
        pass
    return keymap


def save_custom_keymap(settings, mapping: Dict[str, List[str]]) -> None:
    try:
        # reset_to_100는 항상 공백 저장
        settings.setValue("keys/custom/reset_to_100", "")
        for cmd in COMMANDS:
            if cmd.lock_key or cmd.id == "reset_to_100":
                continue
            keys = mapping.get(cmd.id, []) or []
            settings.setValue(f"keys/custom/{cmd.id}", ";".join(keys))
    except Exception:
        pass


def get_effective_keymap(settings) -> Dict[str, List[str]]:
    custom = _load_custom_keymap(settings)
    eff: Dict[str, List[str]] = {}
    for cmd in COMMANDS:
        # 고정키는 기본값 고정
        if cmd.lock_key:
            eff[cmd.id] = cmd.default_keys[:]
            continue
        # 사용자 지정과 기본값을 병합(사용자 지정 우선, 중복 제거)
        merged: List[str] = []
        for src in (custom.get(cmd.id, []) or []) + (cmd.default_keys or []):
            s = str(src).strip()
            if not s:
                continue
            # Space는 애니메이션 전용으로 예약, 다른 명령에서는 제외
            if s.lower() == "space" and cmd.id != "toggle_animation":
                continue
            if s not in merged:
                merged.append(s)
        eff[cmd.id] = merged
    return eff


def apply_shortcuts(viewer) -> None:
    # 기존 단축키 제거
    try:
        for sc in getattr(viewer, "_shortcuts", []) or []:
            try:
                sc.setParent(None)
            except Exception:
                pass
    except Exception:
        pass
    viewer._shortcuts = []

    eff = get_effective_keymap(getattr(viewer, "settings", None))

    for cmd in COMMANDS:
        handler: Callable | None = getattr(viewer, cmd.handler_name, None)
        if not callable(handler):
            continue
        for key in eff.get(cmd.id, []) or []:
            # Space는 애니메이션 토글 전용으로 허용하고, 다른 명령에는 사용 금지
            if key and key.strip().lower() == "space" and cmd.id != "toggle_animation":
                continue
            try:
                sc = QShortcut(QKeySequence(key), viewer)
                if cmd.id == "toggle_animation":
                    try:
                        sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
                    except Exception:
                        pass
                sc.activated.connect(handler)
                viewer._shortcuts.append(sc)
            except Exception:
                # 키 파싱 실패 등은 무시
                pass


