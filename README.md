# Jusawi (주사위) - 주저하는 사진가를 위하여
Jusawi는 사진 뷰어 프로그램입니다.

## 기능
1. 이미지 파일 열기 : JPEG, PNG, BMP, GIF, TIFF, WebP로 가장 대중적인 사진 포맷을 지원합니다.
2. 디렉토리 내 다른 이미지 열람 : 디렉토리 내 다른 사진 파일을 열람할 수 있습니다.
3. 키보드 단축키 지원 : 다양한 단축키를 통해 보다 더 편리한 프로그램 사용이 가능합니다.
4. 전체화면 지원 : 전체화면 지원을 통해 보다 더 자세한 사진 열람이 가능합니다.
5. 보기 모드 : 화면 맞춤, 가로 맞춤, 세로 맞춤, 실제 크기(100%) 맞춤을 지원합니다.
6. 간단 회전/반전 : 좌/우 90° 회전 및 좌우, 상하 반전을 제공합니다.
7. 설정 : 다양한 설정을 통해 보다 더 나에게 맞는 프로그램 사용이 가능합니다.
8. 최근 목록 및 세션 복원 : 최근에 오픈된 사진 파일의 리스트를 볼 수 있고, 프로그램이 재작동될 때 가장 최근 사진을 열람하는 기능을 지원합니다.
9. 애니메이션 재생 : GIF/WebP 재생 및 일시중지, 자동 재생, 루프 재생이 가능합니다.
10. AI 분석 : 사진 한 장을 눌러 캡션·설명·촬영 의도·핵심 태그를 한 번에 생성합니다.
11. 사진 정보 한눈에 보기 : 상단 `정보` 버튼을 통해, 사진의 핵심 정보를 한눈에 볼 수 있습니다. 위치 정보가 있으면 실제 주소가 자동으로 표시되고, 바로 아래 작은 지도를 눌러 서비스를 이용할 수도 있습니다.
12. 자연어 검색 : 인터넷 검색 엔진을 통해 검색하듯이 키워드를 통해 폴더 내 사진을 검색해 보여줍니다.
13. 유사 사진 검색 : 현재 사진과 비슷한 사진을 폴더에서 자동으로 찾아 보여줍니다.
14. 필름 스트립 : 하단 고정 바에 현재 폴더의 사진 썸네일이 가로로 표시되며, 누르면 즉시 열람할 수 있습니다.
15. 별점/플래그 바 : 필름 스트립 아래에 별 5개와 플래그 두 개(승낙/거절)를 한눈에 보고 클릭으로 바로 설정할 수 있습니다.

## 실행법법
1) Python 설치 (3.10~3.12, 권장 3.11)
- [python.org 설치 페이지](https://www.python.org/downloads/)에서 최신 버전을 설치하세요.
- 설치 중 "Add python.exe to PATH" 옵션을 꼭 체크하세요.

2) 프로젝트 받기
- ZIP 다운로드 후 압축 해제하거나, Git을 아신다면 `git clone`으로 받아도 됩니다.

3) PowerShell 열기 → 폴더로 이동
- 예: `cd C:\\Development\\Jusawi`

4) 가상환경 만들기 및 활성화 (권장)

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
```

5) 필수 프로그램 설치

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6) 실행하기

```powershell
python main.py
```

## AI/지도 기능 설정

- 첫 실행 후, 프로젝트 최상단에 `config.yaml` 파일이 자동으로 생성됩니다.
- 아래처럼 키를 넣으면 AI 분석과 주소 표시 기능이 활성화됩니다.

```yaml
ai:
  openai_api_key: "여기에 OpenAI API 키"

map:
  api_keys:
    kakao: "카카오 REST API 키"
    google: "구글 Maps Geocoding API 키"
```

- 키가 없어도 기본 사진 뷰어 기능은 정상 작동합니다.
- 키 발급 안내
  - OpenAI: [API 키 발급](https://platform.openai.com/api-keys)
  - Kakao: [Kakao Developers](https://developers.kakao.com/)
  - Google Maps: [Google Cloud Console](https://console.cloud.google.com/)

## 추가 팁

- 가상환경 종료: PowerShell에서 `deactivate`
- 다음에 다시 실행할 때는 3) 폴더 이동 → 4) `Activate.ps1` → 6) 실행 순서만 하시면 됩니다.
