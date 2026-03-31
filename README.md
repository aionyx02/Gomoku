# Gomoku (C++20 + FTXUI + Python AI)

終端機版本五子棋專案，支援：
- `PvP`（玩家對玩家）
- `PvE`（玩家對 AI）

UI 使用 [FTXUI](https://github.com/ArthurSonzogni/ftxui)，AI 走子由 Python 模型推論，核心棋盤邏輯在 C++。

## 功能特色

- 15x15 棋盤，黑子先手。
- 終端機互動式 UI（方向鍵移動、Enter/Space 落子）。
- `PvP` / `PvE` 模式切換。
- AI 模型推論（`gomoku_model.pt`）與 fallback 機制。
- 音效與背景音樂（`miniaudio`）。

## 環境需求

- CMake `>= 3.20`
- C++20 編譯器（GCC 10+ / Clang 12+ / MSVC 2019+）
- Python `>= 3.13`（CMake 會強制檢查）
- `pybind11`
- AI 推論套件：`numpy`、`torch`

## 快速開始

### 1. 建立 Python 環境並安裝依賴

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
pip install pybind11 -r requirements-ai-runtime.txt
```

macOS / Linux:
```bash
source .venv/bin/activate
pip install pybind11 -r requirements-ai-runtime.txt
```

### 2. 產生建置檔與編譯

```bash
cmake -S . -B build
cmake --build build
```

若你使用多組態生成器（例如 Visual Studio），可改用：

```bash
cmake --build build --config Release
```

### 3. 啟動遊戲

Windows（Visual Studio）：
```bash
build\Release\Gomoku_Project.exe
```

Windows（Ninja / MinGW）：
```bash
build\Gomoku_Project.exe
```

macOS / Linux：
```bash
./build/Gomoku_Project
```

## 操作方式

- `↑ ↓ ← →`：移動游標
- `Enter` / `Space`：落子
- 首頁選單：`Enter` 確認

## AI 模型說明

- 預設模型檔路徑：專案根目錄 `gomoku_model.pt`
- 推論腳本：`python/runModelAndReturnPoint.py`
- `PvE` 模式中若模型或 Python 環境不可用，會自動 fallback 到合法步（遊戲可持續進行）
- 失敗診斷會寫入 `ai_debug.log`

## 音效資源說明

請保留 `voice/` 目錄與以下檔案：

- `voice/backGround.mp3`
- `voice/click.mp3`
- `voice/placeStoneVoice.mp3`

## 專案結構

```text
Gomoku/
|-- CMakeLists.txt
|-- include/gomoku/
|   |-- engine.h
|   |-- game_session.h
|   |-- ai_player.h
|   |-- ui_controller.h
|   |-- voice.h
|   `-- webConnect.h
|-- src/
|   |-- app.cpp
|   |-- engine.cpp
|   |-- game_session.cpp
|   |-- ai_player.cpp
|   |-- ui_controller.cpp
|   |-- voice.cpp
|   |-- webConnect.cpp
|   `-- bindings.cpp
|-- python/
|   |-- runModelAndReturnPoint.py
|   |-- gomoku_net.py
|   `-- ...
|-- voice/
|-- requirements-ai-runtime.txt
`-- gomoku_model.pt
```
