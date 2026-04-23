# Gomoku (C++20 Terminal Edition)

> [中文版請見下方 ↓](#gomoku-c20-終端機版)

---

A terminal-based Gomoku (Five in a Row) game built with **C++20 + FTXUI**, featuring:

- Local two-player (`PvP`)
- Local player-vs-AI (`PvE`, built-in C++ reward-engine AI)
- LAN remote play (`Host` / `Join`)
- Save & load (board reconstructed from move history)
- Replay mode and per-turn timer (toggleable in Settings)

## Features

- 15×15 board
- AI move diagnostics display in PvE mode
- In-game shortcuts: `S` Settings · `U` Undo · `L` Save/Quit · `Q` Disconnect (remote)
- Remote handshake and snapshot sync
- Regression tests covering remote sync and save-failure paths

## System Requirements

| Tool | Minimum Version |
|------|----------------|
| CMake | 3.20+ |
| Compiler | GCC 10+ / Clang 12+ / MSVC 2019+ |
| Network | Required on first configure (downloads FTXUI via `FetchContent`) |

---

## Quick Start (Windows — from a clean machine)

Follow these steps in order. They assume a brand-new Windows 11 machine with nothing pre-installed.

### Step 1 — Install Git

1. Download the installer from <https://git-scm.com/download/win>
2. Run the installer, keep all defaults, and click **Next** until **Install**
3. Verify:
   ```powershell
   git --version
   # expected: git version 2.x.x
   ```

### Step 2 — Install CMake

1. Download the **Windows x64 Installer** (.msi) from <https://cmake.org/download/>
2. During installation, select **"Add CMake to the system PATH for all users"**
3. Verify:
   ```powershell
   cmake --version
   # expected: cmake version 3.x.x
   ```

### Step 3 — Install a C++ Compiler

**Option A — Visual Studio Build Tools (recommended, ~6 GB)**

1. Download from <https://visualstudio.microsoft.com/visual-cpp-build-tools/>
2. Run the installer, choose **"Desktop development with C++"**, and install
3. No extra PATH setup needed; CMake detects MSVC automatically

**Option B — MinGW-w64 (lighter, ~600 MB)**

1. Download **winlibs** release (UCRT, MSVCRT, or LLVM) from <https://winlibs.com/>
2. Extract to `C:\mingw64`
3. Add `C:\mingw64\bin` to your system PATH:
   - Search **"Edit the system environment variables"** → **Environment Variables**
   - Under **System variables**, select `Path` → **Edit** → **New** → paste `C:\mingw64\bin`
4. Verify:
   ```powershell
   g++ --version
   # expected: g++ (MinGW-w64 ...) 13.x.x
   ```

### Step 4 — Clone the Repository

```powershell
git clone https://github.com/<your-username>/oop_gomoku.git
cd oop_gomoku\Gomoku
```

> Replace `<your-username>` with the actual GitHub username.

### Step 5 — Configure (downloads FTXUI automatically)

```powershell
cmake -S . -B cmake-build-debug
```

Expected output (first run takes ~30 s depending on internet speed):

```
-- Fetching ftxui v6.1.9 ...
-- Configuring done
-- Build files have been written to: .../cmake-build-debug
```

> If you chose MinGW, append `-G "MinGW Makefiles"`:
> ```powershell
> cmake -S . -B cmake-build-debug -G "MinGW Makefiles"
> ```

### Step 6 — Build

```powershell
cmake --build cmake-build-debug --target Gomoku_Project
```

A successful build ends with a line like:

```
[100%] Linking CXX executable Gomoku_Project.exe
```

### Step 7 — Run

```powershell
.\cmake-build-debug\Gomoku_Project.exe
```

The terminal UI will launch. Use arrow keys to navigate.

> Use Windows Terminal or PowerShell with a monospace font. Do not run the game inside an IDE output console, because FTXUI needs a real interactive terminal; IDE consoles can make the screen repeat downward or appear misaligned. A terminal size of at least 80x30 is recommended.

### Step 8 — (Optional) Run Regression Tests

```powershell
cmake --build cmake-build-debug --target gomoku_regression_tests
ctest --test-dir cmake-build-debug --output-on-failure
```

Or run the test binary directly:

```powershell
.\cmake-build-debug\gomoku_regression_tests.exe
```

---

## Controls

### Main Menu

| Option | Description |
|--------|-------------|
| `Start Game (PvP)` | Local two-player |
| `Start Game (PvE)` | Player vs AI |
| `Host Remote Game` | Create a LAN room |
| `Join Remote Game` | Connect to a LAN room |
| `Setup` | Toggle undo / per-turn timer |
| `Load Game` | Load a save file |
| `Exit` | Quit |

### In-Game Shortcuts

| Key | Action |
|-----|--------|
| `↑ ↓ ← →` | Move cursor |
| `Enter` / `Space` | Place stone |
| `U` | Undo (must be enabled in Setup) |
| `S` | Open Settings |
| `L` | Save / Quit menu |
| `Esc` | Go back |
| `Q` | Disconnect (remote mode only) |

### Remote Mode

- **Host**: enter a port number, wait for opponent to connect
- **Join**: enter Host IP and port to connect

---

## Save Files

- Default directory: `saves/`
- Filename format: `save_YYYYMMDD_HHMMSS_mmm.gomoku`
- File contents:
  - `mode PVP|PVE`
  - `size <board_size>`
  - Move coordinates (`x y`) one per line

## Audio Assets

The program tries to load these files from `assets/audio/`:

- `backGround.mp3`
- `click.mp3`
- `placeStoneVoice.mp3`

## Project Structure

```
Gomoku/
├── CMakeLists.txt
├── README.md
├── docs/
│   ├── scoring.md
│   └── screenshots/
├── include/gomoku/
│   ├── ai/
│   ├── audio/
│   ├── core/
│   ├── net/
│   └── ui/
├── src/
│   ├── ai/
│   ├── app/
│   ├── audio/
│   ├── core/
│   ├── net/
│   └── ui/
├── assets/
│   └── audio/
└── tests/
    └── regression_tests.cpp
```

---
---

# Gomoku (C++20 終端機版)

以 **C++20 + FTXUI** 實作的終端機五子棋，支援：

- 本機雙人 (`PvP`)
- 本機人機 (`PvE`，內建 C++ reward-engine AI)
- 區網遠端連線對戰 (`Host` / `Join`)
- 對局存檔與讀檔（以走子歷史回放重建棋盤）
- 回放模式與回合計時（可在設定頁開關）

## 功能總覽

- 15×15 棋盤對局
- 人機模式 AI 走子診斷狀態顯示
- 遊戲內快捷鍵：`S` 設定 · `U` 悔棋 · `L` 存檔/離開 · `Q` 斷線（遠端）
- 遠端連線握手與快照同步
- 回歸測試涵蓋遠端同步與存檔失敗路徑

## 系統需求

| 工具 | 最低版本 |
|------|---------|
| CMake | 3.20+ |
| 編譯器 | GCC 10+ / Clang 12+ / MSVC 2019+ |
| 網路 | 首次 configure 需要（自動下載 FTXUI） |

---

## 快速開始

有三種方式，擇一即可：

| 方式 | 適合對象 |
|------|---------|
| [最快捷：winget 一鍵安裝](#最快捷--winget-一鍵安裝-windows-11-內建) | Windows 11 全新電腦，想最少步驟搞定 |
| [方案 A：使用 Visual Studio](#方案-a--使用-visual-studio) | 已有 Visual Studio，或想要完整 IDE 體驗 |
| [方案 B：命令列（CMake + 任意編譯器）](#方案-b--命令列-cmake--任意編譯器) | 只想裝最少工具，或使用 VSCode / 其他編輯器 |

---

### 最快捷 — winget 一鍵安裝（Windows 11 內建）

Windows 11 已內建 `winget`，不需要自己去網站下載安裝程式。

以**系統管理員**身分開啟 PowerShell，依序貼上：

```powershell
# 1. 安裝 Git
winget install Git.Git --source winget

# 2. 安裝 Visual Studio 2022 Community（含 C++ 工作負載，約 6–10 GB）
winget install Microsoft.VisualStudio.2022.Community --source winget --override "--add Microsoft.VisualStudio.Workload.NativeDesktop --quiet --wait"
```

安裝完成後確認：

```powershell
winget list Microsoft.VisualStudio.2022.Community
```

> 安裝完成後需**重新開啟 PowerShell** 讓 PATH 生效。

接著複製並開啟專案：

```powershell
# 3. 複製專案
git clone https://github.com/<your-username>/oop_gomoku.git
```

然後按照下方「[開啟與啟動](#開啟與啟動)」的步驟，用 Visual Studio 開啟資料夾並按 F5 即可。

---

### 方案 A — 使用 Visual Studio

#### 前置條件

若尚未安裝 Visual Studio，請依上方 winget 方式安裝，或：

1. 前往 <https://visualstudio.microsoft.com/> 下載 Visual Studio Installer
2. 執行 Installer，勾選 **「使用 C++ 的桌面開發」** 工作負載
3. 點擊「安裝」，等待完成（約 6–10 GB）

#### 複製專案

用 PowerShell 執行：

```powershell
git clone https://github.com/<your-username>/oop_gomoku.git
```

> 若尚未安裝 Git，請先至 <https://git-scm.com/download/win> 下載並安裝（保持預設值即可）。

#### 開啟與啟動 {#開啟與啟動}

1. 開啟 Visual Studio
2. 選擇 **「開啟本機資料夾」**（Open a local folder）
3. 選取 `oop_gomoku\Gomoku\` 資料夾（就是含有 `CMakeLists.txt` 的那層）
4. VS 會自動偵測 CMake 並執行 Configure，底部輸出視窗會顯示 FTXUI 下載進度
   - 首次需要網路，約等待 30 秒
   - 若底部沒有動靜，請點選單 **「專案」→「刪除快取並重新設定」** 手動觸發
5. Configure 完成後，點擊上方工具列綠色箭頭**旁邊的下拉選單**，選擇 `Gomoku_Project.exe`
   - 若下拉清單是空的，代表 Configure 還沒跑完，請等候底部輸出出現 `Configuring done`
6. 按 **F5** 或綠色箭頭 ▶ — VS 會自動 Build 再執行

程式啟動後即可在終端機視窗中操作。

> **提示**：之後每次開啟只需重複步驟 1–3，再按 F5 即可，Configure 不會重複執行。
>
> **VS 2019 使用者**：若遇到 `could not find git for clone of ftxui` 錯誤，請完全關閉 VS 後重新開啟，讓它吃到安裝 Git 後的新 PATH。

---

### 方案 B — 命令列（Windows — 全新電腦從零設定）

以下步驟適用於一台尚未安裝任何開發工具的 Windows 11 電腦，請依序執行。

#### 步驟一 — 安裝 Git

1. 前往 <https://git-scm.com/download/win> 下載安裝程式
2. 執行安裝程式，保持預設值一路點 **Next** 直到 **Install**
3. 驗證安裝：
   ```powershell
   git --version
   # 預期輸出：git version 2.x.x
   ```

#### 步驟二 — 安裝 CMake

1. 前往 <https://cmake.org/download/> 下載 **Windows x64 Installer**（.msi）
2. 安裝過程中選擇 **「Add CMake to the system PATH for all users」**
3. 驗證安裝：
   ```powershell
   cmake --version
   # 預期輸出：cmake version 3.x.x
   ```

#### 步驟三 — 安裝 C++ 編譯器

**方案 A — Visual Studio Build Tools（推薦，約 6 GB）**

1. 前往 <https://visualstudio.microsoft.com/visual-cpp-build-tools/> 下載
2. 執行安裝程式，勾選 **「使用 C++ 的桌面開發」**，然後安裝
3. CMake 會自動偵測 MSVC，無需額外設定 PATH

**方案 B — MinGW-w64（輕量版，約 600 MB）**

1. 前往 <https://winlibs.com/> 下載最新 release（選 UCRT 或 MSVCRT 版本）
2. 解壓縮至 `C:\mingw64`
3. 將 `C:\mingw64\bin` 加入系統 PATH：
   - 搜尋「編輯系統環境變數」→「環境變數」
   - 在「系統變數」中找到 `Path` → 「編輯」→「新增」→ 貼上 `C:\mingw64\bin`
4. 驗證安裝：
   ```powershell
   g++ --version
   # 預期輸出：g++ (MinGW-w64 ...) 13.x.x
   ```

#### 步驟四 — 複製專案

```powershell
git clone https://github.com/<your-username>/oop_gomoku.git
cd oop_gomoku\Gomoku
```

> 請將 `<your-username>` 替換為實際的 GitHub 使用者名稱。

#### 步驟五 — Configure（自動下載 FTXUI）

```powershell
cmake -S . -B cmake-build-debug
```

首次執行會從 GitHub 下載 FTXUI（約 30 秒，視網路速度而定），成功後輸出類似：

```
-- Fetching ftxui v6.1.9 ...
-- Configuring done
-- Build files have been written to: .../cmake-build-debug
```

> 若使用 MinGW，請改用：
> ```powershell
> cmake -S . -B cmake-build-debug -G "MinGW Makefiles"
> ```

#### 步驟六 — 編譯

```powershell
cmake --build cmake-build-debug --target Gomoku_Project
```

成功時最後一行會出現：

```
[100%] Linking CXX executable Gomoku_Project.exe
```

#### 步驟七 — 執行

```powershell
.\cmake-build-debug\Gomoku_Project.exe
```

終端機 UI 啟動後，使用方向鍵操作。

> 請用 Windows Terminal 或 PowerShell 搭配等寬字型執行。不要在 IDE 的 Output/Run console 內直接跑，因為 FTXUI 需要真正的互動式終端機；IDE console 可能會讓畫面一直往下重複或看起來跑版。建議終端機大小至少 80x30。

#### 步驟八 — （選用）執行回歸測試

```powershell
cmake --build cmake-build-debug --target gomoku_regression_tests
ctest --test-dir cmake-build-debug --output-on-failure
```

或直接執行測試 binary：

```powershell
.\cmake-build-debug\gomoku_regression_tests.exe
```

---

## 操作方式

### 主選單

| 選項 | 說明 |
|------|------|
| `Start Game (PvP)` | 本機雙人對戰 |
| `Start Game (PvE)` | 玩家對 AI |
| `Host Remote Game` | 建立區網房間 |
| `Join Remote Game` | 加入區網房間 |
| `Setup` | 開關悔棋 / 回合計時 |
| `Load Game` | 載入存檔 |
| `Exit` | 離開遊戲 |

### 對局快捷鍵

| 按鍵 | 動作 |
|------|------|
| `↑ ↓ ← →` | 移動游標 |
| `Enter` / `Space` | 落子 |
| `U` | 悔棋（需在 Setup 開啟） |
| `S` | 開啟設定 |
| `L` | 存檔/離開選單 |
| `Esc` | 返回上一層 |
| `Q` | 斷線（僅遠端模式） |

### 遠端模式

- **Host**：輸入埠號後啟動，等待對手連入
- **Join**：輸入對方 IP 與埠號後連線

---

## 存檔與資料

- 預設存檔目錄：`saves/`
- 檔名格式：`save_YYYYMMDD_HHMMSS_mmm.gomoku`
- 檔案內容：
  - `mode PVP|PVE`
  - `size <board_size>`
  - 每一步座標（`x y`），一步一行

## 音效資源

音效檔位於 `assets/audio/`，Build 完成後會自動複製到執行檔同目錄：

| 檔案 | 用途 |
|------|------|
| `backGround.mp3` | 背景音樂循環 |
| `click.mp3` | 選單按鈕點擊音 |
| `placeStoneVoice.mp3` | 落子音效 |
| `selected.mp3` | UI 選取回饋音 |
| `victory.mp3` | 勝利音效 |
| `defeat.mp3` | 失敗音效 |
| `do.wav` | 提示音 |
| `movechess.wav` | 移動棋子音 |
| `dead.ogg` | 特殊事件音效 |

> 音效檔缺少時程式不會崩潰，僅靜音播放。

## 專案結構

```
Gomoku/
├── CMakeLists.txt
├── README.md
├── doc/                          # 詳細設計文件
│   ├── design.md
│   ├── notes.md
│   ├── scoring.md
│   ├── packaging.md
│   ├── detail-guide.md
│   ├── rulecheck.md
│   ├── todolist.md
│   └── c-log.md
├── docs/
│   └── scoring.md
├── include/gomoku/
│   ├── miniaudio.h               # 音效後端（header-only）
│   ├── ai/
│   ├── audio/
│   ├── core/
│   ├── net/
│   └── ui/
├── src/
│   ├── ai/
│   ├── app/
│   ├── audio/
│   ├── core/
│   ├── net/
│   └── ui/
│       ├── ui_controller.cpp
│       ├── ui_controller_state.cpp
│       └── ui_controller_views.cpp
├── assets/
│   └── audio/                    # 9 個音效檔
├── saves/                        # 自動產生的存檔
└── tests/
    ├── regression_tests.cpp      # 遠端同步 & 存檔測試
    └── rulecheck_tests.cpp       # 規則驗證測試
```
