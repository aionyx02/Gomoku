# 打包與發布說明

## 給玩家：如何下載並執行

1. 前往 GitHub Releases 頁面：  
   https://github.com/slimuCS/Gomoku/releases

2. 點選最新版本，下載 `Gomoku-vX.X.X-windows.zip`

3. 解壓縮後，資料夾結構如下：
   ```
   Gomoku-vX.X.X/
   ├── Gomoku_Project.exe
   └── assets/
       └── audio/
           ├── backGround.mp3
           ├── click.mp3
           └── placeStoneVoice.mp3
   ```

4. 直接執行 `Gomoku_Project.exe`

> **注意**：`assets/` 資料夾必須和 `.exe` 放在同一層，否則音效不會播放。  
> 不需要安裝任何額外套件（Visual C++ Redistributable 等）。

---

## 給開發者：如何發布新版本

### 前置條件
- Git 已設定好遠端 `origin`（指向 `https://github.com/slimuCS/Gomoku`）
- 已有 GitHub 帳號且對該 repo 有 push 權限

### 發布流程

1. **確認 main branch 是最新狀態且可正常 build**：
   ```bash
   git checkout main
   git pull
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   ```

2. **決定版本號**（遵循 [Semantic Versioning](https://semver.org/)）：
   - `v1.0.0` — 第一個正式版
   - `v1.1.0` — 新功能
   - `v1.0.1` — bug fix

3. **打 tag 並推送**：
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

4. **GitHub Actions 自動執行**（`.github/workflows/release.yml`）：
   - 在 Windows 環境重新 build
   - 打包 `Gomoku_Project.exe` + `assets/` 成 zip
   - 自動建立 GitHub Release 並上傳 zip

5. **確認結果**：  
   前往 `https://github.com/slimuCS/Gomoku/releases`，確認新 Release 出現且 zip 可正常下載。

### 如果想取消某個 tag

```bash
git tag -d v1.0.0              # 刪除本地 tag
git push origin :refs/tags/v1.0.0  # 刪除遠端 tag
```

---

## 技術細節

| 項目 | 說明 |
|------|------|
| 編譯器 | MSVC (Visual Studio 2022) |
| C++ 標準 | C++20 |
| UI 函式庫 | FTXUI v6.1.9（靜態連結） |
| 音效函式庫 | miniaudio（header-only，靜態） |
| Runtime | 靜態連結 `/MT`，不需要安裝 VC Redistributable |
| 目標平台 | Windows 10/11 x64 |
