# iCAP Air Device Data Analyzer

## 介紹

iCAP Air Device Data Analyzer 是一個用於分析 iCAP Air 裝置資料的小工具

## 功能

目前有下列功能，未來會再擴充

- 分析 Single Device 所下載的，Excel ，並產生分析報告

## 環境

- Python 3.11.x

  **可支援 3.11.X 的版本，請不要安裝其他版本，未經測試，可能會無法執行本工具**

- Windows or Linux

## 如何使用

請按照以下步驟操作：

1. Clone 此儲存庫：
   ```bash
   git clone https://github.com/yourusername/icap-air-sensor-data-analyzer.git
   ```
2. 進入專案目錄：
   ```bash
   cd icap-air-sensor-data-analyzer
   ```
3. 建立虛擬環境並啟動

   以下命令執行完成後，會看到終端機的路徑前方會出現`venv` 字樣

   ```bash
   python -m venv venv

   # Linux
   source venv/bin/activate

   # Windows
   powershell -ExecutionPolicy ByPass -File .\venv\Scripts\activate.ps1

   ```

4. 安裝所需的套件：

   ```bash
   pip install -r requirements.txt
   ```

## 使用方式

1. 設定環境變數，請先修改 `.env.example` 為 `.env`

2. 執行主程式：

   ```bash

   python main.py -i "{下載的報表檔案完整路徑}" -o "{儲存的完整路徑，沒有斜線，檔名會自動產生}"

   # 範例

   # 不指定輸出路徑，預設會存在 output 目錄底下
   python .\main.py -i .\1_2025-01-01_air-data.csv

    # 指定輸出路徑，會存在 abc 底下
   python .\main.py -i .\1_2025-01-01_air-data.csv -o abc

   ```

3. 您可以使用 `python main.py -h` 查看指令說明

## 批次執行方式

本小工具另提供一個批次執行用的腳本，供一次分析大量的報告檔案使用，其功能為批次執行主程式，目錄路徑內可包含子資料夾

```bash

python execute.py "{下載的報表檔案的目錄路徑}"

# 範例

python execute.py "D:\Innodisk\iCAP\icap-air-sensor-data-analyzer\裡面很多報告檔"


```

## 問題與建議

如果有任何問題或建議，請聯絡 Innodisk IPA - Jacky
