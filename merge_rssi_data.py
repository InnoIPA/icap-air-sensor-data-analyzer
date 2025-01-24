from typing import Dict, Any, Optional
import pandas as pd
import json
from pathlib import Path
from dotenv import load_dotenv
import os
import argparse
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dataclasses import dataclass

VERSION = "1.0.0"
AUTHOR = "IPA - Jacky Sung"
BUILD_DATE = "2025/01/17"


@dataclass
class Config:
    time_col: str

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        return cls(
            time_col=os.getenv("TIME_COLUMN", "時間"),
        )


def setup_logger(name: str = "RSSI Data Merger") -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_path = log_dir / f"{timestamp}_application_merge_debug.log"

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


def process_rssi_data(json_data: Dict[str, Any], rssi_type: str) -> pd.DataFrame:
    """處理 RSSI JSON 資料，轉換為 DataFrame 格式

    Args:
        json_data: 包含 RSSI 資料的字典
        rssi_type: RSSI 類型 ('data' 或 'env')

    Returns:
        包含時間戳和轉換後 RSSI 值(dBm)的 DataFrame
    """
    rssi_data = [
        {
            "ts": pd.to_datetime(
                item["ts"], unit="ms", utc=True
            ),  # 轉換 Unix timestamp 為 datetime
            f"{rssi_type}_rssi (dBm)": float(item["value"]) * 255
            - 256,  # 轉換為 dBm 值
        }
        for item in json_data["data"][f"{rssi_type}RSSI"]
    ]
    return pd.DataFrame(rssi_data)


def process_dataframe(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """處理輸入的 DataFrame，確保時間欄位格式正確

    Args:
        df: 輸入的 DataFrame
        time_col: 時間欄位名稱

    Returns:
        處理後的 DataFrame
    """
    df = df.dropna(subset=[time_col])  # 移除時間為空的記錄
    df[time_col] = pd.to_datetime(df[time_col])  # 轉換時間格式
    target_tz = df[time_col].dt.tz
    df = df.sort_values(time_col)
    return df, target_tz


def is_valid_directory(path_str: str) -> bool:
    """驗證目錄路徑是否有效且可寫入

    Args:
        path_str: 目錄路徑字串

    Returns:
        路徑是否有效且可寫入
    """
    try:
        path = Path(path_str)
        if path.suffix:  # 確保是目錄而非檔案
            return False
        try:
            path.resolve()
        except (RuntimeError, ValueError):
            return False
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                return False
        return True
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description=f"RSSI Data Merger v{VERSION} By {AUTHOR} {BUILD_DATE}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s -i INPUT_FILE -d DATA_RSSI_FILE -e ENV_RSSI_FILE [-o OUTPUT_DIR]",
        epilog="""
使用範例:
%(prog)s -i data.csv -d data_rssi.json -e env_rssi.json                # 使用預設輸出目錄 output/
%(prog)s -i data.csv -d data_rssi.json -e env_rssi.json -o reports    # 指定輸出目錄

注意事項:
- 支援的輸入檔案格式: .xlsx 或 .csv
- 輸出檔名會自動根據輸入檔名產生，格式為: 輸入檔名_merged.csv
- 若未指定輸出目錄，將自動使用 output 目錄
""",
    )

    parser.add_argument(
        "-i", "--input", required=True, help="原始資料檔案路徑 (.xlsx 或 .csv)"
    )
    parser.add_argument(
        "-d", "--data-rssi", required=True, help="Data RSSI JSON 檔案路徑"
    )
    parser.add_argument(
        "-e", "--env-rssi", required=True, help="Environment RSSI JSON 檔案路徑"
    )
    parser.add_argument("-o", "--output", help="輸出目錄路徑，預設為 output/")

    args = parser.parse_args()
    input_path = Path(args.input)

    # 驗證輸入檔案格式
    if input_path.suffix.lower() not in [".xlsx", ".csv"]:
        parser.error("錯誤：輸入檔案必須是 .xlsx 或 .csv 格式")

    # 處理輸出目錄
    if args.output is None:
        output_dir = Path("output")
    else:
        if not is_valid_directory(args.output):
            parser.error(
                "錯誤：指定的輸出目錄路徑無效，請確認：\n"
                "1. 必須是目錄路徑，不能包含檔名\n"
                "2. 路徑格式正確\n"
                "3. 具有建立目錄的權限"
            )
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)
    args.output = str(output_dir)

    return args


def main() -> int:
    """主程式流程

    Returns:
        0 表示成功，1 表示失敗
    """
    try:
        # 初始化設定
        logger = setup_logger()
        logger.debug(
            "Application started, version: %s, build data: %s", VERSION, BUILD_DATE
        )

        args = parse_args()
        config = Config.from_env()

        input_path = Path(args.input)
        data_rssi_path = Path(args.data_rssi)
        env_rssi_path = Path(args.env_rssi)
        output_dir = Path(args.output)

        # 讀取原始資料
        logger.info("讀取原始資料: %s", input_path)
        df = (
            pd.read_excel(input_path)
            if input_path.suffix.lower() == ".xlsx"
            else pd.read_csv(input_path)
        )

        if config.time_col not in df.columns:
            raise ValueError(f"找不到時間欄位: {config.time_col}")

        # 處理原始資料時間格式
        df, target_tz = process_dataframe(df, config.time_col)

        # 讀取並處理 dataRSSI
        logger.info("讀取 dataRSSI 資料: %s", data_rssi_path)
        with open(data_rssi_path) as f:
            data_json = json.load(f)
        data_rssi_df = process_rssi_data(data_json, "data")
        # 以 Excel 內的時區進行轉換
        data_rssi_df["ts"] = data_rssi_df["ts"].dt.tz_convert(target_tz)

        # 讀取並處理 envRSSI
        logger.info("讀取 envRSSI 資料: %s", env_rssi_path)
        with open(env_rssi_path) as f:
            env_json = json.load(f)
        env_rssi_df = process_rssi_data(env_json, "env")
        # 以 Excel 內的時區進行轉換
        env_rssi_df["ts"] = data_rssi_df["ts"].dt.tz_convert(target_tz)

        # 合併所有資料
        logger.info("合併資料...")
        # 使用 outer join 確保保留所有時間點的資料
        merged_df = df.merge(
            data_rssi_df.rename(columns={"ts": config.time_col}),
            on=config.time_col,
            how="outer",
        )
        merged_df = merged_df.merge(
            env_rssi_df.rename(columns={"ts": config.time_col}),
            on=config.time_col,
            how="outer",
        )

        # 確保最終結果按時間排序
        merged_df = merged_df.sort_values(config.time_col)

        # 儲存結果
        output_path = output_dir / f"{input_path.stem}_merged.csv"
        merged_df.to_csv(output_path, index=False)
        logger.info("合併完成，結果已儲存至: %s", output_path)

        logger.debug("Analysis completed successfully")
        return 0

    except Exception as e:
        logger.error("Error occurred: %s", str(e), exc_info=True)
        print(f"錯誤: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
