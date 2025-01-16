import pandas as pd
from datetime import datetime, time
import json
from pathlib import Path
from dotenv import load_dotenv
import os
import argparse
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

VERSION = "1.0.0"
AUTHOR = "IPA - Jacky Sung"
BUILD_DATE = "2025/01/07"


@dataclass
class Config:
    threshold: int
    interval: int
    time_col: str
    start_time: time
    end_time: time

    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls(
            threshold=int(os.getenv("OFFLINE_THRESHOLD_SECONDS", 120)),
            interval=int(os.getenv("EXPECTED_INTERVAL_SECONDS", 60)),
            time_col=os.getenv("TIME_COLUMN", "時間"),
            start_time=time.fromisoformat(os.getenv("START_TIME", "00:00:00")),
            end_time=time.fromisoformat(os.getenv("END_TIME", "23:59:59")),
        )


# Logger 設定
def setup_logger(name: str = "iCAP-Device Data Analyzer") -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_path = log_dir / f"{timestamp}_application_debug.log"

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 檔案處理器 - 記錄詳細的技術信息 (DEBUG 級別)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # 控制台處理器 - 只顯示用戶關心的信息 (INFO 級別)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


class DeviceAnalyzer:

    def __init__(self):
        self.logger = setup_logger()
        self.config = Config.from_env()
        self.data = None
        self.stats = None
        self.output_path = None

    def get_output_path(self):
        return self.output_path

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # 處理空值
        self.logger.debug("Processing data frame...")
        df = df.dropna(subset=[self.config.time_col])

        # 轉換時間欄位
        self.logger.debug("Converting time column to datetime...")
        df[self.config.time_col] = pd.to_datetime(df[self.config.time_col])

        # 過濾時間範圍
        self.logger.debug(
            "Filter data range: %s - %s", self.config.start_time, self.config.end_time
        )
        df["time_of_day"] = df[self.config.time_col].dt.time
        df = df[
            (df["time_of_day"] >= self.config.start_time)
            & (df["time_of_day"] <= self.config.end_time)
        ]

        processed_df = df.sort_values(self.config.time_col).drop("time_of_day", axis=1)
        self.logger.debug("Data process done, keep %d data.", len(processed_df))
        return processed_df

    def read_file(self, filepath: Path) -> pd.DataFrame:
        """
        讀取檔案並處理資料
        """
        if not filepath.exists():
            raise FileNotFoundError(f"找不到檔案: {filepath}")

        self.logger.debug(
            "Loading data: %s, file size: %d bytes",
            filepath,
            filepath.stat().st_size,
        )

        self.logger.info("讀取檔案: %s", filepath)

        df = (
            pd.read_excel(filepath)
            if filepath.suffix.lower() == ".xlsx"
            else pd.read_csv(filepath)
        )

        if self.config.time_col not in df.columns:
            raise ValueError(
                f"The specified time column is not found: {self.config.time_col}"
            )

        df = self._process_dataframe(df)
        self.data = df

        self.logger.debug(
            "Data process done: %d rows x %d columns", len(df), len(df.columns)
        )

        self.logger.info("成功讀取 %d 筆記錄", len(df))
        return df

    def analyze(self) -> Dict:
        """
        分析設備連線狀態

        計算內容：
        1. 基本統計（觀測天數、時間範圍）
        2. 記錄完整性（預期記錄數、實際記錄數、遺失率）
        3. 離線統計（離線時間、在線率、離線分布）
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("沒有資料可供分析")

        self.logger.info("開始分析資料...")

        # 計算基本統計
        # - time_diffs：計算相鄰記錄的時間差（秒）
        # - unique_dates：計算總共有幾天的資料
        time_diffs = self.data[self.config.time_col].diff().dt.total_seconds()
        unique_dates = self.data[self.config.time_col].dt.date.nunique()

        # 計算每日記錄數
        # - daily_seconds：每天應該觀測的總秒數
        # - expected_daily_records：每天應有的記錄數 = 總秒數/間隔 + 1
        # - total_expected_records：總預期記錄數 = 每天應有記錄數 × 天數
        daily_seconds = (
            (self.config.end_time.hour - self.config.start_time.hour) * 3600
            + (self.config.end_time.minute - self.config.start_time.minute) * 60
            + (self.config.end_time.second - self.config.start_time.second)
        )
        expected_daily_records = daily_seconds // self.config.interval + 1
        total_expected_records = expected_daily_records * unique_dates

        # 分析離線期間
        # - offline_periods：所有離線期間的列表，每個期間包含開始時間、結束時間、持續時間
        # - total_offline_time：所有離線時間的總和（秒）
        offline_periods = self._analyze_offline_periods(time_diffs)
        total_offline_time = sum(period["duration"] for period in offline_periods)

        # 統計離線分布
        ranges = self._categorize_offline_periods(offline_periods)

        # 計算遺失記錄數：預期總筆數 - 實際筆數
        missed_records = total_expected_records - len(self.data)

        # 計算遺失記錄率：遺失筆數 / 預期總筆數* 100%
        missed_records_rate = missed_records / total_expected_records * 100

        # 初始化 JSON 結果
        result = {
            "period": {
                "start": self.data[self.config.time_col].min(),
                "end": self.data[self.config.time_col].max(),
                "days": unique_dates,
                "hours": daily_seconds * unique_dates / 3600,
            },
            "records": {
                "expected": total_expected_records,
                "actual": len(self.data),
                "missed": missed_records,
                "per_day": expected_daily_records,
                "missed_records_rate": missed_records_rate,
            },
        }

        # 只在資料不完整時進行離線分析
        if missed_records_rate > 0:
            offline_periods = self._analyze_offline_periods(time_diffs)
            ranges = self._categorize_offline_periods(offline_periods)

            result["offline"] = {
                "count": len(offline_periods),
                "minutes": sum(period["duration"] for period in offline_periods) / 60,
                "ranges": ranges,
                "longest": sorted(
                    offline_periods, key=lambda x: x["duration"], reverse=True
                )[:5],
            }
        else:
            # 資料完整時，離線統計為 0
            result["offline"] = {
                "count": 0,
                "minutes": 0,
                "ranges": {"1-2分鐘": 0, "2-5分鐘": 0, "5分鐘以上": 0},
                "longest": [],
            }

        self.stats = result

    def _analyze_offline_periods(self, time_diffs: pd.Series) -> List[Dict]:
        """分析離線期間"""
        offline_periods = []

        # 1. 處理資料間的離線時間
        for i in range(1, len(self.data)):
            diff = time_diffs.iloc[i]
            if diff > self.config.threshold:
                offline_periods.append(
                    {
                        "start": self.data[self.config.time_col].iloc[i - 1],
                        "end": self.data[self.config.time_col].iloc[i],
                        "duration": diff,
                        "missed": round(diff / self.config.interval) - 1,
                    }
                )

        # 2. 依照日期分組處理每天的資料
        daily_data = self.data.groupby(self.data[self.config.time_col].dt.date)

        for date, group in daily_data:
            day_start = pd.Timestamp.combine(date, self.config.start_time).tz_localize(
                None
            )
            day_end = pd.Timestamp.combine(date, self.config.end_time).tz_localize(None)

            first_record = group[self.config.time_col].min().tz_localize(None)
            last_record = group[self.config.time_col].max().tz_localize(None)

            # 檢查當天第一筆資料之前的時間
            if first_record > day_start:
                diff = (first_record - day_start).total_seconds()
                if diff > self.config.threshold:  # 加入閾值判斷
                    offline_periods.append(
                        {
                            "start": day_start,
                            "end": first_record,
                            "duration": diff,
                            "missed": round(diff / self.config.interval) - 1,
                        }
                    )

            # 檢查當天最後一筆資料之後的時間
            if last_record < day_end:
                diff = (day_end - last_record).total_seconds()
                if diff > self.config.threshold:  # 加入閾值判斷
                    offline_periods.append(
                        {
                            "start": last_record,
                            "end": day_end,
                            "duration": diff,
                            "missed": round(diff / self.config.interval) - 1,
                        }
                    )

        return offline_periods

    # 離線時長分組
    def _categorize_offline_periods(self, periods: List[Dict]) -> Dict[str, int]:
        ranges = {"1-2分鐘": 0, "2-5分鐘": 0, "5分鐘以上": 0}
        for period in periods:
            duration_minutes = period["duration"] / 60
            if duration_minutes <= 2:
                ranges["1-2分鐘"] += 1
            elif duration_minutes <= 5:
                ranges["2-5分鐘"] += 1
            else:
                ranges["5分鐘以上"] += 1
        return ranges

    # 儲存報告
    def save_report(self, output_path: Path) -> None:
        if self.stats is None:
            raise ValueError("No analysis data available")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path

        daily_hours = (
            (self.config.end_time.hour - self.config.start_time.hour) * 3600
            + (self.config.end_time.minute - self.config.start_time.minute) * 60
            + (self.config.end_time.second - self.config.start_time.second)
        ) / 3600

        report = self._generate_report_content(daily_hours)

        # 儲存 Markdown 報告
        output_path.write_text(report, encoding="utf-8")
        self.logger.info("報告已儲存至: %s", output_path)

        # 儲存 JSON 資料
        self._save_json_report(output_path)

    def _generate_report_content(self, daily_hours: float) -> str:
        """生成報告內容"""
        s = self.stats
        report = f"""# 設備資料記錄分析報告
生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 觀測資訊
- 資料開始時間: {s['period']['start'].strftime('%Y-%m-%d %H:%M:%S')}
- 資料結束時間: {s['period']['end'].strftime('%Y-%m-%d %H:%M:%S')}
- 觀測天數: {s['period']['days']} 天
- 每日預期時段: {self.config.start_time} - {self.config.end_time} ({daily_hours:.1f}小時)
- 預期間隔: {self.config.interval} 秒
- 離線判定閾值: {self.config.threshold} 秒

## 2. 記錄完整性
- 每日應有記錄: {s['records']['per_day']} 筆
- 總預期記錄: {s['records']['expected']} 筆 ({s['period']['days']} 天)
- 實際記錄數: {s['records']['actual']} 筆
- 遺失記錄數: {s['records']['missed']} 筆
- 資料遺失率: {s['records']['missed_records_rate']:.2f}%"""

        # 只在資料不完整時顯示離線分析
        if s["records"]["missed_records_rate"] > 0:
            report += f"""

## 3. 離線分析
- 離線次數: {s['offline']['count']} 次
- 總離線時間: {s['offline']['minutes']:.2f} 分鐘

離線時長分布:"""

            for name, count in s["offline"]["ranges"].items():
                pct = (
                    count / s["offline"]["count"] * 100
                    if s["offline"]["count"] > 0
                    else 0
                )
                report += f"\n- {name}: {count}次 ({pct:.1f}%)"

            if s["offline"]["longest"]:
                report += "\n\n最長離線時段:"
                for i, p in enumerate(s["offline"]["longest"], 1):
                    report += f"\n {i}. {p['start'].strftime('%Y-%m-%d %H:%M:%S')} - {p['end'].strftime('%Y-%m-%d %H:%M:%S')}"
                    report += f"\n    - 持續時間: {p['duration']/60:.1f} 分鐘"
                    report += f"\n    - 遺失資料: {p['missed']} 筆"

        return report

    def _save_json_report(self, md_path: Path) -> None:
        """儲存 JSON 格式的報告資料"""
        json_path = md_path.with_suffix(".json")

        # 準備 JSON 資料
        json_stats = self.stats.copy()
        json_stats["period"]["start"] = str(json_stats["period"]["start"])
        json_stats["period"]["end"] = str(json_stats["period"]["end"])
        json_stats["config"] = asdict(self.config)
        json_stats["config"]["start_time"] = str(self.config.start_time)
        json_stats["config"]["end_time"] = str(self.config.end_time)

        # 轉換離線記錄中的時間格式
        if json_stats["offline"]["longest"]:
            for p in json_stats["offline"]["longest"]:
                p["start"] = str(p["start"])
                p["end"] = str(p["end"])

        json_path.write_text(
            json.dumps(json_stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.logger.debug("JSON data saved to: %s", json_path)


def is_valid_directory(path_str: str) -> bool:
    """
    驗證字串是否為合法的目錄路徑
    """
    try:
        path = Path(path_str)

        # 確保是目錄路徑，不是檔案路徑
        if path.suffix:
            return False

        # 檢查是否包含非法字元
        try:
            path.resolve()
        except (RuntimeError, ValueError):
            return False

        # 檢查目錄是否存在或可建立
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                return False

        return True
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"設備連線狀態分析工具 v{VERSION} By {AUTHOR} {BUILD_DATE}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s -i INPUT_FILE [-o OUTPUT_DIR]",
        epilog="""
使用範例:
  %(prog)s -i device_data.xlsx                  # 使用預設輸出目錄 output/
  %(prog)s -i device_data.csv -o reports        # 指定輸入檔案和輸出目錄
  %(prog)s --help                               # 顯示此說明

注意事項:
  - 支援的輸入檔案格式: .xlsx 或 .csv
  - 輸出檔名會自動根據輸入檔名產生，格式為: 輸入檔名_analysis.md
  - 若未指定輸出目錄，將自動使用 output 目錄
  - 分析結果會同時產生 .md 和 .json 檔案
""",
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="輸入檔案的路徑 (.xlsx 或 .csv)",
        metavar="INPUT_FILE",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="輸出報告的目錄路徑，若未指定則使用 output 目錄",
        metavar="OUTPUT_DIR",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    # 驗證輸入檔案格式
    if input_path.suffix.lower() not in [".xlsx", ".csv"]:
        parser.error("錯誤：輸入檔案必須是 .xlsx 或 .csv 格式")

    # 處理輸出目錄
    if args.output is None:
        output_dir = Path("output")
    else:
        # 驗證指定的輸出目錄
        if not is_valid_directory(args.output):
            parser.error(
                "錯誤：指定的輸出目錄路徑無效，請確認：\n"
                "1. 必須是目錄路徑，不能包含檔名\n"
                "2. 路徑格式正確\n"
                "3. 具有建立目錄的權限"
            )
        output_dir = Path(args.output)

    # 確保輸出目錄存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 自動生成輸出檔案路徑
    output_filename = f"{input_path.stem}_analysis.md"
    args.output = str(output_dir / output_filename)

    return args


def main() -> int:
    try:
        logger = setup_logger()
        logger.debug(
            "Application started, version: %s, build data: %s", VERSION, BUILD_DATE
        )

        args = parse_args()
        analyzer = DeviceAnalyzer()

        # 讀取檔案
        analyzer.read_file(Path(args.input))

        # 分析資料
        analyzer.analyze()

        # 儲存報告
        analyzer.save_report(Path(args.output))

        logger.debug("Analysis completed successfully")

        print(f"分析完成，請查看報告，報告位於: {analyzer.get_output_path()}")
        return 0
    except Exception as e:
        logger.error("Error occurred: %s", str(e), exc_info=True)
        print(f"錯誤: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
