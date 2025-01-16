import os
from pathlib import Path
import subprocess
import sys
import argparse
from datetime import datetime


def print_separator(char="-", length=80):
    """打印分隔線"""
    print(char * length)


def print_header(text):
    """印出帶有裝飾的標題"""
    print_separator("=", 80)
    print(f">>> {text}")
    print_separator("=", 80)


def print_section(text):
    """印出帶有簡單分隔線的段落標題"""
    print(f"\n{'-' * 50} {text} {'-' * 50}")


def format_time():
    """獲取格式化的當前時間"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def find_data_files(base_path):
    """遞迴尋找所有 xlsx 和 csv 檔案"""
    base_path = Path(base_path)
    patterns = ("*.xlsx", "*.csv")

    for pattern in patterns:
        for file_path in base_path.rglob(pattern):
            if file_path.is_file():
                yield file_path


def process_file(file_path, python_path):
    """處理單個檔案"""
    output_dir = file_path.parent / "output"
    output_dir.mkdir(exist_ok=True)

    cmd = [python_path, "./main.py", "-i", str(file_path), "-o", str(output_dir)]

    try:
        print_section(f"開始處理檔案: {file_path.name}")
        print(f"時間: {format_time()}")
        print(f"完整路徑: {file_path}")
        print(f"輸出目錄: {output_dir}")
        print("\n執行命令:")
        print(f"  {' '.join(cmd)}\n")

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        stdout, stderr = process.communicate()

        if stdout.strip():
            print_separator("-", 80)
            print(stdout.strip())
            print_separator("-", 80)

        if stderr.strip():
            print_separator("-", 80)
            print(stderr.strip())
            print_separator("-", 80)

        if process.returncode != 0:
            print(f"\n ⚠️ 警告: 程式返回錯誤代碼 {process.returncode}")
        else:
            print("\n✅ 處理完成")

    except Exception as e:
        print(f"\n[x] 執行時發生異常: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="處理 Excel 和 CSV 檔案的批次程式")
    parser.add_argument("base_path", help="要處理的根目錄路徑")
    parser.add_argument(
        "--exclude", nargs="+", default=[], help="要排除的目錄名稱（例如：output temp）"
    )

    args = parser.parse_args()

    print_header("批次檔案處理程式")
    print(f"開始時間: {format_time()}")
    print(f"根目錄: {args.base_path}")
    if args.exclude:
        print(f"排除目錄: {', '.join(args.exclude)}")
    print_separator()

    python_path = sys.executable
    print(f"Python 路徑: {python_path}")
    print_separator()

    total_files = 0
    processed_files = 0
    failed_files = []

    for file_path in find_data_files(args.base_path):
        if any(exclude in str(file_path) for exclude in args.exclude):
            continue

        total_files += 1
        try:
            process_file(file_path, python_path)
            processed_files += 1
        except Exception as e:
            failed_files.append((file_path, str(e)))

    # 顯示最終統計
    print_header("處理結果統計")
    print(f"結束時間: {format_time()}")
    print(f"總檔案數: {total_files}")
    print(f"成功處理: {processed_files}")
    print(f"失敗數量: {len(failed_files)}")

    if failed_files:
        print_section("失敗檔案清單")
        for file_path, error in failed_files:
            print(f"- {file_path.name}")
            print(f"  錯誤: {error}")

    print_separator("=")


if __name__ == "__main__":
    main()
