import os
import glob
import time
import argparse
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [BATCH-JUDGE] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results_judged")
    parser.add_argument("--max-parallel", type=int, default=4)

    return parser.parse_args()


def get_response_files(dir_path):
    if not os.path.exists(dir_path):
        logging.error(f"Directory {dir_path} does not exist")
        raise FileNotFoundError(f"Directory {dir_path} does not exist")

    search_path = os.path.join(dir_path, "*.jsonl")
    files = glob.glob(search_path)

    valid_files = []
    for f in files:
        f = f.replace('\\', '/')
        if "_scores.jsonl" not in f:
            valid_files.append(f)

    if not valid_files:
        logger.warning(f"No valid response files found in {dir_path}")

    return valid_files


def run_single_judge(input_file, output_dir):
    base_name = os.path.basename(input_file)
    run_name = os.path.splitext(base_name)[0]

    expected_output = os.path.join(output_dir, f"{run_name}_judged.jsonl")

    if os.path.exists(expected_output):
        logger.warning(f"Judge result for {run_name} already exist.")

    logger.info(f"Judging {run_name}...")

    cmd = [
        "python", "main_judge.py",
        "--input-file", input_file,
        "--output-dir", output_dir,
        "--run-name", run_name,
    ]

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        logger.info(f"Finished judging {run_name} in {duration:.2f}s")

    except subprocess.CalledProcessError:
        logger.error(f"Failed to judge {run_name}")


def run(args):
    input_files = get_response_files(args.input_dir)

    logger.info(f"Found {len(input_files)} response files to judge.")
    logger.info(f"Parallel Number: {args.max_parallel}")

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [
            executor.submit(
                run_single_judge,
                input_file,
                args.output_dir,
            )
            for input_file in input_files
        ]

        # 等待完成
        for _ in as_completed(futures):
            pass

    logger.info(f"Batch judging completed. Processed {len(futures)} files.")


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
