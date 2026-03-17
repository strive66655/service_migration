import csv
from pathlib import Path
from datetime import datetime


class ResultSaver:

    def __init__(self, save_dir="results"):

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.file_path = Path(save_dir) / f"experiment_{timestamp}.csv"

        self.file = open(self.file_path, "w", newline="", encoding="utf-8")

        self.writer = csv.writer(self.file)

        # 写表头
        self.writer.writerow([
            "time_step",
            "avg_latency",
            "step_migrations",
            "total_migrations"
        ])

    def write(self, time_step, avg_latency, step_migrations, total_migrations):

        self.writer.writerow([
            time_step,
            avg_latency,
            step_migrations,
            total_migrations
        ])

    def close(self):
        self.file.close()