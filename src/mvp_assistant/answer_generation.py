from pathlib import Path

from .io_utils import write_submission


class CsvAnswerWriter:
    """Final step: serialize retrieved dish IDs into the required submission CSV."""

    def write(self, output_path: Path, rows: list[tuple[int, list[int]]]) -> Path:
        write_submission(output_path, rows)
        return output_path
