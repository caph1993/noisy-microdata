from pathlib import Path
import csv


class CSV_writer:
    """
    CSV writer for writing line by line as experiments are executed

    Does not support multithreaded writing to the same file.
    """

    def __init__(self, filename):
        self.file = Path(filename)
        self.keys, self.rows = self.read()

    def write(self, row: dict):
        self.ensure_fields(row.keys())
        with open(self.file, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.keys)
            writer.writerow(row)
            self.rows.append(row)

    def ensure_fields(self, fields):
        new_keys = [key for key in fields if key not in self.keys]
        if not new_keys:
            return
        # Rewrite header and all existing rows with the new keys
        # keys, rows = self.read()
        keys = [*self.keys, *new_keys]
        for row in self.rows:
            for key in new_keys:
                if key not in row:
                    row[key] = ""
        with open(self.file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)
        self.keys = keys

    def read(self):
        try:
            with open(self.file, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                keys = [*(reader.fieldnames or [])]
                rows = [*reader]
        except FileNotFoundError:
            keys = []
            rows = []
        return keys, rows
