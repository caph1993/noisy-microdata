import sys
from io import StringIO


class SilenceStdout:
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = StringIO()

    def __exit__(self, *args):
        sys.stdout = self.stdout
