import logging
import sys
from os import path
import sys
import uuid
from datetime import datetime

class DualOutput:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def write(self, message):
        if self.file:
            self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        if self.file:
            self.file.flush()
        self.stdout.flush()

    def __enter__(self):
        self.file = open(self.filename, 'w')
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        sys.stdout = self.stdout

def logging_setup(exp_path, task):

    unique_id = uuid.uuid4().hex[:6]
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_file = path.join(exp_path, f"{task}_{date_time}_{unique_id}.log")

    # make sure file doesn't exist
    if path.exists(log_file):
        raise FileExistsError(f"Log file already exists: {log_file}")

    sys.stdout = DualOutput(log_file)