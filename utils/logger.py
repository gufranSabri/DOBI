import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Logger:
    def __init__(self, file_path, file_mode="a", is_main_process=True):
        if not os.path.exists("/".join(file_path.split("/")[:-1])):
            os.mkdir("/".join(file_path.split("/")[:-1]))

        self.file_path        = file_path
        self.file_mode        = file_mode
        self.is_main_process  = is_main_process

    def __call__(self, message, is_main=True, console_print=False):
        # Under multi-GPU DDP every rank calls the logger; only the main process
        # (rank 0) should actually write, so other ranks don't interleave into the
        # same file or race on the fd.
        if is_main and self.is_main_process:
            with open(self.file_path, self.file_mode) as f:
                f.write(f"{message}\n")

                if console_print:
                    print(message)