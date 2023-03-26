import json
import subprocess
from pathlib import Path
from threading import Lock
from typing import Optional


class Alpaca:
    def __init__(self, alpaca_cli: Path, model_path: Path):
        self.alpaca_cli = alpaca_cli
        self.model_path = model_path
        self.process: Optional[subprocess.Popen] = None
        self.system_info: Optional[dict] = None
        self.lock = Lock()
        self.start()

    def start(self):
        if self.process is not None:
            return
        with self.lock:
            self.process = subprocess.Popen(
                args=[str(self.alpaca_cli), "--model", str(self.model_path)],
                text=True,
                bufsize=1,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            info_str = self._read_stdout()
            info = json.loads(info_str)
            self.system_info = {k: int(v) for k, v in info.items()}

    def _write(self, message: str) -> None:
        self.process.stdin.write(f"{message.strip()}\n")
        self.process.stdin.flush()

    def _read_stdout(self) -> str:
        return self.process.stdout.readline().strip()

    def stop(self):
        with self.lock:
            self._write("quit();")
            self.process.wait(10)
            self.process = None

    def run(
        self,
        text: str,
        top_k: int = 40,
        top_p: float = 0.95,
        temp: float = 0.1,
        repeat_penalty: float = 1.3,
        repeat_last_n: int = 64,
        n_predict: int = 128,
    ) -> dict:
        _input = {
            "input_text": text,
            "n_predict": str(n_predict),
            "top_k": str(top_k),
            "top_p": str(top_p),
            "temp": str(temp),
            "repeat_last_n": str(repeat_last_n),
            "repeat_penalty": str(repeat_penalty),
        }

        with self.lock:
            self._write(json.dumps(_input))
            response = self._read_stdout()
        output = json.loads(response)
        if "error" in output:
            raise Exception(output["error"])
        output = {k: v if k == "output" else int(v) for k, v in output.items()}
        return output

    def run_simple(self, text: str) -> dict:
        full_text = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n\n{text}### Response:\n\n"
        )
        return self.run(full_text)
