from pathlib import Path

import typer

from alpaca import Alpaca

app = typer.Typer()


@app.command()
def main(alpaca_cli_path: Path = typer.Option(...), model_path: Path = typer.Option(...)):
    alpaca = Alpaca(alpaca_cli_path, model_path)
    try:
        while True:
            _input = input("> ").strip()
            if _input in ("q", "quit", "exit", "quit()", "exit()", "quit();", "exit();"):
                return
            if not _input:
                continue
            output = alpaca.run_simple(_input)
            print(output["output"])
    except KeyboardInterrupt:
        pass
    finally:
        alpaca.stop()


if __name__ == "__main__":
    app()
