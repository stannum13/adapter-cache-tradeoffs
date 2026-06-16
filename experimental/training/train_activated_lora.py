from __future__ import annotations

import argparse
import importlib.metadata


def _peft_supports_alora() -> bool:
    try:
        version = importlib.metadata.version("peft")
    except importlib.metadata.PackageNotFoundError:
        return False
    return "alora" in version.lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--invocation-token", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.parse_args()
    if not _peft_supports_alora():
        raise SystemExit(
            "Installed PEFT does not advertise aLoRA-style invocation token support. "
            "Use the activated_lora cache simulator path for CPU experiments."
        )
    raise SystemExit("Activated LoRA training hook is ready for a supported PEFT build.")


if __name__ == "__main__":
    main()
