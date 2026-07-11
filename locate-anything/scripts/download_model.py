from __future__ import annotations

import os

from huggingface_hub import snapshot_download

from locate_anything_service.config import Settings


def main() -> None:
    settings = Settings.from_env()
    local_dir = os.getenv("LOCATE_MODEL_DIR")
    path = snapshot_download(
        repo_id=settings.model_id,
        revision=settings.model_revision,
        token=settings.hf_token,
        local_dir=local_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
