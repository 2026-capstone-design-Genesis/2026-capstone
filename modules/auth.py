import getpass
from typing import Optional


def get_hf_token() -> Optional[str]:
    """Ask for a Hugging Face token at program startup without saving it.

    - Input is hidden in the terminal via getpass.
    - The token is returned only as an in-memory string.
    - It is not written to config, .env, json outputs, logs, or cache files.
    - Press Enter to continue without a token, useful for already-cached/public models.
    """
    token = getpass.getpass(
        "Hugging Face token 입력(없으면 Enter, 입력값은 저장되지 않음): "
    ).strip()
    return token or None
