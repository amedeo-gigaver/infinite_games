import subprocess

from neurons.validator.utils.env import ENVIRONMENT_VARIABLES


def get_commit_short_hash() -> str:
    try:
        commit_short_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
        )

        return commit_short_hash
    except Exception:
        return ENVIRONMENT_VARIABLES.GIT_COMMIT_HASH


commit_short_hash = get_commit_short_hash()
