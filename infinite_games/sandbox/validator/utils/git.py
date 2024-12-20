import subprocess


def get_commit_short_hash():
    try:
        commit_short_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
        )

        return commit_short_hash
    except Exception:
        return None


commit_short_hash = get_commit_short_hash()
