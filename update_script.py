# pm2 start update_script.py --interpreter python3 -- --repo_path /your/repo/path --repo_url https://github.com/amedeo-gigaver/infinite_games.git --branch main --check_interval 600 --pm2_process_name miner or validator
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path


def run_command(command, cwd=None):
    """Run a shell command and return the output."""
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd
    )
    return result.stdout.decode("utf-8").strip(), result.stderr.decode("utf-8").strip()


def check_for_updates(repo_path, branch, pm2_process_name):
    # Fetch latest updates from the repository
    print(f"Fetching updates from remote repository in {repo_path}...")
    run_command("git fetch origin", cwd=repo_path)

    # Check if there are any updates in the remote branch
    local_commit, local_error = run_command(f"git rev-parse {branch}", cwd=repo_path)
    remote_commit, remote_error = run_command(f"git rev-parse origin/{branch}", cwd=repo_path)

    if local_commit != remote_commit:
        print("New updates found. Pulling changes...")
        # Ensure the working directory is clean before pulling
        reset_output, reset_error = run_command("git reset --hard", cwd=repo_path)
        if reset_error:
            print(f"Error during git reset: {reset_error}")
            return
        pull_output, pull_error = run_command("git pull", cwd=repo_path)
        if pull_error:
            print(f"Error during git pull: {pull_error}")
        else:
            print("Successfully pulled changes. Restarting pm2 process...")
            restart_output, restart_error = run_command(f"pm2 restart {pm2_process_name}")
            if restart_error:
                print(f"Error restarting pm2 process: {restart_error}")
            else:
                print("Successfully restarted pm2 process.")
    else:
        print("No updates found. Exiting...")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--repo_path", required=True, help="Path to your git repository")
    parser.add_argument("--repo_url", required=True, help="URL of your git repository")
    parser.add_argument("--branch", required=True, help="Git branch to monitor")
    parser.add_argument(
        "--check_interval",
        type=int,
        required=True,
        help="Interval to check for any new updates (in seconds)",
    )
    parser.add_argument(
        "--pm2_process_name", required=True, help="Name of the pm2 process to restart"
    )

    args = parser.parse_args()

    if Path(args.repo_path).exists():
        while True:
            check_for_updates(args.repo_path, args.branch, args.pm2_process_name)
            print(f"Sleeping for {args.check_interval / 60} minutes...")
            time.sleep(args.check_interval)
    else:
        print(f"Invalid repository path: {args.repo_path}. Please ensure the path exists.")
