import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

SESSION_FILE = Path(__file__).parent / ".colab_session.json"

def load_session():
    if not SESSION_FILE.exists():
        print("[!] No active Colab session found. Run: python colab.py set-ssh <ssh_target>")
        sys.exit(1)
    with open(SESSION_FILE, "r") as f:
        return json.load(f)

def save_session(data):
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_base_ssh(target):
    # target can be user@host or host
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=NUL" if os.name == "nt" else "/dev/null",
        target
    ]

def cmd_set_ssh(args):
    raw = args.target.strip()
    if raw.startswith("ssh "):
        raw = raw[4:].strip()
    
    session = {"target": raw}
    save_session(session)
    print(f"[*] Saved target: {raw}")
    print("[*] Testing connection to remote GPU...")
    
    # Run test
    test_cmd = get_base_ssh(raw) + ["nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"]
    res = subprocess.run(test_cmd, capture_output=True, text=True)
    if res.returncode == 0:
        gpu_info = res.stdout.strip()
        print(f"[+] Connected successfully! Remote GPU:\n    {gpu_info}")
        
        kaggle_token = os.environ.get("KAGGLE_API_TOKEN")
        if kaggle_token:
            print("[*] Forwarding KAGGLE_API_TOKEN to remote...")
            subprocess.run(get_base_ssh(raw) + [f"echo 'export KAGGLE_API_TOKEN={kaggle_token}' >> ~/.bashrc; pip install -q kaggle"])
            print("[+] Remote Kaggle configured!")
    else:
        print(f"[!] Warning: SSH test returned code {res.returncode}:\n{res.stderr}\n{res.stdout}")

def cmd_exec(args):
    session = load_session()
    target = session["target"]
    remote_command = " ".join(args.command)
    print(f"[*] Running: {remote_command}")
    subprocess.run(get_base_ssh(target) + [remote_command])

def cmd_gpu(args):
    session = load_session()
    target = session["target"]
    subprocess.run(get_base_ssh(target) + ["nvidia-smi"])

def main():
    parser = argparse.ArgumentParser(description="Antigravity Colab Remote Bridge CLI")
    subparsers = parser.add_subparsers(dest="action", required=True)

    # set-ssh
    p_set = subparsers.add_parser("set-ssh", help="Set the active SSH target (e.g. user@host.tmate.io)")
    p_set.add_argument("target", help="SSH string or host")
    p_set.set_defaults(func=cmd_set_ssh)

    # exec
    p_exec = subparsers.add_parser("exec", help="Execute command on Colab")
    p_exec.add_argument("command", nargs=argparse.REMAINDER, help="Remote command to execute")
    p_exec.set_defaults(func=cmd_exec)

    # gpu
    p_gpu = subparsers.add_parser("gpu", help="Check remote GPU status")
    p_gpu.set_defaults(func=cmd_gpu)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
