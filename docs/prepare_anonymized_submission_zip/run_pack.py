import subprocess
import sys

cmd = [
    "/mnt/nas/home/hiromi/src/emo/.venv/bin/python",
    "scripts/package_submission_code.py",
]
print("Running:", " ".join(cmd), flush=True)
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in process.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()

process.wait()
sys.exit(process.returncode)
