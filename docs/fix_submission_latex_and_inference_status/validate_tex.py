import re

with open("iclr2027/iclr2027_conference2.tex", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.splitlines()
bgroup_count = 0
egroup_count = 0
env_stack = []

for line_no, line in enumerate(lines, 1):
    c_idx = line.find("%")
    if c_idx >= 0:
        if c_idx > 0 and line[c_idx - 1] == "\\":
            pass
        else:
            line = line[:c_idx]

    bgroup_count += line.count(r"\bgroup")
    egroup_count += line.count(r"\egroup")

    for m in re.finditer(r"\\begin\{([^}]+)\}", line):
        env_stack.append((m.group(1), line_no))
    for m in re.finditer(r"\\end\{([^}]+)\}", line):
        env_name = m.group(1)
        if env_stack and env_stack[-1][0] == env_name:
            env_stack.pop()
        else:
            print(
                f"Env mismatch at line {line_no}: end {env_name}, stack {env_stack[-3:] if env_stack else None}"
            )

print(f"bgroup count: {bgroup_count}, egroup count: {egroup_count}")
print(f"Unclosed environments: {len(env_stack)}")
if env_stack:
    for env in env_stack:
        print(f"  Unclosed {env[0]} from line {env[1]}")
else:
    print("ALL ENVIRONMENTS PERFECTLY CLOSED!")
