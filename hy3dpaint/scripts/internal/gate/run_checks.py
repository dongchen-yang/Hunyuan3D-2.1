"""One command for a step's checks (REFACTOR_TODO.md, How we work), after TEXGen-Emission's fanout.py.

  python3 run_checks.py --rev <commit> --checks paper,release,fresh,resume [--full] [--label <name>]

1. The tree under test: `git archive <rev> hy3dpaint` into $C/trees/<rev>.
2. Gates (paper, release): on the workstation 4090 when it is free right now (both lock files, 20 GB
   of GPU memory, 28 GB of host RAM); otherwise, and for the training checks (fresh, resume), on
   Solar: one job (star2/checks.sbatch) on a GPU type that has baselines, submitted with a deadline,
   so it starts at once or goes away. Nothing waits for a busy GPU; what could not run is reported
   INCOMPLETE.
3. Each result is compared with its GPU type's baseline and envelope (baselines.json):
   compare_views.py for the gates, compare_traincheck.py for the training checks.
Prints one verdict line: PASS, FAIL or INCOMPLETE (exit 0, 1, 2). Gates run the subset20 shapes
unless --full. Outputs: $C/out/<type>/<label>/<check>/.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))   # hy3dpaint/scripts/internal/gate -> fork root
C = os.environ.get("GATE_C", "/local-scratch/localhome/dya78/lightgen_eval_local/hunyuan_refactor_checks")
LOCKS = ["/local-scratch/localhome/dya78/lightgen_4090.lock", "/local-scratch/localhome/dya78/.locks/gpu0-rtx4090.lock"]
PY = "/localhome/dya78/software/miniconda3/envs/hunyuanpaint/bin/python"
SOLAR = {  # type -> gres, cpus and memory for a 3-GPU job
    "bwse": {"gres": "gpu:rtx_pro_6000_blackwell_se:3", "ngpu": 3, "cpus": 12, "mem": "160G"},
}


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


def log(msg):
    print(f"[run_checks {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_tree(rev):
    full = sh(f"git -C {FORK} rev-parse --verify {rev}^{{commit}}").stdout.strip()
    if not full:
        sys.exit(f"unknown revision {rev}")
    name = rev if rev == "pre-refactor" else full[:10]
    tree = os.path.join(C, "trees", name)
    if not os.path.isdir(os.path.join(tree, "hy3dpaint")):
        os.makedirs(tree + ".part", exist_ok=True)
        r = sh(f"git -C {FORK} archive {full} hy3dpaint | tar -x -C {tree}.part")
        if r.returncode:
            sys.exit(f"git archive failed: {r.stderr}")
        os.replace(tree + ".part", tree)
    return name, tree


def local_4090_free():
    for lk in LOCKS:
        if sh(f"flock -n {lk} true").returncode:
            return f"lock {lk} held"
    used = sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 0").stdout.split(",")
    if len(used) == 2 and int(used[1]) - int(used[0]) < 20000:
        return f"only {int(used[1]) - int(used[0])} MiB of GPU memory free"
    avail = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemAvailable"))) // 2**20
    if avail < 28:
        return f"only {avail} GB of host RAM available"
    return None


def run_local_gates(tree, label, gates, full):
    out = {}
    for g in gates:
        d = os.path.join(C, "out", "rtx4090", label, g)
        shas = [] if full else ["subset20"]
        os.makedirs(os.path.dirname(d), exist_ok=True)
        log(f"4090: {g} gate ({'full' if full else 'subset20'})")
        with open(d + ".log", "w") as f:
            rc = subprocess.run(["bash", os.path.join(HERE, "gate.sh"), tree, d, g] + shas, stdout=f, stderr=subprocess.STDOUT).returncode
        out[g] = d if rc == 0 else None
        if rc:
            log(f"4090: {g} gate exit {rc} (log {d}.log)")
    return out


def run_solar(tree_name, tree, label, tasks, typ, hours=2.0):
    """tasks: [(name, kind, shas)]; returns {name: local dir or None}."""
    spec = SOLAR[typ]
    dest = os.path.join(C, "out", typ, label)
    if os.path.exists(dest) or sh(f"ssh star2 'test -e ~/hunyuan_checks/out/{label}'", timeout=60).returncode == 0:
        log(f"solar: label {label} was used before; pick a new one")
        return {n: None for n, _, _ in tasks}
    taskfile = os.path.join(C, "star2_tasks", f"{label}.txt")
    os.makedirs(os.path.dirname(taskfile), exist_ok=True)
    with open(taskfile, "w") as f:
        for i, (name, kind, shas) in enumerate(tasks):
            f.write(f"{i % spec['ngpu']} {name} {kind} {tree_name} {shas}\n")
    r = sh(f"ssh star2 'mkdir -p ~/hunyuan_checks/{{trees,tooling,inputs,tasks,logs,out}}' && "
           f"rsync -a {tree}/ star2:hunyuan_checks/trees/{tree_name}/ && "
           f"rsync -a --delete {HERE}/ star2:hunyuan_checks/tooling/gate/ && "
           f"rsync -a {C}/tooling/repo/ star2:hunyuan_checks/tooling/repo/ && "
           f"rsync -a {C}/inputs/ star2:hunyuan_checks/inputs/ && "
           f"rsync -a {taskfile} star2:hunyuan_checks/tasks/", timeout=900)
    if r.returncode:
        log(f"solar: sync failed: {r.stderr[-300:]}")
        return {n: None for n, _, _ in tasks}
    t = f"{int(hours):02d}:{int(hours % 1 * 60):02d}:00"
    script = open(os.path.join(HERE, "star2", "checks.sbatch")).read()
    for k, v in {"__LABEL__": label, "__TASKS__": f"tasks/{label}.txt", "__NGPU__": str(spec["ngpu"]),
                 "__GRES__": spec["gres"], "__NCPU__": str(spec["cpus"]), "__MEM__": spec["mem"], "__TIME__": t}.items():
        script = script.replace(k, v)
    r = subprocess.run(["ssh", "star2", f"sbatch --deadline=now+{int(hours * 60) + 5}minutes"], input=script,
                       text=True, capture_output=True, timeout=120)
    jid = next((w for w in r.stdout.split() if w.isdigit()), None)
    if not jid:
        log(f"solar: sbatch failed: {r.stdout} {r.stderr}")
        return {n: None for n, _, _ in tasks}
    log(f"solar: job {jid} ({typ}), {len(tasks)} tasks")
    time.sleep(90)
    st = sh(f"ssh star2 'squeue -h -j {jid} -o %T'", timeout=60).stdout.split()
    if "PENDING" in st:
        sh(f"ssh star2 'scancel {jid}'", timeout=60)
        log(f"solar: job {jid} did not start at once; canceled (no free {typ} GPUs now)")
        return {n: None for n, _, _ in tasks}
    status = ""
    while True:
        status = sh(f"ssh star2 'cat ~/hunyuan_checks/out/{label}/status.txt 2>/dev/null'", timeout=60).stdout
        if "CHECKS_" in status:
            break
        if not sh(f"ssh star2 'squeue -h -j {jid} -o %T'", timeout=60).stdout.strip():
            time.sleep(30)
            status = sh(f"ssh star2 'cat ~/hunyuan_checks/out/{label}/status.txt 2>/dev/null'", timeout=60).stdout
            break
        time.sleep(180)
    os.makedirs(dest)
    r = sh(f"rsync -a star2:hunyuan_checks/out/{label}/ {dest}/", timeout=1800)
    log(f"solar: job {jid} status:\n" + status.strip())
    if r.returncode or f"job={jid}" not in status:
        log(f"solar: results not usable (copy exit {r.returncode}; status of job {jid}: {'yes' if f'job={jid}' in status else 'no'})")
        return {n: None for n, _, _ in tasks}
    ok = {l.split()[0] for l in status.splitlines() if " exit=0 " in l}
    return {n: (os.path.join(dest, n) if n in ok else None) for n, _, _ in tasks}


def compare(typ, check, cand, base, shas):
    if typ not in base or check not in base[typ]:
        return False, [f"no {check} baseline for GPU type {typ}"]
    env = base[typ][check]
    if check in ("paper", "release"):
        # the shapes the baseline runs cover: every shape, or the subset20 list (baselines.json "shas")
        eshas = os.path.join(C, "inputs", check, env.get("shas", "shas.txt"))
        want = {l.strip() for l in open(shas) if l.strip()}
        have = {l.strip() for l in open(eshas) if l.strip()}
        if want - have:
            return False, [f"the {typ} {check} baseline covers {len(have)} shapes, not these {len(want)}"]
        cmd = [PY, os.path.join(HERE, "compare_views.py"), "--a", env["run"], "--b", cand, "--shas_file", shas]
        if len(env.get("envelope", [])) >= 4:   # the pass rule: at least 4 runs of pre-refactor
            cmd += ["--envelope", *env["envelope"], "--envelope_shas", eshas,
                    "--envelope_json", os.path.join(C, "envelopes", f"{typ}_{check}_{os.path.basename(eshas)}.json")]
    else:
        cmd = ["python3", os.path.join(HERE, "compare_traincheck.py"), "--a", env["run"], "--b", os.path.join(cand, "record.jsonl")]
        if len(env.get("envelope", [])) >= 4:
            cmd += ["--envelope", *env["envelope"]]
    r = subprocess.run(cmd, text=True, capture_output=True)
    lines = [l for l in r.stdout.splitlines() if "verdict=" in l or "views:" in l or "structure:" in l]
    return r.returncode == 0, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--checks", default="paper,release")
    ap.add_argument("--full", action="store_true", help="every gate shape instead of subset20")
    ap.add_argument("--label")
    ap.add_argument("--solar_type", default="bwse")
    a = ap.parse_args()
    checks = a.checks.split(",")
    base = json.load(open(os.path.join(HERE, "baselines.json")))
    tree_name, tree = make_tree(a.rev)
    label = a.label or f"{tree_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    gates = [c for c in checks if c in ("paper", "release")]
    train = [c for c in checks if c in ("fresh", "resume")]
    results, where = {}, {}
    why = local_4090_free() if gates else None
    local_gates = gates if gates and why is None and "rtx4090" in base else []
    if gates and not local_gates:
        log(f"4090 not used: {why}")
    solar_tasks, skipped = [], []
    for c in gates:
        if c in local_gates:
            continue
        cover = base.get(a.solar_type, {}).get(c, {}).get("shas")
        if a.full and cover != "shas.txt":
            skipped.append(c)   # the Solar baseline covers subset20 only: nothing could judge a full run
        else:
            solar_tasks.append((c, c, "" if a.full else "subset20"))
    if skipped:
        log(f"not run: {','.join(skipped)} --full needs the 4090 (the {a.solar_type} baselines cover subset20)")
    solar_tasks += [(c, c, "") for c in train]
    import threading
    solar_out = {}
    th = None
    if solar_tasks:
        th = threading.Thread(target=lambda: solar_out.update(run_solar(tree_name, tree, label, solar_tasks, a.solar_type)))
        th.start()
    if local_gates:
        for g, d in run_local_gates(tree, label, local_gates, a.full).items():
            results[g], where[g] = d, "rtx4090"
    if th:
        th.join()
        for n, d in solar_out.items():
            results[n], where[n] = d, a.solar_type
    verdict, lines = "PASS", []
    for c in checks:
        d = results.get(c)
        if d is None:
            verdict = "INCOMPLETE" if verdict == "PASS" else verdict
            lines.append(f"{c}: did not run or did not finish ({where.get(c, 'nowhere')})")
            continue
        shas = os.path.join(C, "inputs", c, "shas.txt" if a.full else "subset20.txt") if c in ("paper", "release") else None
        ok, out = compare(where[c], c, d, base, shas)
        lines.append(f"{c} [{where[c]}]: " + " | ".join(out))
        if not ok:
            verdict = "FAIL"
    for l in lines:
        print(f"[run_checks] {l}")
    print(f"[run_checks] rev={a.rev} label={label} verdict={verdict}")
    sys.exit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[verdict])


if __name__ == "__main__":
    main()
