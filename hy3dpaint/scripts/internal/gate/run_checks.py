"""One command for a step's checks, fanned out over every free GPU (REFACTOR_TODO.md, How we work;
after TEXGen-Emission's fanout.py).

  python3 run_checks.py --rev <commit> --checks paper,release,fresh,resume [--full] [--label <name>]
                        [--types rtx4090,bwse,...] [--entry auto|parent|fork] [--plan] [--record_baseline]

1. The tree under test: `git archive <rev> hy3dpaint` into $C/trees/<rev>.
2. Slots, found once, right now (nothing waits for a GPU): the workstation 4090 and sp4090 when free
   (their lock files, 20 GB of GPU memory, 28 GB of host RAM), and on Solar every free GPU of a type
   with baselines on our nodes whose node also has 6 free CPUs and 48 GB of unreserved memory. Gate
   shapes are split over the slots by speed (a shape's views do not depend on the shapes run before
   it; measured on the 4090 and the Blackwell); the training checks go to slots of a type with
   training baselines. Solar slots are single-GPU jobs submitted with a deadline, so a job that
   cannot start at once goes away; what did not run is reported INCOMPLETE.
3. Each type's pieces of a gate are merged and compared with that type's baseline (compare_views.py;
   the envelope only with >= 4 runs); the training checks with compare_traincheck.py.
Prints one verdict: PASS, FAIL or INCOMPLETE (exit 0, 1, 2). --plan prints the split and stops.
--record_baseline (rev pre-refactor, --full, gates only) adds the merged outputs to baselines.json
as runs of their type. Gates run the subset20 shapes unless --full. Outputs: $C/out/<type>/<label>/.
--entry fork runs the gates with the tree's own hy3dpaint/predict_emission_views.py (gate.sh's
GATE_ENTRY) against the same baselines, which the frozen parent predictor wrote, and passes only when
IDENTICAL. --entry auto (default): parent while the tree has the old loader module the parent predictor
imports, else fork (from step 14b on).
"""
import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))   # hy3dpaint/scripts/internal/gate -> fork root
C = os.environ.get("GATE_C", "/local-scratch/localhome/dya78/lightgen_eval_local/hunyuan_refactor_checks")
WS_LOCKS = ["/local-scratch/localhome/dya78/lightgen_4090.lock", "/local-scratch/localhome/dya78/.locks/gpu0-rtx4090.lock"]
SP_LOCK = "/local-scratch/localhome/dya78/lightgen_4090.lock"
PY = "/localhome/dya78/software/miniconda3/envs/hunyuanpaint/bin/python"
ENTRY = "parent"   # gate.sh GATE_ENTRY, set by --entry
TYPES = {   # seconds per shape on a free GPU (the split); Solar gres names
    "rtx4090": {"speed": 4.2},
    "bwse": {"speed": 3.3, "gres": "rtx_pro_6000_blackwell_se", "train": True},   # train: >= 48 GB
    "bwmq": {"speed": 3.4, "gres": "rtx_pro_6000_blackwell_max-q", "train": True},
    "l40s": {"speed": 5.5, "gres": "l40s"},
    "a40": {"speed": 8.0, "gres": "a40"},
}
NOT_OURS = {"cs-venus-18", "cs-venus-20"}   # theia-lab's L40S nodes
SLOT_CPUS, SLOT_MEM_MB = 6, 48 * 1024
GATES, TRAIN = ("paper", "release"), ("fresh", "resume")
MIN_SHAPES_PER_SLOT = 8   # below this the model load dominates


def sh(cmd, timeout=120):
    try:
        return subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", "timeout")


# Every call to star2 shares one connection: two dozen threads logging in at once hit star2's limit on
# new connections and failed (2026-10-17), and the poll read a failed squeue as a finished job.
_MUX = "-o ControlMaster=auto -o ControlPath=~/.ssh/cm-hunyuan-checks-%C -o ControlPersist=900 -o BatchMode=yes"
SSH = f"ssh {_MUX} star2"
RSYNC = f"rsync -e 'ssh {_MUX}'"


def star2(cmd, timeout=120, tries=6):
    """sh() for a command that talks to star2, retried while ssh itself fails (exit 255; rsync's 12 and
    255 when its connection drops). Returns the last result: a caller reads exit 255 as "unknown"."""
    for k in range(tries):
        r = sh(cmd, timeout)
        if r.returncode not in (255, 12):
            return r
        time.sleep(5 * 2 ** k)
    return r


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


# ------------------------------------------------------------------ slots

def ws_free():
    for lk in WS_LOCKS:
        if sh(f"flock -n {lk} true").returncode:
            return f"lock {lk} held"
    q = sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 0").stdout.split(",")
    if len(q) == 2 and int(q[1]) - int(q[0]) < 20000:
        return f"{int(q[1]) - int(q[0])} MiB of GPU memory free"
    avail = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemAvailable"))) // 2**20
    return None if avail >= 28 else f"{avail} GB of host RAM available"


def sp_free():
    r = sh(f"ssh -o BatchMode=yes -o ConnectTimeout=10 sp4090 'flock -n {SP_LOCK} true && "
           "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 0 && "
           "grep MemAvailable /proc/meminfo'", timeout=40)
    if r.returncode:
        return "lock held or unreachable"
    lines = r.stdout.strip().split("\n")
    used, total = [int(x) for x in lines[0].split(",")]
    avail = int(lines[1].split()[1]) // 2**20
    if total - used < 20000:
        return f"{total - used} MiB of GPU memory free"
    return None if avail >= 28 else f"{avail} GB of host RAM available"


def solar_slots(types):
    """{type: [node, ...]}: one entry per usable GPU of the type on our nodes, right now."""
    r = star2(f"{SSH} 'scontrol -o show node'", timeout=60)
    out = {t: [] for t in types}
    gres2type = {TYPES[t]["gres"]: t for t in types}
    for line in r.stdout.splitlines():
        f = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
        node = f.get("NodeName", "")
        if not node.startswith("cs-venus-") or node in NOT_OURS:
            continue
        if any(s in f.get("State", "") for s in ("DOWN", "DRAIN", "FAIL", "RESERVED", "MAINT", "NOT_RESPONDING")):
            continue
        m = re.match(r"gpu:([^:(]+):(\d+)", f.get("Gres", ""))
        if not m or m.group(1) not in gres2type:
            continue
        alloc = re.search(r"gres/gpu=(\d+)", f.get("AllocTRES", ""))
        free_gpu = int(m.group(2)) - (int(alloc.group(1)) if alloc else 0)
        free_cpu = int(f.get("CPUTot", 0)) - int(f.get("CPUAlloc", 0))
        free_mem = int(f.get("RealMemory", 0)) - int(f.get("AllocMem", 0))
        out[gres2type[m.group(1)]] += [node] * max(0, min(free_gpu, free_cpu // SLOT_CPUS, free_mem // SLOT_MEM_MB))
    return out


# ------------------------------------------------------------------ plan

def gate_shas(gate, full):
    return [l.strip() for l in open(os.path.join(C, "inputs", gate, "shas.txt" if full else "subset20.txt")) if l.strip()]


def covers(base, typ, check, full=True):
    """A type judges a check once it has 2 runs that agree exactly, or 4 for an envelope."""
    e = base.get(typ, {}).get(check)
    n = len(e.get("envelope", [])) if e else 0
    return (n >= 4 or (n >= 2 and e.get("identical", True))) and (not full or e.get("shas", "shas.txt") == "shas.txt")


def split(n, slots):
    """Sizes of n shapes over slots, by speed."""
    w = [1.0 / TYPES[s["type"]]["speed"] for s in slots]
    sizes = [int(n * x / sum(w)) for x in w]
    for k in range(n - sum(sizes)):
        sizes[k % len(sizes)] += 1
    return sizes


def plan(gates, train, full, base, types, record):
    slots, why = [], {}
    if "rtx4090" in types and (record or "rtx4090" in base):
        why["ws"] = ws_free()
        if why["ws"] is None:
            slots.append({"type": "rtx4090", "host": "ws"})
        if base.get("_admitted_hosts", {}).get("sp4090") == "rtx4090":
            why["sp4090"] = sp_free()
            if why["sp4090"] is None:
                slots.append({"type": "rtx4090", "host": "sp4090"})
    solar_types = [t for t in types if "gres" in TYPES[t] and (record or t in base)]
    for t, nodes in solar_slots(solar_types).items():
        slots += [{"type": t, "host": "solar", "node": n} for n in nodes]
    jobs = []   # (slot index, kind, name, shas or None)
    tslots = [i for i, s in enumerate(slots) if TYPES[s["type"]].get("train")
              and (record or all(covers(base, s["type"], c) for c in train))]
    tslots.sort(key=lambda i: TYPES[slots[i]["type"]]["speed"])
    for k, c in enumerate(train):
        if tslots:
            jobs.append((tslots[k % len(tslots)], c, c, None))
    busy = {j[0] for j in jobs}
    for g in gates:
        shas = gate_shas(g, full)
        ok = [i for i, s in enumerate(slots) if (record or covers(base, s["type"], g, full)) and i not in busy]
        if not ok:
            ok = [i for i, s in enumerate(slots) if record or covers(base, s["type"], g, full)]
        ok.sort(key=lambda i: (slots[i]["host"] == "solar", TYPES[slots[i]["type"]]["speed"]))   # lab boxes start at once
        ok = ok[:max(1, len(shas) // MIN_SHAPES_PER_SLOT)]
        start = 0
        for i, n in zip(ok, split(len(shas), [slots[i] for i in ok]) if ok else []):
            if n:
                jobs.append((i, g, f"{g}_s{i}", shas[start:start + n])); start += n
    return slots, jobs, why


# ------------------------------------------------------------------ run

def write_shas(path, shas):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(shas) + "\n")


def run_ws(tree, label, mine, results):
    for _, kind, name, shas in mine:
        d = os.path.join(C, "out", "rtx4090", label, name)
        write_shas(d + ".shas", shas)
        with open(d + ".log", "w") as lf:
            rc = subprocess.run(["bash", os.path.join(HERE, "gate.sh"), tree, d, kind, d + ".shas"],
                                stdout=lf, stderr=subprocess.STDOUT,
                                env=dict(os.environ, GATE_ENTRY=ENTRY)).returncode
        results[name] = (d, shas, "rtx4090") if rc == 0 else None
        if rc:
            log(f"ws: {name} exit {rc} ({d}.log)")


def run_sp(tree_name, tree, label, mine, results):
    rc = sh(f"rsync -a {tree}/ sp4090:{C}/trees/{tree_name}/ && rsync -a --delete {HERE}/ sp4090:{C}/tooling/gate/ "
            f"&& ssh sp4090 'mkdir -p {C}/out/rtx4090/{label}'", timeout=900).returncode
    for _, kind, name, shas in mine:
        d = f"{C}/out/rtx4090/{label}/{name}"   # the same path on both machines
        write_shas(d + ".shas", shas)
        ok = False
        if not rc and sh(f"rsync -a {d}.shas sp4090:{d}.shas").returncode == 0:
            r = sh(f"ssh sp4090 'GATE_C={C} GATE_LOCKS={SP_LOCK} GATE_ACTIVATE={C}/tooling/activate_sp4090.sh GATE_ENTRY={ENTRY} "
                   f"bash {C}/tooling/gate/gate.sh {C}/trees/{tree_name} {d} {kind} {d}.shas > {d}.log 2>&1'", timeout=7200)
            sh(f"rsync -a sp4090:{d}/ {d}/ && rsync -a sp4090:{d}.log {d}.log", timeout=900)
            ok = r.returncode == 0 and os.path.isfile(os.path.join(d, "manifest.json"))
        results[name] = (d, shas, "rtx4090") if ok else None
        if not ok:
            log(f"sp4090: {name} failed ({d}.log)")


def run_solar(tree_name, label, i, slot, mine, results, hours=1.5):
    typ, jl = slot["type"], f"{label}_s{i}"
    tdir = os.path.join(C, "star2_tasks", jl)
    lines = []
    for _, kind, name, shas in mine:
        if kind in GATES and ENTRY == "fork":
            kind += "+fork"
        if shas is None:
            lines.append(f"0 {name} {kind} {tree_name}")
        else:
            write_shas(os.path.join(tdir, f"{name}.txt"), shas)
            lines.append(f"0 {name} {kind} {tree_name} tasks/{jl}/{name}.txt")
    write_shas(os.path.join(tdir, "tasks.txt"), lines)
    train = any(k in TRAIN for _, k, _, _ in mine)
    script = open(os.path.join(HERE, "star2", "checks.sbatch")).read()
    t = f"{int(hours):02d}:{int(hours % 1 * 60):02d}:00"
    for k, v in {"__LABEL__": jl, "__TASKS__": f"tasks/{jl}/tasks.txt", "__NGPU__": "1",
                 "__GRES__": f"gpu:{TYPES[typ]['gres']}:1", "__NCPU__": str(SLOT_CPUS),
                 "__MEM__": "120G" if train else "48G", "__TIME__": t}.items():
        script = script.replace(k, v)
    fail = lambda why: (log(f"solar {typ} s{i}: {why}"), [results.__setitem__(n, None) for _, _, n, _ in mine])
    if star2(f"{SSH} 'mkdir -p ~/hunyuan_checks/tasks/{jl}' && {RSYNC} -a {tdir}/ star2:hunyuan_checks/tasks/{jl}/").returncode:
        return fail("task sync failed")
    sub_cmd = (f"sbatch --parsable --deadline=now+{int(hours * 60) + 5}minutes "
               f"--exclude={','.join(sorted(NOT_OURS))}")
    for k in range(6):   # resubmit only when ssh failed before sbatch could print a job id
        r = subprocess.run(["ssh", *_MUX.split(), "star2", sub_cmd], input=script, text=True,
                           capture_output=True, timeout=120)
        if not (r.returncode == 255 and not r.stdout.strip()):
            break
        time.sleep(5 * 2 ** k)
    jid = r.stdout.strip().split(";")[0]
    if r.returncode or not jid.isdigit():
        return fail(f"sbatch failed: {r.stdout} {r.stderr}")
    log(f"solar {typ} s{i}: job {jid}: {', '.join(n for _, _, n, _ in mine)}")
    time.sleep(120)
    if "PENDING" in star2(f"{SSH} 'squeue -h -j {jid} -o %T'", timeout=60).stdout:
        star2(f"{SSH} 'scancel {jid}'", timeout=60)
        return fail(f"job {jid} did not start at once; canceled")
    read_status = lambda: star2(f"{SSH} 'cat ~/hunyuan_checks/out/{jl}/status.txt 2>/dev/null || true'", timeout=60)
    unknown = 0
    while True:
        st = read_status()
        if st.returncode == 0 and "CHECKS_" in st.stdout:
            status = st.stdout
            break
        q = star2(f"{SSH} 'squeue -h -j {jid} -o %T'", timeout=60)
        if q.returncode in (0, 1) and not q.stdout.strip():   # the job is gone (1: no longer known)
            time.sleep(30)
            status = read_status().stdout
            break
        unknown = unknown + 1 if q.returncode not in (0, 1) else 0
        if unknown >= 20:   # star2 unreachable for ~30 min: give up on this slot, the job keeps its outputs
            status = ""
            log(f"solar {typ} s{i}: star2 unreachable; job {jid} left running")
            break
        time.sleep(90)
    dest = os.path.join(C, "out", typ, label, f"s{i}")
    os.makedirs(dest, exist_ok=True)
    good = (star2(f"{RSYNC} -a star2:hunyuan_checks/out/{jl}/ {dest}/", timeout=1800).returncode == 0
            and f"job={jid}" in status)
    ok = {l.split()[0] for l in status.splitlines() if " exit=0 " in l} if good else set()
    for _, kind, name, shas in mine:
        results[name] = (os.path.join(dest, name), shas, typ) if name in ok else None
    log(f"solar {typ} s{i}: job {jid} done, {len(ok)} of {len(mine)} ok")


def dispatch(tree_name, tree, label, slots, jobs):
    results, by_slot = {}, {}
    for j in jobs:
        by_slot.setdefault(j[0], []).append(j)
    solar_ok = True
    if any(slots[i]["host"] == "solar" for i in by_slot):
        r = star2(f"{SSH} 'mkdir -p ~/hunyuan_checks/trees ~/hunyuan_checks/tasks ~/hunyuan_checks/logs ~/hunyuan_checks/out' && "
                  f"{RSYNC} -a {tree}/ star2:hunyuan_checks/trees/{tree_name}/ && {RSYNC} -a --delete {HERE}/ star2:hunyuan_checks/tooling/gate/ && "
                  f"{RSYNC} -a {C}/tooling/repo/ star2:hunyuan_checks/tooling/repo/ && {RSYNC} -a {C}/inputs/ star2:hunyuan_checks/inputs/", timeout=900)
        solar_ok = r.returncode == 0
        if not solar_ok:
            log(f"solar: sync failed: {r.stderr[-300:]}")
    ths = []
    for i, mine in by_slot.items():
        s = slots[i]
        if s["host"] == "ws":
            th = threading.Thread(target=run_ws, args=(tree, label, mine, results))
        elif s["host"] == "sp4090":
            th = threading.Thread(target=run_sp, args=(tree_name, tree, label, mine, results))
        elif solar_ok:
            th = threading.Thread(target=run_solar, args=(tree_name, label, i, s, mine, results))
        else:
            for _, _, n, _ in mine:
                results[n] = None
            continue
        th.start(); ths.append(th)
    for th in ths:
        th.join()
    return results


# ------------------------------------------------------------------ compare

def merge(label, typ, gate, pieces):
    m = os.path.join(C, "out", typ, label, f"merged_{gate}")
    os.makedirs(m, exist_ok=True)
    for d in pieces:
        sh(f"rsync -a --exclude manifest.json --exclude gpu.txt --exclude examples.json --exclude '.*' {d}/ {m}/", timeout=900)
    return m


def compare_gate(typ, gate, cand, shas, base):
    env = base[typ][gate]
    eshas = os.path.join(C, "inputs", gate, env.get("shas", "shas.txt"))
    if set(shas) - {l.strip() for l in open(eshas) if l.strip()}:
        return False, [f"the {typ} {gate} baseline does not cover these shapes"]
    f = cand + ".shas"
    write_shas(f, shas)
    cmd = [PY, os.path.join(HERE, "compare_views.py"), "--a", env["run"], "--b", cand, "--shas_file", f]
    if len(env.get("envelope", [])) >= 4:
        cmd += ["--envelope", *env["envelope"], "--envelope_shas", eshas,
                "--envelope_json", os.path.join(C, "envelopes", f"{typ}_{gate}_{os.path.basename(eshas)}.json")]
    r = subprocess.run(cmd, text=True, capture_output=True)
    out = [l for l in r.stdout.splitlines() if "verdict=" in l or "views:" in l]
    # The fork entry must reproduce the parent predictor's bytes (step 13), not only within noise.
    ok = r.returncode == 0 and (ENTRY != "fork" or any("verdict=IDENTICAL" in l for l in out))
    return ok, out


def compare_train(typ, check, cand, base):
    env = base[typ][check]
    cmd = ["python3", os.path.join(HERE, "compare_traincheck.py"), "--a", env["run"], "--b", os.path.join(cand, "record.jsonl")]
    if len(env.get("envelope", [])) >= 4:
        cmd += ["--envelope", *env["envelope"]]
    r = subprocess.run(cmd, text=True, capture_output=True)
    return r.returncode == 0, [l for l in r.stdout.splitlines() if "verdict=" in l or "structure:" in l]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--checks", default="paper,release")
    ap.add_argument("--full", action="store_true", help="every gate shape instead of subset20")
    ap.add_argument("--label")
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--plan", action="store_true", help="print the split and stop")
    ap.add_argument("--record_baseline", action="store_true")
    ap.add_argument("--entry", choices=("auto", "parent", "fork"), default="auto",
                    help="the gates' predictor: the frozen parent one, the tree's own, or auto (by the tree)")
    a = ap.parse_args()
    checks, types = a.checks.split(","), a.types.split(",")
    gates, train = [c for c in checks if c in GATES], [c for c in checks if c in TRAIN]
    bpath = os.path.join(HERE, "baselines.json")
    base = json.load(open(bpath))
    tree_name, tree = make_tree(a.rev)
    global ENTRY
    old_loader = os.path.isfile(os.path.join(tree, "hy3dpaint", "src", "data", "dataloader", "lightgen_emission_loader.py"))
    ENTRY = a.entry if a.entry != "auto" else ("parent" if old_loader else "fork")
    if ENTRY == "parent" and not old_loader:
        sys.exit(f"the parent predictor imports src.data.dataloader.lightgen_emission_loader, which {a.rev} lacks; "
                 "use --entry fork")
    if a.record_baseline and (a.rev != "pre-refactor" or (gates and not a.full) or ENTRY != "parent"
                              or len(types) != 1):
        sys.exit("--record_baseline takes --rev pre-refactor, --full for gates, the parent entry and one "
                 "--types (a run split over types would record each type's part as a full run)")
    log(f"gates run the {ENTRY} predictor")
    label = a.label or f"{tree_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    if os.path.exists(os.path.join(C, "out", "rtx4090", label)) or any(
            os.path.exists(os.path.join(C, "out", t, label)) for t in TYPES):
        sys.exit(f"label {label} was used before")
    slots, jobs, why = plan(gates, train, a.full, base, types, a.record_baseline)
    for h, w in why.items():
        if w:
            log(f"{h} not used: {w}")
    by = {}
    for i, kind, name, shas in jobs:
        by.setdefault(i, []).append(f"{name}({len(shas) if shas else 'train'})")
    for i, items in sorted(by.items()):
        s = slots[i]
        log(f"slot {i}: {s['type']} {s['host']}{' ' + s['node'] if s['host'] == 'solar' else ''}: {' '.join(items)}")
    if not jobs:
        log("no free slot for any check")
    if a.plan:
        return 0
    results = dispatch(tree_name, tree, label, slots, jobs)
    verdict, lines, merged = "PASS", [], {}
    for g in gates:
        req, done = gate_shas(g, a.full), {}
        if not req:
            verdict = "FAIL"
            lines.append(f"{g}: the shapes file lists no shape")
            continue
        for _, kind, name, shas in jobs:
            if kind == g and results.get(name):
                d, s, typ = results[name]
                done.setdefault(typ, ([], []))
                done[typ][0].append(d)
                done[typ][1].extend(s)
        missing = set(req) - {x for v in done.values() for x in v[1]}
        if missing:
            verdict = "INCOMPLETE" if verdict == "PASS" else verdict
            lines.append(f"{g}: {len(missing)} of {len(req)} shapes did not run")
        for typ, (pieces, s) in sorted(done.items()):
            m = merge(label, typ, g, pieces)
            merged[(typ, g)] = m
            if a.record_baseline:
                lines.append(f"{g} [{typ}]: {len(s)} shapes -> {m}")
                continue
            ok, out = compare_gate(typ, g, m, s, base)
            lines.append(f"{g} [{typ}, {len(s)} shapes]: " + " | ".join(out))
            if not ok:
                verdict = "FAIL"
    for c in train:
        r = next((results.get(n) for _, k, n, _ in jobs if k == c), None)
        if not r:
            verdict = "INCOMPLETE" if verdict == "PASS" else verdict
            lines.append(f"{c}: did not run or did not finish")
            continue
        if a.record_baseline:
            merged[(r[2], c)] = os.path.join(r[0], "record.jsonl")
            lines.append(f"{c} [{r[2]}]: {merged[(r[2], c)]}")
            continue
        ok, out = compare_train(r[2], c, r[0], base)
        lines.append(f"{c} [{r[2]}]: " + " | ".join(out))
        if not ok:
            verdict = "FAIL"
    if a.record_baseline and verdict == "PASS":
        # Each new run is compared with the type's first; the file is locked, as two record runs of
        # different types may finish together.
        with open(os.path.join(C, "baselines.lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            disk = json.load(open(bpath))
            for (typ, g), m in sorted(merged.items()):
                e = disk.setdefault(typ, {}).setdefault(g, {"run": None, "envelope": []} if g in TRAIN else
                                                        {"run": None, "shas": "shas.txt", "envelope": []})
                if e["run"]:
                    cmd = (["python3", os.path.join(HERE, "compare_traincheck.py"), "--a", e["run"], "--b", m]
                           if g in TRAIN else
                           [PY, os.path.join(HERE, "compare_views.py"), "--a", e["run"], "--b", m,
                            "--shas_file", os.path.join(C, "inputs", g, "shas.txt")])
                    r = subprocess.run(cmd, text=True, capture_output=True)
                    v = re.search(r"verdict=(\w+)", r.stdout)
                    v = v.group(1) if v else f"no verdict (exit {r.returncode})"
                    e["identical"] = e.get("identical", True) and v == "IDENTICAL"
                    lines.append(f"{g} [{typ}]: run {len(e['envelope']) + 1} vs run 1: {v}")
                e["envelope"].append(m)
                e["run"] = e["run"] or m
            with open(bpath + ".tmp", "w") as f:
                json.dump(disk, f, indent=1)
            os.replace(bpath + ".tmp", bpath)   # a reader never sees a half-written file
        lines.append("baselines.json updated (a type judges a check with 2 identical runs, or 4 runs as an envelope)")
    for l in lines:
        print(f"[run_checks] {l}")
    print(f"[run_checks] rev={a.rev} label={label} verdict={verdict}")
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
