import pathlib, subprocess, re, sys, urllib.request
import tempfile
root = pathlib.Path(tempfile.mkdtemp(prefix="fherma-blackwell-")) / "source"
def run(*args):
    command = [str(a) for a in args]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    return result.stdout

gpu = run("nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader").strip()
if "RTX PRO 6000" not in gpu or "Blackwell" not in gpu:
    raise SystemExit("Expected an RTX PRO 6000 Blackwell, got: " + gpu)
run("nvcc", "--version")
arch = "sm_" + re.search(r"(\d+)\.(\d+)\s*$", gpu).group(0).replace(".", "").strip()
if not root.exists():
    run("git", "clone", "--depth", "1", "--branch", "experiment-tensor-core-crt",
        "https://github.com/aritomas3123-boop/fherma-gpu-polymul.git", root)
commit = "5ee9e0ab34ae82c294943c52262dba55105558fe"
if run("git", "-C", root, "rev-parse", "HEAD").strip() != commit:
    run("git", "-C", root, "fetch", "--depth", "1", "origin", commit)
    run("git", "-C", root, "checkout", "--detach", commit)
oracle = root / "oracle.py"
if not oracle.exists():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/fherma-ai/polynomial-mult-cupqc-example/main/oracle.py", oracle)
for name, flag in [("baseline", 0), ("tensor", 1)]:
    build = root / name
    run("cmake", "-S", root, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
        f"-DFHERMA_CUDA_ARCH={arch}", f"-DCMAKE_CUDA_FLAGS=-DFHERMA_TENSOR_CRT={flag}")
    run("cmake", "--build", build, "-j", "2")

import json, shutil
verified = 0
for n, w in [(128, 128), (1024, 128), (128, 1747), (256, 868)]:
    case = root / f"smoke-{n}-{w}"
    shutil.rmtree(case, ignore_errors=True)
    run(sys.executable, oracle, "make", case, n, w, 32)
    for name in ["baseline", "tensor"]:
        shutil.rmtree(case / "out", ignore_errors=True)
        run(root / name / "solution", case)
        run(sys.executable, oracle, "verify", case)
        rows = json.loads((case / "out" / "results.json").read_text())["cases"]
        assert len(rows) == 32 and all(row["status"] == "ok" for row in rows)
        print("VERIFIED", n, w, name, len(rows), flush=True)
    verified += 32
print("SMOKE TEST COMPLETE", verified, "cases per version", flush=True)

import hashlib, importlib.util, json, random, shutil, statistics, sys
spec = importlib.util.spec_from_file_location("fherma_oracle", oracle)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()

for n, w in [(32768, 868), (65536, 1747)]:
    print("PREPARING FULL SIZE", n, w, flush=True)
    length = (w + 31) // 32
    q = reference.negacyclic_prime(n, w)
    case = root / f"full-{n}-{w}"
    shutil.rmtree(case, ignore_errors=True)
    (case / "point").mkdir(parents=True)
    (case / "point" / "q.bin").write_bytes(reference.pack([q], length))
    (case / "manifest.json").write_text(json.dumps(dict(N=n, W=w, L=length, cases=12)))
    expected = {}
    for i in range(12):
        rng = random.Random(560101 + 97 * i + n)
        a = [rng.randrange(q) for _ in range(n)]
        if i < 6:
            b = [rng.randrange(q) for _ in range(n)]
        else:
            b = [0] * n
            shifts = [0, n - 1] if i == 6 else rng.sample(range(n), 2)
            b[shifts[0]], b[shifts[1]] = q - 1, rng.randrange(1, q)
            answer = [0] * n
            for shift in shifts:
                for index, value in enumerate(a):
                    j = index + shift
                    sign = 1 if j < n else -1
                    answer[j % n] = (answer[j % n] + sign * value * b[shift]) % q
            expected[i] = hashlib.sha256(reference.pack(answer, length)).digest()
        folder = case / "cases" / f"{i:06}"
        folder.mkdir(parents=True)
        (folder / "a.bin").write_bytes(reference.pack(a, length))
        (folder / "b.bin").write_bytes(reference.pack(b, length))
    baseline = None
    times = {"baseline": [], "tensor": []}
    for name in ["baseline", "tensor", "tensor", "baseline"]:
        shutil.rmtree(case / "out", ignore_errors=True)
        run(root / name / "solution", case)
        rows = json.loads((case / "out" / "results.json").read_text())["cases"]
        assert len(rows) == 12 and all(row["status"] == "ok" for row in rows), rows
        observed = [digest(case / "out" / f"{i:06}" / "c.bin") for i in range(12)]
        assert all(observed[i] == value for i, value in expected.items()), (n, w, name, "independent mismatch")
        if baseline is None:
            baseline = observed
        else:
            assert baseline == observed, (n, w, name, "differential mismatch")
        times[name].extend(row["seconds"] * 1000 for row in rows[1:6])
        print("FULL VERIFIED", n, w, name, "6 dense differential / 6 sparse reference", flush=True)
    base_ms = statistics.median(times["baseline"])
    tensor_ms = statistics.median(times["tensor"])
    print(f"FULL RESULT N={n} W={w}: BASELINE {base_ms:.4f} ms / TENSOR {tensor_ms:.4f} ms", flush=True)
print("FULL TEST COMPLETE", flush=True)
