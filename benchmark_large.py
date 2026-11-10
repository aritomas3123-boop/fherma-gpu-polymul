#!/usr/bin/env python3
"""Large FHERMA-like end-to-end timings with dense random A and sparse B.
Not the official FHERMA dataset or official GPU, but the output is fully checked.
One case is warm-up, followed by four cases per shape.
"""
import importlib.util, json, pathlib, random, shutil, statistics, struct, subprocess, time
ROOT=pathlib.Path("/content/fherma-test")
spec=importlib.util.spec_from_file_location("fherma_oracle",ROOT/"oracle.py")
oracle=importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)

def prepare(N,W,cases=5):
    L=(W+31)//32
    d=pathlib.Path("/content")/f"fherma-full-{N}-{W}"
    if d.exists(): shutil.rmtree(d)
    d.mkdir()
    (d/"point").mkdir()
    q=oracle.negacyclic_prime(N,W)
    (d/"point"/"q.bin").write_bytes(oracle.pack([q],L))
    (d/"manifest.json").write_text(json.dumps({"N":N,"W":W,"L":L,"cases":cases}))
    print("Preparing dense random A, sparse B; N=",N,"W=",W,"L=",L,flush=True)
    for seed in range(cases):
        rand=random.Random(seed+987654)
        a=[rand.randrange(q) for _ in range(N)]
        b=[0]*N
        positions=rand.sample(range(N),2)
        for pos in positions:b[pos]=rand.randrange(1,q)
        expected=[0]*N
        # A is completely dense; B has two nonzero coefficients to make exact
        # verification feasible without O(N^2) multi-thousand-bit Python loops.
        for shift in positions:
            scalar=b[shift]
            for i,av in enumerate(a):
                index=i+shift
                if index>=N:
                    index-=N
                    expected[index]=(expected[index]-av*scalar)%q
                else:
                    expected[index]=(expected[index]+av*scalar)%q
        folder=d/"cases"/f"{seed:06d}"
        folder.mkdir(parents=True)
        (folder/"a.bin").write_bytes(oracle.pack(a,L))
        (folder/"b.bin").write_bytes(oracle.pack(b,L))
        exp=d/"expected"/f"{seed:06d}"
        exp.mkdir(parents=True)
        (exp/"c.bin").write_bytes(oracle.pack(expected,L))
        print(" prepared case",seed+1,"of",cases,flush=True)
    return d

def measure(d,N,W,label):
    shutil.rmtree(d/"out",ignore_errors=True)
    subprocess.run([f"/content/fherma-{label}/solution",str(d)],check=True)
    subprocess.run(["python3",str(ROOT/"oracle.py"),"verify",str(d)],check=True)
    result=json.loads((d/"out"/"results.json").read_text())
    rows=result["cases"]
    if not all(x["status"]=="ok" for x in rows):raise RuntimeError(f"{label} has a crashed case: {rows}")
    values=[x["seconds"]*1e6 for x in rows]
    med=statistics.median(values[1:])
    print(f"N={N} W={W} {label}: first={values[0]:.1f} us; warm median={med:.1f} us; all={[round(v,1) for v in values]}; VERIFY PASS",flush=True)
    return med

def main():
    print("CUDA GPU: ",subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True,check=True).stdout.strip())
    for N,W in ((32768,868),(65536,1747)):
        d=prepare(N,W)
        base=measure(d,N,W,"baseline")
        fused=measure(d,N,W,"fused")
        print(f"SHAPE N={N} W={W}: baseline={base:.1f} us vs fused={fused:.1f} us (change={(base-fused)/base*100:+.1f}%)",flush=True)
    print("NOT AN OFFICIAL FHERMA SCORE. Measured Tesla T4 with structured sparse operand B; needs official Blackwell full random benchmark.")

if __name__=="__main__":main()
