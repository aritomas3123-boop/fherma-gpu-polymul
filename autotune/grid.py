#!/usr/bin/env python3
"""Bounded, reproducible GPU autotuner for FHERMA-like candidate selection.

This is NOT a billion GPU benchmark or the official ranking. The 12-candidate
configuration grid is compiled, then timed on the attached GPU (if any).
Before any reported timing, all output is checked with an independent exact
reference, with the scorer's fherma_run-inclusive timing boundary.
"""
import argparse
import csv
import itertools
import json
import pathlib
import re
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from benchmark_large import prepare, measure

def command(*args):
    print("+", " ".join(map(str,args)),flush=True)
    subprocess.run([str(x) for x in args], check=True)

def gpu_arch():
    result=subprocess.check_output(
        ["nvidia-smi","--query-gpu=compute_cap","--format=csv,noheader"],
        text=True).strip().splitlines()[0]
    m=re.search(r"(\d+)\.(\d+)",result)
    if not m:
        raise RuntimeError("Could not determine GPU compute capability")
    return "sm_"+m.group(1)+m.group(2)

def combos():
    # One independent variable at a time in the source, bounded 12-combination grid.
    for tile, columns, crt in itertools.product((256,512,1024),(8,16),(16,32)):
        yield (tile,columns,crt)

def build(tile,columns,crt,arch):
    name=f"ntt{tile}-col{columns}-crt{crt}"
    dest=pathlib.Path("/content")/f"tune-{name}"
    defs=[
        f"-DFHERMA_CUDA_ARCH={arch}","-DCMAKE_BUILD_TYPE=Release",
        f"-DFHERMA_NTT_TILE={tile}",f"-DFHERMA_NTT_COLUMNS={columns}",
        "-DFHERMA_NTT_WORKERS=16",f"-DFHERMA_CRT_TILE_X={crt}",
        "-DFHERMA_NTT_TILED=ON","-DFHERMA_FUSED_REINDEX=OFF",
        "-DFHERMA_PROFILE=OFF",
    ]
    command("cmake","-S",HERE,"-B",dest,*defs)
    command("cmake","--build",dest,"-j","2")
    return name,dest/"solution"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-builds",type=int,default=12)
    ap.add_argument("--max-minutes",type=int,default=40)
    ap.add_argument("--finalists",type=int,default=3)
    ap.add_argument("--output",default="/content/fherma-autotune-results.csv")
    args=ap.parse_args()
    if not(1<=args.max_builds<=12 and 1<=args.finalists<=12 and 1<=args.max_minutes<=120):
        raise ValueError("Bounds: builds 1-12, finalists 1-12, time 1-120 min")
    arch=gpu_arch()
    print("GPU active; compute architecture",arch,flush=True)
    # Exact test input generation is independent from the candidate and its outputs.
    # The second operand is intentionally sparse to make Python big-int oracle feasible;
    # official FHERMA random-input testing is still mandatory.
    small=prepare(32768,868,cases=3)
    start=time.monotonic()
    rows=[]
    build_count=0
    for tile,columns,crt in combos():
        if build_count>=args.max_builds or (time.monotonic()-start)>args.max_minutes*60:
            print("STOP: bounded limit reached",flush=True)
            break
        build_count+=1
        name,exe=build(tile,columns,crt,arch)
        # benchmark_large.measure expects two fixed executable paths; use
        # a new label corresponding to the candidate rather than altering any source.
        target=pathlib.Path("/content")/f"fherma-{name}"
        target.mkdir(exist_ok=True)
        out=target/"solution"
        if out.exists(): out.unlink()
        out.symlink_to(exe)
        record=dict(name=name,ntt_tile=tile,columns=columns,crt=crt,
                    shape="32768x868",median_us="",correct="",error="")
        try:
            value=measure(small,32768,868,name)
            record["median_us"]=round(value,3)
            record["correct"]="PASS"
        except Exception as ex:
            record["correct"]="FAIL"
            record["error"]=str(ex)[:400]
        rows.append(record)
        with open(args.output,"w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=record.keys())
            writer.writeheader();writer.writerows(rows)
        print("CANDIDATE",json.dumps(record),flush=True)
    valid=sorted((r for r in rows if r["correct"]=="PASS"),
                 key=lambda r:float(r["median_us"]))
    print("PRELIMINARY RANK (T4 or connected GPU only):",json.dumps(valid),flush=True)
    # The largest FHERMA-sized benchmark is run on shortlisted correct candidates.
    # Full official Blackwell testing still required.
    if valid:
        large=prepare(65536,1747,cases=3)
        for r in valid[:min(args.finalists,len(valid))]:
            if (time.monotonic()-start)>args.max_minutes*60:
                print("STOP: time limit reached before next finalist",flush=True);break
            try:
                value=measure(large,65536,1747,r["name"])
                print("FINALIST",r["name"],"large_shape_us",round(value,3),"EXACTNESS PASS",flush=True)
            except Exception as ex:
                print("FINALIST FAIL",r["name"],str(ex),flush=True)
    print("DONE; no official FHERMA submission and no unverified speed claim",flush=True)

if __name__=="__main__":
    main()
