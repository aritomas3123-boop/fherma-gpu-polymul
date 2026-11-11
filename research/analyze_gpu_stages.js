#!/usr/bin/env node
"use strict";
// Reads ACTUAL ARI_DIAG_MEDIAN lines captured from FHERMA Blackwell runner.
// This does not synthesize or estimate GPU stage timings.
const fs=require("fs");
const files=process.argv.slice(2);
if(!files.length){console.error("Usage: node research/analyze_gpu_stages.js official_stderr.log [more.log]");process.exit(2)}
const entries=[];
for(const file of files){
  const data=fs.readFileSync(file,"utf8");
  for(const m of data.matchAll(/ARI_DIAG_MEDIAN[^\r\n]*/g)){
    const line=m[0],n=/\bn=(\d+)/.exec(line),w=/\bw=(\d+)/.exec(line),calls=/\bcalls=(\d+)/.exec(line);
    if(!n||!w||!calls)continue;
    const stage=Object.fromEntries([...line.matchAll(/([A-Za-z_]+)_ms=(\d+(?:\.\d+)?)/g)].map(x=>[x[1],Number(x[2])]));
    if(!Number.isFinite(stage.full_wall))continue;
    entries.push({file,N:Number(n[1]),W:Number(w[1]),calls:Number(calls[1]),stage});
  }
}
if(!entries.length){console.error("NO REAL STAGE MEDIANS FOUND — refusing to invent benchmark timings.");process.exit(3)}
const output=[];
for(const e of entries){
 const values=Object.entries(e.stage).filter(([k])=>k!=="full_wall");
 // GPU event intervals overlap across streams; sort intervals to locate hotspots,
 // but do not add them or infer whole-call savings directly.
 values.sort((a,b)=>b[1]-a[1]);
 output.push({point:[e.N,e.W],source:e.file,measurements:e.calls,fullDiagnosticWallMs:e.stage.full_wall,
     largestMeasuredIntervals:values.slice(0,8).map(([name,ms])=>({stage:name,ms})),
     warnings:["Overlapping GPU intervals are not additive","Diagnostic CUDA events perturb measured wall time",
               "Do not extrapolate to 700 us from a single point","A/B official uninstrumented baseline required"]});
}
console.log(JSON.stringify({kind:"MEASURED BLACKWELL STAGE REPORT",profiles:output},null,2));
