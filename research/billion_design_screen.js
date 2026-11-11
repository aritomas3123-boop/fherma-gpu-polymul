#!/usr/bin/env node
"use strict";
// Exploratory screening of parameterized CUDA architectures for FHERMA.
// This is a STATIC SURROGATE, not a GPU simulator/benchmark or speed prediction.
// A full search space contains exactly 8^10 = 1,073,741,824 candidate tuples.
// 2,000,000 tuples are actually evaluated with a deterministic, seeded PRNG.
// A true latency estimate requires measured calibration on the same Blackwell runner.
const dims=[
 ["tile",[128,256,512,1024,2048,4096,8192,16384]],
 ["columns",[4,8,16,32,64,128,256,512]],
 ["workers",[1,2,4,8,16,32,64,128]],
 ["crtWidth",[4,8,16,32,64,128,256,512]],
 ["crtRows",[1,2,4,8,16,32,64,128]],
 ["primeGroup",[1,2,4,8,16,32,64,128]],
 ["encodeGroup",[1,2,4,8,16,32,64,128]],
 ["stagesPerLaunch",[1,2,3,4,6,8,12,16]],
 ["transfer",["current","pinned","async","pooled","double-buffer","staged","overlap","resident"]],
 ["nttDesign",["current","four-step","six-step","radix4","radix8","warp-shuffle","fused-inverse","tensor-core"]],
];
const N=65536,L=55,K=118,iterations=Number(process.argv[2]||2000000);
if(!Number.isSafeInteger(iterations)||iterations<1||iterations>20000000)throw Error("iterations must be 1..20000000");
let seed=0x7e57b00b;function next(){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return seed>>>0}
const top=[],rejections={nttThreads:0,nttShared:0,crtThreads:0,crtShared:0,encodeThreads:0,unsupported:0},seen=new Set();
function check(x){
 const [tile,columns,workers,crtWidth,crtRows,primeGroup,encodeGroup,stages,transfer,nttDesign]=x;
 if(tile/2>1024||tile>2048||N%tile!==0){rejections.nttThreads++;return false;}
 if(columns*workers>1024||columns>tile||tile%columns){rejections.nttThreads++;return false;}
 if(2*(N/tile)*columns*4>96*1024||2*tile*4>96*1024){rejections.nttShared++;return false;}
 if(crtWidth*crtRows>1024){rejections.crtThreads++;return false;}
 if(crtWidth*(16*L+4*(K+1+L+2))>96*1024){rejections.crtShared++;return false;}
 if(32*encodeGroup>1024){rejections.encodeThreads++;return false;}
 if(nttDesign!=="current"&&nttDesign!=="four-step"&&nttDesign!=="radix4"&&nttDesign!=="fused-inverse"){rejections.unsupported++;return false;}
 if(transfer==="resident"){rejections.unsupported++;return false;} // Timed host inputs cannot be assumed resident
 return true;
}
// This score is an intentionally simple architectural proxy.
// It is NOT microseconds and cannot certify any reduction.
function score(x){
 const [tile,cols,workers,cw,rows,pg,eg,st,transfer,ntt]=x;
 const transformBlocks=(N/tile + tile/cols)*K;
 const ctBlocks=Math.ceil(N/cw);
 const sharedNt=2*(N/tile)*cols*4;
 const sharedCRT=cw*(16*L+4*(K+1+L+2));
 const launchProxy=st>1?1/st:1;
 const transferRisk=transfer==="current"?0:transfer==="pinned"?40:transfer==="async"?80:300;
 const unverifiedDesignPenalty=ntt==="current"?0:250;
 return transformBlocks*launchProxy + ctBlocks*0.5 + sharedNt/128 + sharedCRT/512 + workers*0.2 + pg*0.1+eg*0.05 +transferRisk+unverifiedDesignPenalty;
}
let feasible=0;
const baseline=[512,16,16,8,16,8,8,1,"current","current"];
function insert(x){let s=score(x);if(top.length===20&&s>=top[top.length-1].proxy)return;const entry={proxy:s,configuration:Object.fromEntries(dims.map((d,i)=>[d[0],x[i]]))};top.push(entry);top.sort((a,b)=>a.proxy-b.proxy);if(top.length>20)top.pop();}
insert(baseline);
for(let i=0;i<iterations;i++){
 // Deterministic stratified pseudo-random exploration of 8^10 tuples.
 const idx=Array.from({length:10},()=>next()&7);
 const x=idx.map((v,j)=>dims[j][1][v]);
 if(!check(x))continue;
 feasible++;
 insert(x);
}
const result={
 kind:"STATIC HARDWARE-CONSTRAINT SCREENING, NOT A GPU MEASUREMENT",
 searchSpace:1073741824,actuallyEvaluated:iterations,passedFilters:feasible,
 rejected:rejections,baselineProxy:score(baseline),
 calibratedWithBlackwellStageTimings:false,measuredSpeedupUs:null,
 caution:"A low proxy score is not evidence of lower latency. Baseline must be profiled and candidates compiled, checked for exactness, then benchmarked on FHERMA.",
 shortlist:top
};
console.log(JSON.stringify(result,null,2));
