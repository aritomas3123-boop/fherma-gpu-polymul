#!/usr/bin/env node
// Static autotune configuration search, NOT a measured GPU benchmark.
"use strict";
const N=Number(process.argv[2]||65536), K=Number(process.argv[3]||118);
if (![32768,65536].includes(N)||!Number.isInteger(K)||K<1) throw Error("bad N/K");
const candidates=[], reject={threads:0,shared:0,geometry:0};
for(const tile of [256,512,1024])
for(const cols of [8,16,32])
for(const workers of [4,8,16,32])
for(const crt of [16,32])
for(const fused of [0,1]){
 const threads=cols*workers, highCount=N/tile;
 const highShared=2*highCount*cols*4;
 const lowShared=2*tile*4;
 if(tile%cols!==0 || cols>tile || highCount<2){reject.geometry++;continue;}
 if(threads>1024 || tile/2>1024){reject.threads++;continue;}
 if(Math.max(highShared,lowShared)>96*1024){reject.shared++;continue;}
 const blocks=(N/tile+tile/cols)*K;
 const proxy=blocks+highShared/512+threads/128+(crt===32?2:0)+(fused?1:0);
 candidates.push({tile,cols,workers,crt,fused,threads,highShared,lowShared,blocks,proxy,
 cmake:"-DFHERMA_NTT_MAX_TILE="+tile+" -DFHERMA_NTT_LOW_COLUMNS="+cols+" -DFHERMA_NTT_HIGH_WORKERS="+workers+" -DFHERMA_CRT_TILE_X="+crt+" -DFHERMA_FUSED_REINDEX="+fused+" -DFHERMA_PROFILE=OFF"});
}
candidates.sort((a,b)=>a.proxy-b.proxy||a.tile-b.tile||a.cols-b.cols);
const baseline=candidates.find(x=>x.tile===512&&x.cols===16&&x.workers===16&&x.crt===16&&x.fused===0);
console.log(JSON.stringify({kind:"STATIC FILTER — NOT A MEASURED SPEEDUP",point:{N,K},examined:candidates.length+Object.values(reject).reduce((a,b)=>a+b,0),valid:candidates.length,rejected:reject,baseline,recommendedForCompile:[baseline,...candidates.filter(x=>x.crt===16&&x.fused===0&&x.tile!==512).slice(0,4)],note:"Only an official RTX PRO 6000 run can establish speed and exactness."},null,2));
