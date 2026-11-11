'use strict';
// Synthetic exact compression feasibility for FHERMA (not competition data).
// Run: node research/compression_feasibility.js
const crypto=require('crypto'),zlib=require('zlib'),assert=require('assert');
const N=65536,W=1747,L=Math.ceil(W/32),bitsTop=W-(L-1)*32;
const originalBytes=N*L*4,raw=crypto.randomBytes(originalBytes);
for(let i=0;i<N;i++){
 const p=(i*L+L-1)*4;
 raw.writeUInt32LE(raw.readUInt32LE(p)&((2**bitsTop)-1),p);
}
const now=()=>Number(process.hrtime.bigint())/1e6;
let t=now(),deflated=zlib.deflateRawSync(raw,{level:1}),zipMs=now()-t;
t=now();const unzipped=zlib.inflateRawSync(deflated),unzipMs=now()-t;
assert(raw.equals(unzipped));
const packed=Buffer.alloc(N*Math.ceil(W/8));
t=now();const B=Math.ceil(W/8);
for(let i=0;i<N;i++)raw.copy(packed,i*B,i*L*4,i*L*4+B);
const packMs=now()-t;
for(let i=0;i<N;i++){
 const off=i*L*4,po=i*B;
 for(let j=0;j<B;j++)assert.strictEqual(raw[off+j],packed[po+j]);
 for(let j=B;j<L*4;j++)assert.strictEqual(raw[off+j],0);
}
const out={
 type:'synthetic uniform-bit stress test, NOT FHERMA benchmark',
 point:{N,W,L},operandBytes:originalBytes,exactBytePackingBytes:packed.length,
 exactBitPackingLowerBoundBytes:N*W/8,zlibBytes:deflated.length,
 bytePackingSavedBytes:originalBytes-packed.length,
 bitPackingPotentialSavedBytes:originalBytes-N*W/8,
 syntheticCPUms:{zip:Number(zipMs.toFixed(2)),unzip:Number(unzipMs.toFixed(2)),bytePack:Number(packMs.toFixed(2))},
 maxTheoreticalInputH2DTransferSavingUsAt64GBps:2*(originalBytes-N*W/8)/(64e9)*1e6,
 verified:true,
 warning:'Input data synthetic; CPU timings vary. Compression does not prove GPU speed. Packed data would require an unpack kernel and additional CPU work.'
};
console.log(JSON.stringify(out,null,2));
