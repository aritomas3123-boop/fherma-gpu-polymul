// RNS CRT exact GEMM algebraic prototype, no NVIDIA GPU benchmark.
// Matrix operands: T[N x (K+1)] canonical unsigned uint32 CRT terms;
// W[(K+1) x L] unsigned uint32 wide CRT basis words.
// Unlike the current scalar sum_column, this admits 16x16 INT8 Tensor Core GEMM.
// Each uint32 = sum_{d=0..3} byte[d]*2^(8d), giving 16 byte GEMMs.
// Recombine with BigInt carry logic exactly; no FP rounding.
"use strict";
const assert=require("assert").strict;
const K=118, P=K+1, L=55, ROWS=17;
let state=0x69420117;
function rnd(){state^=state<<13;state^=state>>>17;state^=state<<5;return state>>>0}
function exactScalar(terms,weights,row,col){
 let acc=0n;for(let j=0;j<P;j++)acc+=BigInt(terms[row*P+j])*BigInt(weights[j*L+col]);return acc
}
function exactInt8TensorAlgebra(terms,weights,row,col){
 const dots=new Uint32Array(16);
 for(let j=0;j<P;j++){
   const a=terms[row*P+j],b=weights[j*L+col];
   for(let u=0;u<4;u++)for(let v=0;v<4;v++)
     dots[u*4+v]+=((a>>>(u*8))&255)*((b>>>(v*8))&255)
 }
 let acc=0n;
 for(let u=0;u<4;u++)for(let v=0;v<4;v++){
   assert(dots[u*4+v]<=P*255*255,"INT32 dot range exceeded");
   acc+=BigInt(dots[u*4+v])<<BigInt(8*(u+v))
 }
 return acc;
}
function exactKernelSevenDiagonal(terms,weights,row,col){
 const D=new Uint32Array(7);
 for(let j=0;j<P;j++){
  const a=terms[row*P+j],b=weights[j*L+col];
  for(let u=0;u<4;u++)for(let v=0;v<4;v++)
   D[u+v]+=((a>>>(u*8))&255)*((b>>>(v*8))&255);
 }
 let carry=0n,words=[0n,0n,0n];
 for(let d=0;d<7;d++){
  carry+=BigInt(D[d]);
  words[Math.floor(d/4)]|=(carry&255n)<<BigInt(8*(d%4));
  carry>>=8n;
 }
 words[1]|=(carry&255n)<<24n;
 words[2]=carry>>8n;
 return words[0]+(words[1]<<32n)+(words[2]<<64n);
}
const modes=["zero","maximum","alternating","random","highbits","lowbits"];
let checks=0;
for(const mode of modes){
 const t=new Uint32Array(ROWS*P),w=new Uint32Array(P*L);
 function value(i){if(mode==="zero")return 0;if(mode==="maximum")return 0xffffffff;
 if(mode==="alternating")return i%2?0xffffffff:0;
 if(mode==="highbits")return 0xffff0000;
 if(mode==="lowbits")return 0x0000ffff;return rnd()}
 for(let i=0;i<t.length;i++)t[i]=value(i);
 for(let i=0;i<w.length;i++)w[i]=value(i+37);
 for(let row=0;row<ROWS;row++)for(let col=0;col<L;col++){
   const a=exactScalar(t,w,row,col),b=exactInt8TensorAlgebra(t,w,row,col);
   assert.equal(a,b,"CRT GEMM mismatch");
   assert.equal(a,exactKernelSevenDiagonal(t,w,row,col),"seven-diagonal CUDA reconstruction mismatch");
   assert(a < (1n<<72n),"CRT sum overflow > 72 bits");
   checks++;
 }
 console.log("PASS "+mode+" exact 119-prime x 55-limb CRT dot products");
}
console.log("TOTAL EXACT MATCHES="+checks);
console.log("Single output dot is < 119*(2^32-1)^2 < 2^71; 16 UINT8 dots each < 7,737,975, fitting INT32.");
console.log("NOT a CUDA implementation, throughput measurement, or evidence of a 700-us whole call.");
