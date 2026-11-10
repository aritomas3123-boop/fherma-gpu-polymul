// Experimental CPU correctness oracle for unsigned 32 x 32 -> exact wide CRT dot products,
// decomposed to 16 signed INT8 dot products plus exact zero-point corrections.
// NO GPU/Tensor Core kernel yet. NOT a performance result.
"use strict";
const K=118, LIMBS=55, MASK=0xffffffffn;
let seed=0xa543e21d;
function rand(){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return seed>>>0;}
function dotOriginal(a,b) {
 let sum=0n;
 for(let k=0;k<a.length;k++)sum+=BigInt(a[k])*BigInt(b[k]);
 return sum;
}
function dotInt8(a,b){
 const x=new Int32Array(K*4), y=new Int32Array(K*4);
 const sumX=[0,0,0,0],sumY=[0,0,0,0];
 for(let k=0;k<K;k++)for(let i=0;i<4;i++){
   const ix=((a[k]>>>(8*i))&255)-128, iy=((b[k]>>>(8*i))&255)-128;
   x[k*4+i]=ix;y[k*4+i]=iy;sumX[i]+=ix;sumY[i]+=iy;
 }
 let total=0n;
 for(let i=0;i<4;i++)for(let j=0;j<4;j++){
   let acc=0;
   for(let k=0;k<K;k++)acc+=x[k*4+i]*y[k*4+j];
   // Signed INT8 MMA yields acc. Reconstruct unsigned byte dot exactly:
   // (x+128)*(y+128) = xy+128*x+128*y+16384.
   const corrected=acc+128*sumX[i]+128*sumY[j]+16384*K;
   if(corrected<0||corrected>255*255*K)throw Error("byte correction out of range");
   total+=BigInt(corrected)<<BigInt(8*(i+j));
 }
 return total;
}
for(let trial=0;trial<10;trial++){
 for(let limb=0;limb<LIMBS;limb++){
  const a=new Uint32Array(K),b=new Uint32Array(K);
  for(let k=0;k<K;k++){
   const mode=trial%5;
   a[k]=mode===0?0:mode===1?0xffffffff:mode===2?0x80808080:mode===3?((k&1)?0:0xffffffff):rand();
   b[k]=mode===0?0xffffffff:mode===1?0xffffffff:mode===2?0x7f7f7f7f:mode===3?((k&1)?0xffffffff:0):rand();
  }
  const orig=dotOriginal(a,b), alt=dotInt8(a,b);
  if(orig!==alt)throw Error("Mismatch trial "+trial+" limb "+limb+": "+orig+" != "+alt);
 }
 console.log("PASS exact 32-bit CRT column reconstruction, trial "+trial+" / 10, "+LIMBS+" limbs");
}
console.log("ALL PASS: "+(10*LIMBS)+" independent 118-term wide dot products; 16 signed INT8 partial products each.");
console.log("This is arithmetic proof-of-concept only: NO GPU correctness or speed measured.");
