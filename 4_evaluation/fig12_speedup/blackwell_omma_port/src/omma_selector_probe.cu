#include <cstdint>
template<int BA,int TA,int BB,int TB>
__global__ void k(float* o, const uint32_t* in, uint16_t rb, uint16_t rt) {
  uint32_t a[4]={in[0],in[1],in[2],in[3]}, b[2]={in[4],in[5]}, fa=in[6], fb=in[7];
  float d[4]={0,0,0,0};
  const uint16_t ba=BA, ta=TA, bb=BB, tb=TB;
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(fa),"h"(ba),"h"(ta),"r"(fb),"h"(bb),"h"(tb));
  o[threadIdx.x]=d[0]+d[1]+d[2]+d[3];
}
template __global__ void k<0,0,0,0>(float*,const uint32_t*,uint16_t,uint16_t);
template __global__ void k<0,1,0,0>(float*,const uint32_t*,uint16_t,uint16_t);
template __global__ void k<0,0,0,2>(float*,const uint32_t*,uint16_t,uint16_t);
template __global__ void k<0,3,0,3>(float*,const uint32_t*,uint16_t,uint16_t);
// mxf8f6f4-style kind with 1X scale vectors for comparison of operand form
__global__ void k_mx(float* o, const uint32_t* in) {
  uint32_t a[4]={in[0],in[1],in[2],in[3]}, b[2]={in[4],in[5]}, fa=in[6], fb=in[7];
  float d[4]={0,0,0,0}; const uint16_t z=0;
  asm volatile("mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(fa),"h"(z),"h"(z),"r"(fb),"h"(z),"h"(z));
  o[threadIdx.x]=d[0]+d[1]+d[2]+d[3];
}
