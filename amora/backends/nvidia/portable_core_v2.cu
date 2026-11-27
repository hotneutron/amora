// SPDX-License-Identifier: Apache-2.0
// Explicit v2 CUDA node realization. No vendor BLAS or CPU arithmetic in device execution.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef AMORA_OP_MASK
#define AMORA_OP_MASK 2147483647ULL
#endif
constexpr uint64_t kMask = AMORA_OP_MASK;
enum Op {CAST,ADD,MUL,RELU,ROLL,SCALE,STAT,ROTARY,LSE,MAXSCAN,BILINEAR,COUNT,GELU,
         AFFINE,EMBED,SCATTER,SEGMENT,SILU,ROUND,SOFTMAX,VARIANCE,TOPK,WMEAN,MHASH,
         GMIN,GL2,WSUM,EMEAN,DHASH,GMAX,STENCIL};

__device__ float getf(const void *p, int type, int index) {
  return type == 0 ? __half2float(static_cast<const __half *>(p)[index])
                   : static_cast<const float *>(p)[index];
}
__device__ void putf(void *p, int type, int index, float value) {
  if (type == 0) static_cast<__half *>(p)[index] = __float2half_rn(value);
  else static_cast<float *>(p)[index] = value;
}
__device__ float reduce32(float value, float *scratch, bool maximum) {
  int lane = threadIdx.x;
  scratch[lane] = value;
  __syncthreads();
  for (int step=16; step; step/=2) {
    if (lane < step)
      scratch[lane] = maximum ? fmaxf(scratch[lane],scratch[lane+step])
                             : scratch[lane]+scratch[lane+step];
    __syncthreads();
  }
  return scratch[0];
}

template<int O>
__global__ void node_kernel(const void *x, const void *y, void *out, void *out1,
    int xt,int yt,int ot,int xm,int xn,int om,int on,int xs,int ys,int flags,int k) {
  if constexpr (O==STAT || O==LSE || O==SOFTMAX || O==VARIANCE) {
    __shared__ float scratch[32];
    int r=blockIdx.x,lane=threadIdx.x;
    bool maximum = O==LSE || O==SOFTMAX || (O==STAT && (flags&1));
    float v=maximum ? -3.402823466e38F : 0.0f;
    for (int c=lane;c<xn;c+=32)
      if (O!=SOFTMAX || c<=r) {
        float a=getf(x,xt,r*xs+c);
        v=maximum?fmaxf(v,a):v+a;
      }
    float core=reduce32(v,scratch,maximum);
    __syncthreads();
    if constexpr (O==STAT) {
      if (!(flags&1)) core/=xn;
      if (flags&2) core=__half2float(__float2half_rn(core));
      for(int c=lane;c<on;c+=32)putf(out,ot,r*on+c,core);
    } else if constexpr (O==VARIANCE) {
      float mean=core/xn, sum=0.0f;
      for(int c=lane;c<xn;c+=32){float a=getf(x,xt,r*xs+c)-mean;sum+=a*a;}
      float variance=reduce32(sum,scratch,false)/xn;
      for(int c=lane;c<on;c+=32)putf(out,ot,r*on+c,(getf(x,xt,r*xs+c)-mean)/sqrtf(variance+1e-5f));
    } else {
      float sum=0;
      for(int c=lane;c<xn;c+=32)
        if(O!=SOFTMAX || c<=r)sum+=__expf(getf(x,xt,r*xs+c)-core);
      sum=reduce32(sum,scratch,false);
      for(int c=lane;c<on;c+=32)
        putf(out,ot,r*on+c,O==LSE ? core+__logf(sum)
                         : c<=r ? __expf(getf(x,xt,r*xs+c)-core)/sum : 0.0f);
    }
    return;
  }
  if constexpr (O==MAXSCAN || O==AFFINE || O==TOPK) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=om)return;
    if constexpr(O==TOPK){
      int chosen[8];
      for(int j=0;j<k;j++){
        float best=-3.402823466e38F;int index=-1;
        for(int c=0;c<xn;c++){
          bool used=false;for(int t=0;t<j;t++)if(chosen[t]==c)used=true;
          float value=getf(x,xt,r*xs+c);
          if(!used && (index<0 || value>best)){best=value;index=c;}
        }
        chosen[j]=index;putf(out,ot,r*on+j,best);static_cast<int *>(out1)[r*on+j]=index;
      }
    }else{
      float carry=O==MAXSCAN?-3.402823466e38F:0.0f;
      for(int c=0;c<xn;c++){
        float value=getf(x,xt,r*xs+c);
        if constexpr(O==MAXSCAN)carry=fmaxf(carry,value);
        else carry=value*carry+getf(y,yt,r*ys+c);
        putf(out,ot,r*on+c,carry);
      }
    }
    return;
  }
  int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=om*on)return;
  int r=index/on,c=index%on;
  float value=0.0f;
  if constexpr(O==EMBED){
    for(int j=0;j<4;j++){
      int source=static_cast<const int *>(y)[r*ys+j];
      value+=getf(x,xt,source*xs+c);
    }
  }else if constexpr(O==SCATTER){
    for(int source=0;source<xm;source++)
      if(static_cast<const int *>(y)[source*ys]==r)value+=getf(x,xt,source*xs+c);
  }else if constexpr(O==SEGMENT || O==COUNT){
    if(O==SEGMENT || c<xn/4)
      for(int j=0;j<4;j++){
        float a=getf(x,xt,r*xs+c*4+j);
        value+=O==SEGMENT?a:float(a>0);
      }
  }else if constexpr(O>=WMEAN && O<=GMAX){
    if(r<xm/4){
      float sum=0,den=0,low=3.402823466e38F,high=-3.402823466e38F;
      for(int j=0;j<4;j++){
        bool alt=(O==MHASH || O==DHASH) && j>=2;
        int source=((r*4+j)*(alt?17:13)+c*(alt?5:7))%xm;
        float a=getf(O==MHASH && j>=2?y:x,O==MHASH && j>=2?yt:xt,source*(O==MHASH && j>=2?ys:xs)+c);
        if constexpr(O==WMEAN || O==WSUM){
          float weight=getf(y,yt,source*ys+c);
          if(O==WMEAN)weight=fabsf(weight)+0.25f;
          sum+=a*weight;den+=weight;
        }else if constexpr(O==GL2)sum+=a*a;
        else sum+=a;
        low=fminf(low,a);high=fmaxf(high,a);
      }
      if(O==WMEAN)value=sum/den;
      else if(O==EMEAN)value=sum*0.25f;
      else if(O==GMIN)value=low;
      else if(O==GMAX)value=high;
      else if(O==GL2)value=sqrtf(sum+1e-6f);
      else value=sum;
    }
  }else{
    float v=getf(x,xt,r*xs+c);
    if constexpr(O==CAST)value=v;
    else if constexpr(O==ADD)value=v+getf(y,yt,r*ys+c);
    else if constexpr(O==MUL)value=v*getf(y,yt,r*ys+c);
    else if constexpr(O==RELU)value=fmaxf(v,0.0f);
    else if constexpr(O==SCALE)value=v*0.5f; // exact identity-matrix fixture simplification
    else if constexpr(O==ROLL)value=getf(x,xt,((r+xm-1)%xm)*xs+c);
    else if constexpr(O==ROTARY){
      float mate=getf(x,xt,r*xs+(c^1)),angle=getf(y,yt,r*ys+c);
      value=(c&1)?v*__sinf(angle)+mate*__cosf(angle):v*__cosf(angle)-mate*__sinf(angle);
    }else if constexpr(O==BILINEAR){
      int down=min(r+1,xm-1),right=min(c+1,xn-1);
      value=0.375f*v+0.125f*getf(x,xt,r*xs+right)+0.375f*getf(x,xt,down*xs+c)
            +0.125f*getf(x,xt,down*xs+right);
    }else if constexpr(O==STENCIL){
      value=0.5f*v+0.125f*(getf(x,xt,max(r-1,0)*xs+c)+getf(x,xt,min(r+1,xm-1)*xs+c)
                      +getf(x,xt,r*xs+max(c-1,0))+getf(x,xt,r*xs+min(c+1,xn-1)));
    }else if constexpr(O==ROUND || O==GELU){
      float z=floorf(fmaxf(-127.0f,fminf(127.0f,v*32.0f))+0.5f)/32.0f;
      if constexpr(O==ROUND)value=z;
      else{z+=getf(y,yt,r*ys+c);value=z/(1.0f+__expf(-2.0f*0.7978845608028654f*z*(1.0f+0.044715f*z*z)));}
    }else if constexpr(O==SILU)value=v*getf(y,yt,r*ys+c)/(1.0f+__expf(-v));
  }
  putf(out,ot,index,value);
}

struct Value {int dtype,m,n,stride,offset,count,exact;std::string hash;void *storage=nullptr;};
struct Node {int op,x,y,out,out1,flags,k;};
struct Case {std::string id;int inputs;std::vector<Value> values;std::vector<Node> nodes;std::vector<int> outputs;};
int bytes(int dtype){return dtype==0?2:4;}
void check_cuda(cudaError_t result,const char *where){
  if(result!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(result));
}
void *pointer(Value &v){return static_cast<char *>(v.storage)+v.offset*bytes(v.dtype);}
template<int O> void launch(Case &c,Node &node){
  auto &x=c.values[node.x],&out=c.values[node.out];
  Value &y=node.y>=0?c.values[node.y]:x;
  int block=(O==STAT || O==LSE || O==SOFTMAX || O==VARIANCE ||
             O==MAXSCAN || O==AFFINE || O==TOPK)?32:128;
  int grid=(O==STAT || O==LSE || O==SOFTMAX || O==VARIANCE)?out.m:
           (O==MAXSCAN || O==AFFINE || O==TOPK)?(out.m+block-1)/block:(out.m*out.n+block-1)/block;
  node_kernel<O><<<grid,block>>>(pointer(x),node.y>=0?pointer(y):nullptr,pointer(out),
    node.out1>=0?pointer(c.values[node.out1]):nullptr,x.dtype,y.dtype,out.dtype,
    x.m,x.n,out.m,out.n,x.stride,y.stride,node.flags,node.k);
}
void execute(Case &c){
  for(auto &node:c.nodes){
#define CHOICE(I) case I: if constexpr(kMask & (1ULL<<I))launch<I>(c,node);else throw std::runtime_error("operator outside compiled mask");break
    switch(node.op){
      CHOICE(0);CHOICE(1);CHOICE(2);CHOICE(3);CHOICE(4);CHOICE(5);CHOICE(6);CHOICE(7);
      CHOICE(8);CHOICE(9);CHOICE(10);CHOICE(11);CHOICE(12);CHOICE(13);CHOICE(14);CHOICE(15);
      CHOICE(16);CHOICE(17);CHOICE(18);CHOICE(19);CHOICE(20);CHOICE(21);CHOICE(22);CHOICE(23);
      CHOICE(24);CHOICE(25);CHOICE(26);CHOICE(27);CHOICE(28);CHOICE(29);CHOICE(30);
      default:throw std::runtime_error("unknown operator");
    }
#undef CHOICE
    check_cuda(cudaGetLastError(),"kernel launch");
  }
}
std::vector<unsigned char> load_bank(const std::string &root,const Value &v){
  std::ifstream in(root+"/banks/"+v.hash+".bin",std::ios::binary);
  std::vector<unsigned char> data(size_t(v.count)*bytes(v.dtype));
  if(!in.read(reinterpret_cast<char *>(data.data()),data.size()))throw std::runtime_error("bank read failed");
  return data;
}
float host_float(const unsigned char *p,int dtype,int index){
  return dtype==0?__half2float(reinterpret_cast<const __half *>(p)[index])
         :dtype==1?reinterpret_cast<const float *>(p)[index]:float(reinterpret_cast<const int *>(p)[index]);
}
struct Comparison {bool passed=true;double maximum=0;int mismatches=0;int first=-1;};
Comparison compare(Case &c,const std::string &root){
  Comparison result;
  for(int i=0;i<int(c.values.size());i++){
    auto &v=c.values[i];auto want=load_bank(root,v);
    std::vector<unsigned char> got(want.size());
    check_cuda(cudaMemcpy(got.data(),v.storage,got.size(),cudaMemcpyDeviceToHost),"result download");
    if(i<c.inputs){
      if(got!=want){result.passed=false;result.mismatches++;if(result.first<0)result.first=i;}
      continue;
    }
    double atol=v.dtype==0?0.001:0.00002,rtol=v.dtype==0?0.005:0.0002;
    for(int j=0;j<v.count;j++){
      double a=host_float(got.data(),v.dtype,j),b=host_float(want.data(),v.dtype,j);
      double delta=std::abs(a-b);if(std::isfinite(delta))result.maximum=std::max(result.maximum,delta);
      bool okay=std::isfinite(a)&&std::isfinite(b)&&(v.exact?a==b:delta<=atol+rtol*std::abs(b));
      if(!okay){result.passed=false;result.mismatches++;if(result.first<0)result.first=i;}
    }
  }
  return result;
}
std::vector<Case> read_cases(const std::string &root){
  std::ifstream in(root+"/cases.txt");int count;
  if(!(in>>count)||count<=0)throw std::runtime_error("invalid case header");
  std::vector<Case> cases(count);
  for(auto &c:cases){
    int nv,nn,no;in>>c.id>>nv>>c.inputs>>nn>>no;
    if(nv<=0 || c.inputs<=0 || nn<=0)throw std::runtime_error("invalid graph");
    c.values.resize(nv);c.nodes.resize(nn);c.outputs.resize(no);
    for(auto &v:c.values)in>>v.dtype>>v.m>>v.n>>v.stride>>v.offset>>v.count>>v.exact>>v.hash;
    for(auto &n:c.nodes)in>>n.op>>n.x>>n.y>>n.out>>n.out1>>n.flags>>n.k;
    for(auto &o:c.outputs)in>>o;
    if(!in)throw std::runtime_error("invalid case payload");
  }
  return cases;
}
int main(int argc,char **argv){
  try{
    std::string root,outfile,selection;int start=0,stop=-1,device=0,samples=0,warmup=3;
    for(int i=1;i<argc;i++){
      std::string a=argv[i];auto pos=a.find('=');if(pos==std::string::npos)throw std::runtime_error("expected --name=value");
      std::string name=a.substr(0,pos),value=a.substr(pos+1);
      if(name=="--corpus")root=value;else if(name=="--result")outfile=value;
      else if(name=="--start")start=std::stoi(value);else if(name=="--stop")stop=std::stoi(value);
      else if(name=="--selection")selection=value;else if(name=="--device")device=std::stoi(value);
      else if(name=="--samples")samples=std::stoi(value);else if(name=="--warmup")warmup=std::stoi(value);
      else throw std::runtime_error("unknown option "+name);
    }
    auto cases=read_cases(root);std::vector<int> chosen;
    if(!selection.empty()){std::ifstream f(selection);int index;while(f>>index)chosen.push_back(index);}
    else{if(stop<0)stop=cases.size();for(int i=start;i<stop;i++)chosen.push_back(i);}
    if(chosen.empty() || samples<0 || warmup<0)throw std::runtime_error("empty or invalid selection");
    if(std::ifstream(outfile).good())throw std::runtime_error("refuse result overwrite");
    std::ofstream results(outfile);if(!results)throw std::runtime_error("result file open failed");
    check_cuda(cudaSetDevice(device),"device selection");cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop,device),"device properties");
    std::ostringstream uuid;uuid<<"GPU-";
    for(int i=0;i<16;i++){if(i==4||i==6||i==8||i==10)uuid<<"-";uuid<<std::hex<<std::setw(2)<<std::setfill('0')<<(int(prop.uuid.bytes[i])&255);}
    std::cout<<"AMORA_V2_DEVICE {\"name\":\""<<prop.name<<"\",\"uuid\":\""<<uuid.str()
             <<"\",\"logical_device\":"<<device<<",\"sm\":"<<prop.major<<",\"sms\":"<<prop.multiProcessorCount
             <<",\"operator_mask\":"<<kMask<<"}"<<std::endl;
    int failures=0;
    for(int index:chosen){
      if(index<0 || index>=int(cases.size()))throw std::runtime_error("case index outside manifest");
      auto &c=cases[index];
      for(auto &v:c.values){
        check_cuda(cudaMalloc(&v.storage,size_t(v.count)*bytes(v.dtype)),"buffer allocation");
        if(&v-c.values.data()<c.inputs){auto data=load_bank(root,v);check_cuda(cudaMemcpy(v.storage,data.data(),data.size(),cudaMemcpyHostToDevice),"input upload");}
      }
      cudaEvent_t begin,end;check_cuda(cudaEventCreate(&begin),"event create");check_cuda(cudaEventCreate(&end),"event create");
      float first_ms=0;
      check_cuda(cudaEventRecord(begin),"event begin");execute(c);check_cuda(cudaEventRecord(end),"event end");
      check_cuda(cudaEventSynchronize(end),"program completion");check_cuda(cudaEventElapsedTime(&first_ms,begin,end),"event span");
      auto before=compare(c,root);std::vector<float> times;
      if(before.passed){
        if(samples==0)times.push_back(first_ms);
        else{
          for(int w=0;w<warmup;w++)execute(c);
          check_cuda(cudaDeviceSynchronize(),"warmup completion");
          for(int s=0;s<samples;s++){
            float ms;check_cuda(cudaEventRecord(begin),"event begin");execute(c);check_cuda(cudaEventRecord(end),"event end");
            check_cuda(cudaEventSynchronize(end),"sample completion");check_cuda(cudaEventElapsedTime(&ms,begin,end),"event span");times.push_back(ms);
          }
        }
      }
      auto after=compare(c,root);bool pass=before.passed&&after.passed;
      if(!pass){failures++;times.clear();}
      std::ostringstream row;row<<std::setprecision(9)<<"{\"index\":"<<index<<",\"case_id\":\""<<c.id
        <<"\",\"passed\":"<<(pass?"true":"false")<<",\"node_ports_checked\":"<<c.values.size()-c.inputs
        <<",\"max_abs_error\":"<<std::max(before.maximum,after.maximum)
        <<",\"mismatches\":"<<before.mismatches+after.mismatches
        <<",\"first_failed_value\":"<<(before.first>=0?before.first:after.first)<<",\"samples_ms\":[";
      for(size_t j=0;j<times.size();j++){if(j)row<<",";row<<times[j];}
      row<<"]}";results<<row.str()<<std::endl;std::cout<<"AMORA_V2_RESULT "<<row.str()<<std::endl;
      check_cuda(cudaEventDestroy(begin),"event destroy");check_cuda(cudaEventDestroy(end),"event destroy");
      for(auto &v:c.values)check_cuda(cudaFree(v.storage),"buffer free");
    }
    return failures?1:0;
  }catch(const std::exception &error){std::cerr<<"AMORA_V2_ERROR "<<error.what()<<std::endl;return 2;}
}

