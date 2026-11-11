/*
 * reduction_bench.cu
 * ==================
 * Single-executable benchmark for SUM, MAX, and ARGMAX reductions using
 * (a) shared-memory tree reduction and (b) warp-shuffle reduction.
 *
 * Command-line interface
 * ----------------------
 *   reduction_bench
 *     --operation   sum | max | argmax
 *     --algorithm   shared | shuffle | both
 *     --size        <N>          (default 1000000)
 *     --seed        <S>          (default 42)
 *     --repetitions <R>          (default 5)
 *     --warmup      <W>          (default 50)
 *     --pattern     random_signed|positive|all_equal|dup_max|max_at_end
 *     --output-format json       (always json)
 *
 * Stdout: JSON result document.
 * Stderr: CUDA diagnostics and debug messages.
 * Exit:   0 on success, 1 on error.
 */

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <climits>
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// Constants
// ============================================================

static const int THREADS        = 256;
static const int WARPS_PER_BLK  = THREADS / 32;   // 8
static const int MAX_BLOCKS     = 1024;

// CUDA error check: print to stderr and exit on failure.
#define CUDA_CHECK(call) do {                                        \
    cudaError_t _err = (call);                                       \
    if (_err != cudaSuccess) {                                       \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(_err) << "\n";      \
        std::exit(1);                                                 \
    }                                                                \
} while (0)

// ============================================================
// ARGMAX pair type
// ============================================================

struct Pair {
    float val;
    int   idx;
};

// Merge rule: higher value wins; on tie, lower index wins (first-occurrence).
__device__ __host__
inline Pair best(Pair a, Pair b) {
    if (a.val > b.val) return a;
    if (b.val > a.val) return b;
    return (a.idx < b.idx) ? a : b;
}

// ============================================================
// SHARED-MEMORY KERNELS
// ============================================================

__global__ void kmSharedSum(const float* in, float* part, int n) {
    __shared__ float s[THREADS];
    int tid = threadIdx.x, i = blockIdx.x * blockDim.x + tid;
    int step = gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (; i < n; i += step) acc += in[i];
    s[tid] = acc; __syncthreads();
    for (int str = blockDim.x / 2; str > 0; str >>= 1) {
        if (tid < str) s[tid] += s[tid + str]; __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = s[0];
}

__global__ void kmFinalSum(const float* part, float* out, int cnt) {
    __shared__ float s[THREADS];
    int tid = threadIdx.x; float acc = 0.0f;
    for (int i = tid; i < cnt; i += blockDim.x) acc += part[i];
    s[tid] = acc; __syncthreads();
    for (int str = blockDim.x / 2; str > 0; str >>= 1) {
        if (tid < str) s[tid] += s[tid + str]; __syncthreads();
    }
    if (tid == 0) out[0] = s[0];
}

__global__ void kmSharedMax(const float* in, float* part, int n) {
    __shared__ float s[THREADS];
    int tid = threadIdx.x, i = blockIdx.x * blockDim.x + tid;
    int step = gridDim.x * blockDim.x;
    float acc = -FLT_MAX;
    for (; i < n; i += step) acc = fmaxf(acc, in[i]);
    s[tid] = acc; __syncthreads();
    for (int str = blockDim.x / 2; str > 0; str >>= 1) {
        if (tid < str) s[tid] = fmaxf(s[tid], s[tid + str]); __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = s[0];
}

__global__ void kmFinalMax(const float* part, float* out, int cnt) {
    __shared__ float s[THREADS];
    int tid = threadIdx.x; float acc = -FLT_MAX;
    for (int i = tid; i < cnt; i += blockDim.x) acc = fmaxf(acc, part[i]);
    s[tid] = acc; __syncthreads();
    for (int str = blockDim.x / 2; str > 0; str >>= 1) {
        if (tid < str) s[tid] = fmaxf(s[tid], s[tid + str]); __syncthreads();
    }
    if (tid == 0) out[0] = s[0];
}

__global__ void kmSharedArgmax(const float* in, Pair* part, int n) {
    __shared__ Pair s[THREADS];
    int tid = threadIdx.x, i = blockIdx.x * blockDim.x + tid;
    int step = gridDim.x * blockDim.x;
    Pair acc = { -FLT_MAX, INT_MAX };
    for (; i < n; i += step) { Pair c = { in[i], i }; acc = best(acc, c); }
    s[tid] = acc; __syncthreads();
    for (int str = blockDim.x / 2; str > 0; str >>= 1) {
        if (tid < str) s[tid] = best(s[tid], s[tid + str]); __syncthreads();
    }
    if (tid == 0) part[blockIdx.x] = s[0];
}

__global__ void kmFinalArgmax(const Pair* part, Pair* out, int cnt) {
    __shared__ Pair s[THREADS];
    int tid = threadIdx.x; Pair acc = { -FLT_MAX, INT_MAX };
    for (int i = tid; i < cnt; i += blockDim.x) acc = best(acc, part[i]);
    s[tid] = acc; __syncthreads();
    for (int str = blockDim.x / 2; str > 0; str >>= 1) {
        if (tid < str) s[tid] = best(s[tid], s[tid + str]); __syncthreads();
    }
    if (tid == 0) out[0] = s[0];
}

// ============================================================
// WARP-SHUFFLE KERNELS
// ============================================================

__global__ void kmShuffleSum(const float* in, float* part, int n) {
    __shared__ float ws[WARPS_PER_BLK];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int i = blockIdx.x * blockDim.x + tid, step = gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (; i < n; i += step) acc += in[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, off);
    if (lane == 0) ws[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < WARPS_PER_BLK) ? ws[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xFFFFFFFF, t, off);
        if (lane == 0) part[blockIdx.x] = t;
    }
}

__global__ void kmShuffleMax(const float* in, float* part, int n) {
    __shared__ float ws[WARPS_PER_BLK];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int i = blockIdx.x * blockDim.x + tid, step = gridDim.x * blockDim.x;
    float acc = -FLT_MAX;
    for (; i < n; i += step) acc = fmaxf(acc, in[i]);
    for (int off = 16; off > 0; off >>= 1) acc = fmaxf(acc, __shfl_down_sync(0xFFFFFFFF, acc, off));
    if (lane == 0) ws[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < WARPS_PER_BLK) ? ws[lane] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1) t = fmaxf(t, __shfl_down_sync(0xFFFFFFFF, t, off));
        if (lane == 0) part[blockIdx.x] = t;
    }
}

__global__ void kmShuffleArgmax(const float* in, Pair* part, int n) {
    __shared__ Pair ws[WARPS_PER_BLK];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int i = blockIdx.x * blockDim.x + tid, step = gridDim.x * blockDim.x;
    Pair acc = { -FLT_MAX, INT_MAX };
    for (; i < n; i += step) { Pair c = { in[i], i }; acc = best(acc, c); }
    for (int off = 16; off > 0; off >>= 1) {
        float pv = __shfl_down_sync(0xFFFFFFFF, acc.val, off);
        int   pi = __shfl_down_sync(0xFFFFFFFF, acc.idx, off);
        acc = best(acc, { pv, pi });
    }
    if (lane == 0) ws[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        Pair t = (lane < WARPS_PER_BLK) ? ws[lane] : Pair{ -FLT_MAX, INT_MAX };
        for (int off = 16; off > 0; off >>= 1) {
            float pv = __shfl_down_sync(0xFFFFFFFF, t.val, off);
            int   pi = __shfl_down_sync(0xFFFFFFFF, t.idx, off);
            t = best(t, { pv, pi });
        }
        if (lane == 0) part[blockIdx.x] = t;
    }
}

// ============================================================
// CPU REFERENCES
// ============================================================

static double cpuSum(const std::vector<float>& v) {
    double s = 0; for (float x : v) s += x; return s;
}
static float cpuMax(const std::vector<float>& v) {
    float m = -FLT_MAX; for (float x : v) if (x > m) m = x; return m;
}
static Pair cpuArgmax(const std::vector<float>& v) {
    Pair b = { -FLT_MAX, -1 };
    for (int i = 0; i < (int)v.size(); i++) if (v[i] > b.val) { b.val = v[i]; b.idx = i; }
    return b;
}

// ============================================================
// INPUT GENERATION
// ============================================================

static std::vector<float> genInput(int n, int seed, const std::string& pat) {
    std::vector<float> v(n);
    std::srand(seed);
    if (pat == "all_equal") { for (int i=0;i<n;i++) v[i]=7.0f; }
    else if (pat == "positive") { for (int i=0;i<n;i++) v[i]=(float)(std::rand()%10000+1); }
    else if (pat == "max_at_end") { for (int i=0;i<n;i++) v[i]=(float)(std::rand()%10000); v[n-1]=99999.0f; }
    else if (pat == "dup_max") {
        for (int i=0;i<n;i++) v[i]=(float)(std::rand()%9000);
        if (n>=2) { v[0]=9999.0f; v[n/2]=9999.0f; }
    } else {  // random_signed
        for (int i=0;i<n;i++) v[i]=(float)(std::rand()%20001-10000);
    }
    return v;
}

// ============================================================
// RESULT STRUCT AND TIMING HELPERS
// ============================================================

struct AlgResult {
    float value = 0; int argmax_index = -1; bool correct = false;
    float median_us = 0; std::vector<float> samples;
};

static float medianOf(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    int m = (int)v.size();
    if (!m) return 0;
    return (m&1) ? v[m/2] : (v[m/2-1]+v[m/2])*0.5f;
}

// ============================================================
// PER-ALGORITHM RUN FUNCTIONS
// ============================================================

template<typename LaunchFn>
static std::vector<float> timeKernels(LaunchFn fn, int warmup, int reps) {
    for (int i=0;i<warmup;i++) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    std::vector<float> samples;
    for (int i=0;i<reps;i++) {
        CUDA_CHECK(cudaEventRecord(s)); fn(); CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,s,e)); samples.push_back(ms*1000.0f);
    }
    CUDA_CHECK(cudaEventDestroy(s)); CUDA_CHECK(cudaEventDestroy(e));
    return samples;
}

static AlgResult runSharedSum(const float* d_in,float* d_fp,float* d_fo,int n,int blk,int w,int r,float ref){
    AlgResult res;
    auto fn=[&](){ kmSharedSum<<<blk,THREADS>>>(d_in,d_fp,n); kmFinalSum<<<1,THREADS>>>(d_fp,d_fo,blk); };
    res.samples=timeKernels(fn,w,r); res.median_us=medianOf(res.samples);
    fn(); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&res.value,d_fo,sizeof(float),cudaMemcpyDeviceToHost));
    res.correct=(std::fabs(res.value-ref)<std::fabs(ref)*1e-3f+1.0f); return res;
}
static AlgResult runShuffleSum(const float* d_in,float* d_fp,float* d_fo,int n,int blk,int w,int r,float ref){
    AlgResult res;
    auto fn=[&](){ kmShuffleSum<<<blk,THREADS>>>(d_in,d_fp,n); kmFinalSum<<<1,THREADS>>>(d_fp,d_fo,blk); };
    res.samples=timeKernels(fn,w,r); res.median_us=medianOf(res.samples);
    fn(); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&res.value,d_fo,sizeof(float),cudaMemcpyDeviceToHost));
    res.correct=(std::fabs(res.value-ref)<std::fabs(ref)*1e-3f+1.0f); return res;
}
static AlgResult runSharedMax(const float* d_in,float* d_fp,float* d_fo,int n,int blk,int w,int r,float ref){
    AlgResult res;
    auto fn=[&](){ kmSharedMax<<<blk,THREADS>>>(d_in,d_fp,n); kmFinalMax<<<1,THREADS>>>(d_fp,d_fo,blk); };
    res.samples=timeKernels(fn,w,r); res.median_us=medianOf(res.samples);
    fn(); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&res.value,d_fo,sizeof(float),cudaMemcpyDeviceToHost));
    res.correct=(std::fabs(res.value-ref)<1e-3f); return res;
}
static AlgResult runShuffleMax(const float* d_in,float* d_fp,float* d_fo,int n,int blk,int w,int r,float ref){
    AlgResult res;
    auto fn=[&](){ kmShuffleMax<<<blk,THREADS>>>(d_in,d_fp,n); kmFinalMax<<<1,THREADS>>>(d_fp,d_fo,blk); };
    res.samples=timeKernels(fn,w,r); res.median_us=medianOf(res.samples);
    fn(); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&res.value,d_fo,sizeof(float),cudaMemcpyDeviceToHost));
    res.correct=(std::fabs(res.value-ref)<1e-3f); return res;
}
static AlgResult runSharedArgmax(const float* d_in,Pair* d_pp,Pair* d_po,int n,int blk,int w,int r,Pair ref){
    AlgResult res;
    auto fn=[&](){ kmSharedArgmax<<<blk,THREADS>>>(d_in,d_pp,n); kmFinalArgmax<<<1,THREADS>>>(d_pp,d_po,blk); };
    res.samples=timeKernels(fn,w,r); res.median_us=medianOf(res.samples);
    fn(); CUDA_CHECK(cudaDeviceSynchronize());
    Pair p; CUDA_CHECK(cudaMemcpy(&p,d_po,sizeof(Pair),cudaMemcpyDeviceToHost));
    res.value=p.val; res.argmax_index=p.idx;
    res.correct=(std::fabs(p.val-ref.val)<1e-3f)&&(p.idx==ref.idx); return res;
}
static AlgResult runShuffleArgmax(const float* d_in,Pair* d_pp,Pair* d_po,int n,int blk,int w,int r,Pair ref){
    AlgResult res;
    auto fn=[&](){ kmShuffleArgmax<<<blk,THREADS>>>(d_in,d_pp,n); kmFinalArgmax<<<1,THREADS>>>(d_pp,d_po,blk); };
    res.samples=timeKernels(fn,w,r); res.median_us=medianOf(res.samples);
    fn(); CUDA_CHECK(cudaDeviceSynchronize());
    Pair p; CUDA_CHECK(cudaMemcpy(&p,d_po,sizeof(Pair),cudaMemcpyDeviceToHost));
    res.value=p.val; res.argmax_index=p.idx;
    res.correct=(std::fabs(p.val-ref.val)<1e-3f)&&(p.idx==ref.idx); return res;
}

// ============================================================
// JSON HELPERS
// ============================================================

static std::string esc(const std::string& s){
    std::string o; for(char c:s){if(c=='"')o+="\\\"";else if(c=='\\')o+="\\\\";else o+=c;} return o;
}
static std::string ff(double v,int p=4){char b[64];std::snprintf(b,sizeof(b),"%.*f",p,v);return b;}

static std::string toJson(const AlgResult& r,bool isAm){
    std::ostringstream o;
    o<<"{\"value\":"<<ff(r.value)<<",";
    if(isAm) o<<"\"argmax_index\":"<<r.argmax_index<<",";
    else     o<<"\"argmax_index\":null,";
    o<<"\"correct\":"<<(r.correct?"true":"false")<<",";
    o<<"\"latency_us\":"<<ff(r.median_us)<<",";
    o<<"\"samples_us\":[";
    for(int i=0;i<(int)r.samples.size();i++){if(i)o<<","; o<<ff(r.samples[i]);}
    o<<"]}"; return o.str();
}

// ============================================================
// ARGUMENT PARSING
// ============================================================

struct Args {
    std::string op="sum",alg="both",pat="random_signed";
    int n=1000000,seed=42,reps=5,warmup=50;
};

static Args parseArgs(int argc,char** argv){
    Args a;
    for(int i=1;i<argc;i+=2){
        if(i+1>=argc){ std::cerr<<"Missing value for "<<argv[i]<<"\n"; std::exit(1); }
        std::string k=argv[i],v=argv[i+1];
        if(k=="--operation")a.op=v;
        else if(k=="--algorithm")a.alg=v;
        else if(k=="--size")a.n=std::atoi(v.c_str());
        else if(k=="--seed")a.seed=std::atoi(v.c_str());
        else if(k=="--repetitions")a.reps=std::atoi(v.c_str());
        else if(k=="--warmup")a.warmup=std::atoi(v.c_str());
        else if(k=="--pattern")a.pat=v;
        else if(k=="--output-format"){/*always json*/}
        else{std::cerr<<"Unknown arg: "<<k<<"\n";std::exit(1);}
    }
    if(a.n<1){std::cout<<"{\"status\":\"error\",\"message\":\"size must be >= 1\"}\n";std::exit(1);}
    if(a.reps<1){std::cout<<"{\"status\":\"error\",\"message\":\"repetitions must be >= 1\"}\n";std::exit(1);}
    return a;
}

// ============================================================
// MAIN
// ============================================================

int main(int argc,char** argv){
    Args args=parseArgs(argc,argv);

    // GPU info
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::string gpuName=prop.name;

    // Input
    int n=args.n;
    std::vector<float> h=genInput(n,args.seed,args.pat);
    int blocks=std::min((n+THREADS-1)/THREADS,MAX_BLOCKS);

    // CPU references
    double cSumRef=(args.op=="sum")?cpuSum(h):0.0;
    float  cMaxRef=(args.op=="max")?cpuMax(h):-FLT_MAX;
    Pair   cAmRef =(args.op=="argmax")?cpuArgmax(h):Pair{-FLT_MAX,-1};

    // GPU alloc
    float *d_in=nullptr,*d_fp=nullptr,*d_fo=nullptr;
    Pair  *d_pp=nullptr,*d_po=nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fp,blocks*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fo,sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pp,blocks*sizeof(Pair)));
    CUDA_CHECK(cudaMalloc(&d_po,sizeof(Pair)));
    CUDA_CHECK(cudaMemcpy(d_in,h.data(),n*sizeof(float),cudaMemcpyHostToDevice));

    bool doSh=(args.alg=="shared"||args.alg=="both");
    bool doSf=(args.alg=="shuffle"||args.alg=="both");
    bool isAm=(args.op=="argmax");

    AlgResult shRes,sfRes;
    if(args.op=="sum"){
        if(doSh) shRes=runSharedSum (d_in,d_fp,d_fo,n,blocks,args.warmup,args.reps,(float)cSumRef);
        if(doSf) sfRes=runShuffleSum(d_in,d_fp,d_fo,n,blocks,args.warmup,args.reps,(float)cSumRef);
    } else if(args.op=="max"){
        if(doSh) shRes=runSharedMax (d_in,d_fp,d_fo,n,blocks,args.warmup,args.reps,cMaxRef);
        if(doSf) sfRes=runShuffleMax(d_in,d_fp,d_fo,n,blocks,args.warmup,args.reps,cMaxRef);
    } else if(args.op=="argmax"){
        if(doSh) shRes=runSharedArgmax (d_in,d_pp,d_po,n,blocks,args.warmup,args.reps,cAmRef);
        if(doSf) sfRes=runShuffleArgmax(d_in,d_pp,d_po,n,blocks,args.warmup,args.reps,cAmRef);
    } else {
        std::cout<<"{\"status\":\"error\",\"message\":\"Unknown operation: "<<esc(args.op)<<"\"}\n";
        return 1;
    }

    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_fp)); CUDA_CHECK(cudaFree(d_fo));
    CUDA_CHECK(cudaFree(d_pp)); CUDA_CHECK(cudaFree(d_po));

    bool ok=true;
    if(doSh) ok=ok&&shRes.correct;
    if(doSf) ok=ok&&sfRes.correct;

    std::ostringstream j;
    j<<"{\n";
    j<<"  \"status\": \"ok\",\n";
    j<<"  \"operation\": \""<<esc(args.op)<<"\",\n";
    j<<"  \"input_size\": "<<n<<",\n";
    j<<"  \"seed\": "<<args.seed<<",\n";
    j<<"  \"pattern\": \""<<esc(args.pat)<<"\",\n";
    j<<"  \"correct\": "<<(ok?"true":"false")<<",\n";
    j<<"  \"results\": {\n";
    if(doSh&&doSf){
        j<<"    \"shared\": " <<toJson(shRes,isAm)<<",\n";
        j<<"    \"shuffle\": "<<toJson(sfRes,isAm)<<"\n";
    } else if(doSh){
        j<<"    \"shared\": " <<toJson(shRes,isAm)<<"\n";
    } else {
        j<<"    \"shuffle\": "<<toJson(sfRes,isAm)<<"\n";
    }
    j<<"  },\n";
    if(doSh&&doSf&&sfRes.median_us>0){
        float spd=shRes.median_us/sfRes.median_us;
        float lr=(shRes.median_us-sfRes.median_us)/shRes.median_us*100.0f;
        j<<"  \"comparison\": {\n";
        j<<"    \"speedup_shared_over_shuffle\": "<<ff(spd)<<",\n";
        j<<"    \"latency_reduction_percent\": "<<ff(lr)<<"\n";
        j<<"  },\n";
    } else {
        j<<"  \"comparison\": null,\n";
    }
    j<<"  \"metadata\": {\n";
    j<<"    \"gpu_name\": \""<<esc(gpuName)<<"\",\n";
    j<<"    \"threads_per_block\": "<<THREADS<<",\n";
    j<<"    \"blocks\": "<<blocks<<",\n";
    j<<"    \"warmup\": "<<args.warmup<<",\n";
    j<<"    \"repetitions\": "<<args.reps<<"\n";
    j<<"  }\n";
    j<<"}\n";
    std::cout<<j.str();
    return 0;
}
