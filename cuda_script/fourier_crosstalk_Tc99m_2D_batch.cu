// nvcc -O3 -use_fast_math -arch=sm_90 -o system_matrix_cuda_simple system_matrix_cuda_simple.cu -lcurand
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <numeric>
#include <complex>
#include <filesystem>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cufft.h>

// ---------------- Geometry & constants ----------------
#ifndef GX
#define GX 35
#endif
#ifndef GY
#define GY 35
#endif
#ifndef GZ
#define GZ 17
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;

static constexpr double SPACING = 3.0;   // mm
static constexpr double FIRST_X = -51.0; // mm (centers)
static constexpr double FIRST_Y = -51.0; // mm
static constexpr double FIRST_Z = -42.0; // mm
static constexpr double ZTOP    = 7.5;   // mm (top plane)
static constexpr int    NSAMPLES = 1000; // random samples per cube

// FOV (same as your CPU)
static constexpr double VX0=-50.0, VXSTEP=2.0, VX1=50.0;
static constexpr double VY0=0.0, VYSTEP=2.0, VY1=0.0;
static constexpr double VZ0=-50.0, VZSTEP=2.0, VZ1=50.0, VZOFF=150.0;
static constexpr int NX=int((VX1-VX0)/VXSTEP+1); // 75
static constexpr int NY=int((VY1-VY0)/VYSTEP+1); // 75
static constexpr int NZ=int((VZ1-VZ0)/VZSTEP+1); // 75
static constexpr int NVOX = NX*NY*NZ;

// Angles
static constexpr int NANG = 12;
static constexpr int VOXELS_PER_BLOCK = 64;
static constexpr int CUBE_WORKERS = 8;

// Physics (mm^-1)
static constexpr double MU_PB       = 2.59;
static constexpr double MU_PLASTIC  = 0.0;
static constexpr double MU_GAGG     = 0.48;
static constexpr double MU_PE_GAGG  = 0.40;
static constexpr double DELTA_V     = 0.3*0.3*0.3;


// ---------------- Utilities ----------------
#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr,"CUDA error at %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(1);                                                \
    }                                                           \
} while(0)

#define CUBLAS_CHECK(call) do {                                 \
    cublasStatus_t st = (call);                                 \
    if (st != CUBLAS_STATUS_SUCCESS) {                          \
        fprintf(stderr,"CUBLAS error at %s:%d: %d\n",           \
                __FILE__, __LINE__, (int)st);                   \
        exit(1);                                                \
    }                                                           \
} while(0)

// ---------------- Small helpers ----------------
__host__ __device__ inline int clampi(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }
__host__ __device__ inline int idx3(int ix,int iy,int iz){ return (iz*GY + iy)*GX + ix; }

struct Vec3 {
    double x,y,z;
    __host__ __device__ Vec3():x(0),y(0),z(0) {}
    __host__ __device__ Vec3(double X,double Y,double Z):x(X),y(Y),z(Z){}
    __host__ __device__ Vec3 operator+(const Vec3&o)const{return Vec3(x+o.x,y+o.y,z+o.z);}
    __host__ __device__ Vec3 operator-(const Vec3&o)const{return Vec3(x-o.x,y-o.y,z-o.z);}
    __host__ __device__ Vec3 operator*(double s)const{return Vec3(x*s,y*s,z*s);}
    __host__ __device__ double dot(const Vec3&o)const{return x*o.x+y*o.y+z*o.z;}
    __host__ __device__ double norm()const{return sqrt(x*x+y*y+z*z);}
    __host__ __device__ Vec3 normalized()const{ double n=norm(); return n>0?(*this*(1.0/n)):Vec3(0,0,0); }
};

struct GeometryBatch {
    std::vector<std::string> tags;
    std::vector<std::string> cube_files;
    std::vector<std::string> map_files;
};

struct GpuWorkspace {
    int dev;
    int Ncubes;
    int v_count;
    size_t row_count;
    size_t col_count;
    size_t matrix_elems;

    int*      d_map      = nullptr;
    double3*  d_cubes    = nullptr;
    double*   d_H        = nullptr;
    
    cuDoubleComplex* d_F   = nullptr;
    cuDoubleComplex* d_Hc  = nullptr;  // complex H (M x P)
    cuDoubleComplex* d_Hf  = nullptr;  // HΦ (M x P)
    cuDoubleComplex* d_Bc  = nullptr;  // (HΦ)^H(HΦ) complex (P x P)
    double*          d_B   = nullptr;  // abs(d_Bc) real (P x P)
    double*          d_B_diag = nullptr; // diag(d_B) (P)

    double* d_ens_ct = nullptr;
    cublasHandle_t handle = nullptr;
};

GpuWorkspace create_gpu_workspace(int dev, int Ncubes, int v_count)
{
    GpuWorkspace ws;
    ws.dev      = dev;
    ws.Ncubes   = Ncubes;
    ws.v_count  = v_count;
    ws.row_count = static_cast<size_t>(Ncubes) * NANG;
    ws.col_count = static_cast<size_t>(v_count);
    ws.matrix_elems = ws.row_count * ws.col_count;

    CUDA_CHECK(cudaSetDevice(dev));

    CUDA_CHECK(cudaMalloc(&ws.d_map,   GX*GY*GZ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws.d_cubes, Ncubes    * sizeof(double3)));
    CUDA_CHECK(cudaMalloc(&ws.d_H,     ws.matrix_elems * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&ws.d_F, ws.col_count * ws.col_count * sizeof(cuDoubleComplex)));

    CUDA_CHECK(cudaMalloc(&ws.d_Hc, ws.matrix_elems * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&ws.d_Hf, ws.matrix_elems * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&ws.d_Bc, ws.col_count * ws.col_count * sizeof(cuDoubleComplex)));

    CUDA_CHECK(cudaMalloc(&ws.d_B,    ws.col_count * ws.col_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ws.d_B_diag, ws.col_count * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&ws.d_ens_ct, sizeof(double)));

    CUBLAS_CHECK(cublasCreate(&ws.handle));

    return ws;
}

void destroy_gpu_workspace(GpuWorkspace& ws)
{
    CUDA_CHECK(cudaSetDevice(ws.dev));

    if (ws.handle) {
        cublasDestroy(ws.handle);
        ws.handle = nullptr;
    }

    cudaFree(ws.d_map);
    cudaFree(ws.d_cubes);
    cudaFree(ws.d_H);
    cudaFree(ws.d_F);
    cudaFree(ws.d_Hc);
    cudaFree(ws.d_Hf);
    cudaFree(ws.d_Bc);
    cudaFree(ws.d_B);
    cudaFree(ws.d_B_diag);
    cudaFree(ws.d_ens_ct);

    ws.d_map   = nullptr;
    ws.d_cubes = nullptr;
    ws.d_H     = nullptr;
    ws.d_F     = nullptr;
    ws.d_Hc    = nullptr;
    ws.d_Hf    = nullptr;
    ws.d_Bc    = nullptr;
    ws.d_B     = nullptr;
    ws.d_B_diag= nullptr;
    ws.d_ens_ct= nullptr;
}

// ---------------- Ray tracing (device) ----------------
#ifndef MAX_STEPS_PER_RAY
#define MAX_STEPS_PER_RAY 512
#endif

__device__ inline void ray_tracing_cubes_no_shielding1_discrete_symmetric_dev(
    const Vec3& sourcePos,
    const Vec3& pixPos,
    double ztop,
    const int* __restrict__ map_new2, // 35x35x8 materials (1,2=GAGG;3=Pb;4=acrylic)
    // outputs
    Vec3&   hit_top,   // intersection with top (or special case)
    Vec3&   hit_exit,  // last point before detector / end of object
    double& sum_muL,  // accumulated μ·L along the path in object
    int&    nsteps
){
    nsteps = 0;
    sum_muL = 0.0;

    Vec3 vec = pixPos - sourcePos;
    Vec3 dir = vec.normalized();
    double abz = fabs(dir.z);

    if (abz < 1e-15) {
        hit_top  = sourcePos;
        hit_exit = sourcePos;
        return;
    }

    Vec3 intersect_top = sourcePos + dir * ( fabs(sourcePos.z - ztop) / abz );
    double tTarget = fabs(sourcePos.z - pixPos.z) / abz;

    // quick XY check vs grid extent (+1.5 half-size)
    bool inside_xy = (fabs(intersect_top.x) <= fabs(FIRST_X + SPACING*(GX-1)) + 1.5) &&
                     (fabs(intersect_top.y) <= fabs(FIRST_Y + SPACING*(GY-1)) + 1.5);

    if (!inside_xy){
        hit_top  = intersect_top;
        hit_exit = intersect_top;
        return;
    }

    hit_top  = intersect_top;
    hit_exit = intersect_top;

    // face normals dot dir
    double vd[6];
    vd[0] = Vec3(0,0,-1).dot(dir); // bottom
    vd[1] = -vd[0];                // top
    vd[2] = Vec3(-1,0,0).dot(dir); // left
    vd[3] = -vd[2];                // right
    vd[4] = Vec3(0,-1,0).dot(dir); // front
    vd[5] = -vd[4];                // back

    int ind_out[6]; int n_out=0;
    #pragma unroll
    for(int i=0;i<6;++i) if (vd[i]>0.0) ind_out[n_out++] = i;

    Vec3 inputPos = intersect_top;

    // start bin at top layer
    auto find_idx = [] __device__ (double first, double spacing, int count, double val){
        double off = (val - first)/spacing;
        int idx = (int)floor(off + 0.5);
        return clampi(idx, 0, count-1);
    };
    int indx = find_idx(FIRST_X, SPACING, GX, inputPos.x);
    int indy = find_idx(FIRST_Y, SPACING, GY, inputPos.y);
    int indz = GZ-1;

    Vec3 cube_ctr(FIRST_X + SPACING*indx,
                  FIRST_Y + SPACING*indy,
                  FIRST_Z + SPACING*indz);

    int cube_ind[3] = {indx, indy, indz};
    int ind_far[6]; 
    int n_far=0;

    double tfar = fabs(sourcePos.z - ztop) / abz;

    int safety=0;
    while (tfar < tTarget && safety < MAX_STEPS_PER_RAY){
        ++safety;

        // shift next voxel by last chosen faces
        Vec3 shift(0,0,0);
        for(int k=0;k<n_far;++k){
            int idx = ind_far[k];
            if (idx==0) shift.z += -1;
            else if (idx==1) shift.z +=  1;
            else if (idx==2) shift.x += -1;
            else if (idx==3) shift.x +=  1;
            else if (idx==4) shift.y += -1;
            else if (idx==5) shift.y +=  1;
        }
        cube_ctr = cube_ctr + shift*SPACING;
        cube_ind[0] += (int)shift.x;
        cube_ind[1] += (int)shift.y;
        cube_ind[2] += (int)shift.z;

        // planes
        double d[6];
        d[0] = cube_ctr.z - 1.5;
        d[1] = -(cube_ctr.z + 1.5);
        d[2] = cube_ctr.x - 1.5;
        d[3] = -(cube_ctr.x + 1.5);
        d[4] = cube_ctr.y - 1.5;
        d[5] = -(cube_ctr.y + 1.5);

        double vn[6];
        vn[0] = Vec3(0,0,-1).dot(sourcePos) + d[0];
        vn[1] = Vec3(0,0, 1).dot(sourcePos) + d[1];
        vn[2] = Vec3(-1,0,0).dot(sourcePos) + d[2];
        vn[3] = Vec3( 1,0,0).dot(sourcePos) + d[3];
        vn[4] = Vec3(0,-1,0).dot(sourcePos) + d[4];
        vn[5] = Vec3(0, 1,0).dot(sourcePos) + d[5];

        // smallest positive -vn/vd among outward faces
        double min_tfar = 1e300;
        for (int k=0;k<n_out;++k){
            int idx = ind_out[k];
            double t = -vn[idx]/vd[idx];
            if (t>0.0 && t<min_tfar) min_tfar = t;
        }
        tfar = min_tfar;

        Vec3 resultPos = sourcePos + dir * tfar;

        // recompute ind_far: faces with t1==tfar and vd>0
        double t1[6];
        #pragma unroll
        for(int i=0;i<6;++i) t1[i] = -vn[i]/vd[i];

        int n_new=0;
        for(int i=0;i<6;++i)
            if (fabs(t1[i]-min_tfar) < 1e-9 && vd[i] > 0.0) ind_far[n_new++] = i;
        n_far = n_new;

        if (tfar < tTarget){

            double seglen = (resultPos - inputPos).norm();

            int ix = clampi(cube_ind[0],0,GX-1);
            int iy = clampi(cube_ind[1],0,GY-1);
            int iz = clampi(cube_ind[2],0,GZ-1);
            int matv = map_new2[idx3(ix,iy,iz)];
            
            double mu = 0.0;
            if (matv==1 || matv==2)      mu = MU_GAGG;
            else if (matv==3)            mu = MU_PB;
            else                         mu = MU_PLASTIC;

            sum_muL += mu * seglen;
            hit_exit = resultPos;
            ++nsteps;

            if (nsteps >= MAX_STEPS_PER_RAY) break;
        }
        inputPos = resultPos;
    }
}

// ---------------- Kernel: one thread per voxel ----------------
__global__ void sysmat_kernel(
    const int*    __restrict__ map_new2,     
    const double3*__restrict__ cube_centers, 
    int Ncubes,
    double*       __restrict__ H,            // (Ncubes*NANG) x NVOX, column-major by voxel
    int v_start,
    int v_count,
    unsigned long long seed_base)
{
    // block-level tiling
    int vox_tile = blockIdx.x;     // which tile of voxels
    int ang      = blockIdx.y;     // angle index

    int local_vox = threadIdx.x;   // 0 .. VOXELS_PER_BLOCK-1
    int worker    = threadIdx.y;   // 0 .. CUBE_WORKERS-1 (which subset of cubes)

    // global voxel index for this thread
    int col_local = vox_tile * VOXELS_PER_BLOCK + local_vox;
    if (col_local >= v_count || ang >= NANG) return;

    int v_global = v_start + col_local;

    curandState state;
    unsigned long long seq = ((unsigned long long)v_global * (unsigned long long)NANG + (unsigned long long)ang) * (unsigned long long)CUBE_WORKERS + (unsigned long long)worker;
    curand_init(seed_base, seq, 0, &state);

    // decode voxel index: x-fastest
    int vz = v_global / (NX*NY);
    int rem= v_global % (NX*NY);
    int vy = rem / NX;
    int vx = rem % NX;

    // source voxel center (with z offset +150)
    const double sx0 = VX0 + VXSTEP*vx;
    const double sy0 = VY0 + VYSTEP*vy;
    const double sz0 = VZ0 + VZSTEP*vz + VZOFF;

    // 3.14159265358979323846 / 36 = 5 deg
    const double theta = (3.14159265358979323846/(NANG/2)) * ang; // ang * 30deg
    const double ct = cos(theta);
    const double st = sin(theta);

    // Rotate the OBJECT by theta 
    const double sx =  ct*sx0 + st*(sz0-150.0);
    const double sy =  sy0;
    const double sz =  -st*sx0 + ct*(sz0-150.0) + 150;

    const size_t row_count = (size_t)Ncubes*(size_t)NANG;

    // per-thread outputs from ray tracer
    Vec3   hit_top, hit_exit;
    double sum_muL;
    int    nsteps;

    for(int j = worker; j < Ncubes; j += CUBE_WORKERS){
        
        // center of cube j
        double3 c3 = cube_centers[j];
        Vec3 c(c3.x, c3.y, c3.z);

        double prob_sum = 0.0;

        // NSAMPLES random points in cube (uniform in [-0.5,+0.5]^3)
        for(int n=0;n<NSAMPLES;++n){

            double rx = (curand_uniform_double(&state)-0.5)*3.0/2/0.5;
            double ry = (curand_uniform_double(&state)-0.5)*3.0/2/0.5;
            double rz = (curand_uniform_double(&state)-0.5)*3.0/2/0.5;
            Vec3 p = Vec3(c.x+rx, c.y+ry, c.z+rz);

            // ray trace
            ray_tracing_cubes_no_shielding1_discrete_symmetric_dev(Vec3(sx,sy,sz), p, ZTOP, map_new2, hit_top, hit_exit, sum_muL, nsteps);

            // survival
            double survive;
            double Ldet = (p - hit_exit).norm();
            survive = exp(-(sum_muL + MU_GAGG * Ldet));

            // geometry term
            double dx = p.x - sx, dy = p.y - sy, dz = p.z - sz;
            double R2 = dx*dx + dy*dy + dz*dz;
            if (R2 > 1e-18){
                prob_sum += DELTA_V * MU_PE_GAGG * survive / R2;
            }
        }

        // local voxel index within this chunk
        size_t col = (size_t)col_local;  
        size_t row = (size_t)ang * (size_t)Ncubes + (size_t)j;
        H[row + col * row_count] = prob_sum;

    }
}

// ---------------- Create Fourier Basis ----------------
void build_F_basis(std::vector<cuDoubleComplex>& F_host)
{
    F_host.resize((size_t)NVOX * (size_t)NVOX);
    std::cout << "Generating F_basis:\n";

    std::vector<double> xy(NX);
    for (int i = 0; i < NX; ++i) {
        xy[i] = VX0 + VXSTEP * i;
    }

    // vx, vy = linspace(-fs/2, fs/2, N)
    double fs = 1.0/VXSTEP;
    double v_start = -fs/2.0;          
    double v_end   =  fs/2.0;          
    std::vector<double> vx(NX), vy(NZ);
    for (int i = 0; i < NX; ++i) {
        double t = (NX == 1) ? 0.0 : (double)i / (double)(NX - 1);
        double v = v_start + t * (v_end - v_start);
        vx[i] = v;
        vy[i] = v;
    }

    double scale = 1.0 / std::sqrt((double)NVOX);

    for (int idy = 0; idy < NZ; ++idy) {
        double fy = vy[idy];
        for (int idx = 0; idx < NX; ++idx) {
            double fx = vx[idx];

            int j = idx + idy * NZ; 

            for (int py = 0; py < NZ; ++py) {
                double y = xy[py];

                for (int px = 0; px < NX; ++px) {
                    double x = xy[px];

                    int i = px + py * NZ;  // row

                    double phase = -2.0 * M_PI * (fx * x + fy * y);
                    double c = std::cos(phase);
                    double s = std::sin(phase);

                    cuDoubleComplex val;
                    val.x = scale * c;
                    val.y = scale * s;

                    F_host[(size_t)i + (size_t)j * (size_t)NVOX] = val;
                }
            }
        }
    }
}

// ---------------- Host I/O ----------------
static std::vector<double3> load_cube_pos(const char* path){
    std::ifstream f(path);
    if(!f) throw std::runtime_error("Failed to open cube_pos.txt");
    std::vector<double3> v; double x,y,z;
    while (f >> x >> y >> z) v.push_back(make_double3(x,y,z));
    return v; 
}

static std::vector<int> load_map_txt_35x35x17(const char* path){
    std::ifstream f(path);
    if(!f) throw std::runtime_error("Failed to open map.txt");
    std::vector<int> m2d(35*17*35);
    for(int r=0;r<35*17;++r)
        for(int c=0;c<35;++c)
            f >> m2d[r*35 + c];
    // reshape to [GX,GY,GZ] with x-fastest inside y inside z:
    std::vector<int> m3d(GX*GY*GZ,0);
    for(int z=0; z<GZ; ++z)
        for(int i=0; i<GX; ++i)
            for(int j=0; j<GY; ++j)
                m3d[idx3(i,j,z)] = m2d[(z*35 + i)*35 + j];
    return m3d;
}

GeometryBatch load_geometry(const std::string& geom_dir)
{
    GeometryBatch batch;

    std::vector<std::string> tags;
    std::vector<std::string> cube_files;
    std::vector<std::string> map_files;

    const std::string prefix = "cube_pos_";
    const std::string suffix = ".txt";

    // Scan directory for cube_pos_<TAG>.txt
    for (const auto& entry : fs::directory_iterator(geom_dir)) {

        const std::string fname = entry.path().filename().string();

        if (fname.rfind(prefix, 0) != 0) {   
            continue;
        }

        const std::string tag = fname.substr(prefix.size(), fname.size() - prefix.size() - suffix.size());

        std::string cube_path = entry.path().string();
        std::string map_path  = (fs::path(geom_dir) / ("map_" + tag + ".txt")).string();

        tags.push_back(tag);
        cube_files.push_back(cube_path);
        map_files.push_back(map_path);
    }

    const size_t N = tags.size();

    // Sort tags
    std::vector<size_t> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {return tags[a] < tags[b];});

    batch.tags.resize(N);
    batch.cube_files.resize(N);
    batch.map_files.resize(N);

    for (size_t i = 0; i < N; ++i) {
        batch.tags[i]       = tags[order[i]];
        batch.cube_files[i] = cube_files[order[i]];
        batch.map_files[i]  = map_files[order[i]];
    }

    return batch;
}

// GPU Kernel Functions
__global__ void real_to_complex_kernel(const double* __restrict__ in, cuDoubleComplex* __restrict__ out, size_t n)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t k = idx; k < n; k += stride) {
        out[k].x = in[k];
        out[k].y = 0.0;
    }
}

__global__ void complex_to_real_kernel(const cuDoubleComplex* __restrict__ Bc, double* __restrict__ B, size_t n)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t k = idx; k < n; k += stride) {
        double re = Bc[k].x;
        double im = Bc[k].y;
        B[k] = sqrt(re*re + im*im);
    }
}

__global__ void diag_kernel(const double* __restrict__ B, double* __restrict__ diag, int col_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < col_count) diag[i] = B[(size_t)i + (size_t)i * (size_t)col_count];
}

__global__ void normalize_columns_kernel(double* __restrict__ B, const double* __restrict__ diag, int col_count)
{
    size_t total = (size_t)col_count * (size_t)col_count;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t k = idx; k < total; k += stride) {
        int j = (int)(k / (size_t)col_count); // column
        double d = diag[j];
        if (d > 0.0) B[k] /= d;
    }
}

__global__ void compute_avg_crosstalk_kernel(const double* __restrict__ B, size_t n, double* __restrict__ out)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    double local = 0.0;
    for (size_t k = idx; k < n; k += stride) {
        double v = B[k];
        local += v * v;
    }
    atomicAdd(out, local);
}

// -------------------- Per-GPU worker --------------------
double run_gpu_chunk(GpuWorkspace& ws,
                   int v_start,
                   const std::vector<int>& h_map,
                   const std::vector<double3>& h_cubes)
{
    CUDA_CHECK(cudaSetDevice(ws.dev));
    size_t row_count=ws.row_count;
    size_t col_count=ws.col_count;
    size_t matrix_elems=ws.matrix_elems;

    CUDA_CHECK(cudaMemcpy(ws.d_map,h_map.data(),GX*GY*GZ*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws.d_cubes,h_cubes.data(),ws.Ncubes*sizeof(double3),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ws.d_H,0,matrix_elems*sizeof(double)));

    int vox_tiles = (ws.v_count + VOXELS_PER_BLOCK - 1) / VOXELS_PER_BLOCK;
    dim3 blk(VOXELS_PER_BLOCK, CUBE_WORKERS); 
    dim3 grd(vox_tiles, NANG);

    unsigned long long seed_base = 1337ULL + (unsigned long long)ws.dev;
    sysmat_kernel<<<grd,blk>>>(ws.d_map,ws.d_cubes,ws.Ncubes,ws.d_H,v_start,ws.v_count,seed_base);
    CUDA_CHECK(cudaGetLastError());   
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        const int t = 256;
        const int b = (int)((matrix_elems + t - 1) / t);
        real_to_complex_kernel<<<b, t>>>(ws.d_H, ws.d_Hc, matrix_elems);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);

        CUBLAS_CHECK(cublasZgemm(ws.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 (int)row_count, (int)col_count, (int)col_count,
                                 &alpha,
                                 ws.d_Hc,  (int)row_count,
                                 ws.d_F, (int)col_count,
                                 &beta,
                                 ws.d_Hf,  (int)row_count));
    }

    {
        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);

        CUBLAS_CHECK(cublasZgemm(ws.handle, CUBLAS_OP_C, CUBLAS_OP_N,
                                 (int)col_count, (int)col_count, (int)row_count,
                                 &alpha,
                                 ws.d_Hf, (int)row_count,
                                 ws.d_Hf, (int)row_count,
                                 &beta,
                                 ws.d_Bc, (int)col_count));
    }

    {
        const size_t n = col_count * col_count;
        const int t = 256;
        const int b = (int)((n + t - 1) / t);
        complex_to_real_kernel<<<b, t>>>(ws.d_Bc, ws.d_B, n);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        const int t = 256;
        const int b = (int)((col_count + t - 1) / t);
        diag_kernel<<<b, t>>>(ws.d_B, ws.d_B_diag, (int)col_count);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        const size_t n = col_count * col_count;
        const int t = 256;
        const int b = (int)((n + t - 1) / t);
        normalize_columns_kernel<<<b, t>>>(ws.d_B, ws.d_B_diag, (int)col_count);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemset(ws.d_ens_ct, 0, sizeof(double)));
    {
        const size_t n = col_count * col_count;
        const int t = 256;
        const int b = 4096; 
        compute_avg_crosstalk_kernel<<<b, t>>>(ws.d_B, n, ws.d_ens_ct);
        CUDA_CHECK(cudaGetLastError());
    }

    double ens_ct = 0.0;
    CUDA_CHECK(cudaMemcpy(&ens_ct, ws.d_ens_ct, sizeof(double), cudaMemcpyDeviceToHost));

    const double mean_ens_ct = std::sqrt(ens_ct / (double)(col_count * col_count));

    return mean_ens_ct;
}

// ---------------- main ----------------
int main(int argc, char** argv){
    try{
        const std::string geom_path = argv[1];
        const std::string out_path  = argv[2];

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        std::cout << "Found " << device_count << " CUDA device(s).\n";

        GeometryBatch batch = load_geometry(geom_path);
        const auto& tags       = batch.tags;
        const auto& cube_files = batch.cube_files;
        const auto& map_files  = batch.map_files;

        const size_t N = tags.size();
        std::cout << "Found " << N << " geometry designs.\n";

        std::vector<double> avg_crosstalk(N, 0.0);
        std::mutex io_mutex;

        std::vector<cuDoubleComplex> h_F;
        build_F_basis(h_F);

        auto worker = [&](int dev_id) {

            int Ncubes_fixed = 450;
            int v_count      = NVOX;

            GpuWorkspace ws = create_gpu_workspace(dev_id, Ncubes_fixed, v_count);
            CUDA_CHECK(cudaMemcpy(ws.d_F,h_F.data(),NVOX*NVOX*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice));

            for (size_t i = 0; i < N; ++i) {

                if (static_cast<int>(i % device_count) != dev_id) {
                    continue;
                }

                const std::string& tag        = tags[i];
                const std::string& cube_path  = cube_files[i];
                const std::string& map_path   = map_files[i];

                try {
                    // ---- Load cubes for this design ----
                    std::vector<double3> h_cubes = load_cube_pos(cube_path.c_str());
                    const int Ncubes_all = static_cast<int>(h_cubes.size());

                    // Filter bottom-layer cubes: z <= -30
                    std::vector<double3> h_bottom_cubes;
                    h_bottom_cubes.reserve(Ncubes_all);
                    for (int k = 0; k < Ncubes_all; ++k) {
                        if (h_cubes[k].z <= -30.0) {
                            h_bottom_cubes.push_back(h_cubes[k]);
                        }
                    }
                    const int Ncubes_bottom = static_cast<int>(h_bottom_cubes.size());
                    std::vector<int> h_map = load_map_txt_35x35x17(map_path.c_str());

                    std::vector<double> F(static_cast<size_t>(NVOX) * NVOX, 0.0);

                    double tmp = run_gpu_chunk(ws, 0, h_map, h_bottom_cubes);
                    avg_crosstalk[i] = tmp;

                    {
                        std::lock_guard<std::mutex> lock(io_mutex);
                        std::cout << "[GPU " << dev_id << "] design " << tag << " -> Average Fourier Crosstalk = " << tmp << "\n";
                    }
                }
                catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(io_mutex);
                    std::cerr << "[GPU " << dev_id << "] ERROR for tag " << tag << ": " << e.what() << "\n";
                }
            }

            destroy_gpu_workspace(ws);

        };

        // ---------------- Launch one CPU thread per GPU ----------------
        std::vector<std::thread> threads;
        threads.reserve(device_count);
        for (int d = 0; d < device_count; ++d) {
            threads.emplace_back(worker, d);
        }
        for (auto& t : threads) {
            t.join();
        }

        std::ofstream out(out_path,std::ios::binary);
        out.write(reinterpret_cast<const char*>(avg_crosstalk.data()), (std::streamsize)(avg_crosstalk.size()*sizeof(double)));
        out.close();
    } 
    catch (const std::exception& e){
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
