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
#include <filesystem>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

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

namespace fs = std::filesystem;

static constexpr double SPACING = 3.0;   // mm
static constexpr double FIRST_X = -51.0; // mm (centers)
static constexpr double FIRST_Y = -51.0; // mm
static constexpr double FIRST_Z = -42.0; // mm
static constexpr double ZTOP    = 7.5;   // mm (top plane)
static constexpr int    NSAMPLES = 100;  // random samples per cube

// FOV (same as your CPU)
static constexpr double VX0=-74.0, VXSTEP=2.0, VX1=74.0;
static constexpr double VY0=0.0, VYSTEP=2.0, VY1=0.0;
static constexpr double VZ0=-74.0, VZSTEP=2.0, VZ1=74.0, VZOFF=150.0;
static constexpr int NX=int((VX1-VX0)/VXSTEP+1); // 75
static constexpr int NY=int((VY1-VY0)/VYSTEP+1); // 75
static constexpr int NZ=int((VZ1-VZ0)/VZSTEP+1); // 75
static constexpr int NVOX = NX*NY*NZ;

// Angles
static constexpr int NANG = 24;
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
    double*   d_f        = nullptr;
    double*   d_g        = nullptr;
    double*   d_w        = nullptr;
    double*   d_F        = nullptr;

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
    CUDA_CHECK(cudaMalloc(&ws.d_f,     NVOX        * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ws.d_g,     ws.row_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ws.d_w,     ws.row_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ws.d_F,     ws.col_count * ws.col_count * sizeof(double)));

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
    cudaFree(ws.d_f);
    cudaFree(ws.d_g);
    cudaFree(ws.d_w);
    cudaFree(ws.d_F);

    ws.d_map   = nullptr;
    ws.d_cubes = nullptr;
    ws.d_H     = nullptr;
    ws.d_f     = nullptr;
    ws.d_g     = nullptr;
    ws.d_w     = nullptr;
    ws.d_F     = nullptr;
}


__device__ inline void ray_tracing_sci_spect(
    const Vec3& sourcePos,
    const Vec3& pixPos,
    double ztop,
    const int* __restrict__ d_map, 
    double& sum_muL  // accumulated μ·L along the path in object
){
    sum_muL = 0.0;

    const double mu_table[] = {
        MU_PLASTIC, // 0
        MU_GAGG,    // 1
        MU_GAGG,    // 2
        MU_PB       // 3
    };

    Vec3 vec = pixPos - sourcePos;
    Vec3 dir = vec.normalized();
    double abz = fabs(dir.z);

    if (abz < 1e-15) {
        return;
    }

    Vec3 intersect_top = sourcePos + dir * (fabs(sourcePos.z - ztop) / abz );
    double tTarget = fabs(sourcePos.z - pixPos.z) / abz;

    // quick XY check vs grid extent (+1.5 half-size)
    bool inside_xy = (fabs(intersect_top.x) <= fabs(FIRST_X + SPACING*(GX-1)) + 1.5) &&
                     (fabs(intersect_top.y) <= fabs(FIRST_Y + SPACING*(GY-1)) + 1.5);

    if (!inside_xy){
        return;
    }

    // face normals dot dir
    double vd[6];
    vd[0] = -dir.z; // bottom
    vd[1] = dir.z;  // top
    vd[2] = -dir.x; // left
    vd[3] = dir.x;  // right
    vd[4] = -dir.y; // front
    vd[5] = dir.y;  // back

    int ind_out[6]; int n_out=0;
    #pragma unroll
    for(int i=0;i<6;++i) if (vd[i]>0.0) ind_out[n_out++] = i;

    double inv_vd[6];
    for(int k=0;k<n_out;++k){
        inv_vd[k] = 1.0 / vd[ind_out[k]];
    }

    double t_prev = fabs(sourcePos.z - ztop) / abz;
    double t_far   = t_prev;
    double t_last_valid = 0;

    int indx = (int)floor((intersect_top.x - FIRST_X)/SPACING + 0.5);
    int indy = (int)floor((intersect_top.y - FIRST_Y)/SPACING + 0.5);
    int indz = GZ-1;

    Vec3 cube_ctr(FIRST_X + SPACING*indx,
                  FIRST_Y + SPACING*indy,
                  FIRST_Z + SPACING*indz);

    int cube_ind[3] = {indx, indy, indz};
    int ind_far[6]; 
    int n_far=0;

    // double tfar = fabs(sourcePos.z - ztop) / abz;
    double half_spacing = SPACING/2;

    while (t_far < tTarget){

        // shift next voxel by last chosen faces
        for(int k=0;k<n_far;++k){
            switch(ind_far[k]){
                case 0: cube_ind[2]--; cube_ctr.z -= SPACING; break;
                case 1: cube_ind[2]++; cube_ctr.z += SPACING; break;
                case 2: cube_ind[0]--; cube_ctr.x -= SPACING; break;
                case 3: cube_ind[0]++; cube_ctr.x += SPACING; break;
                case 4: cube_ind[1]--; cube_ctr.y -= SPACING; break;
                case 5: cube_ind[1]++; cube_ctr.y += SPACING; break;
            }
        }

        double vn[6];
        vn[0] = -sourcePos.z + cube_ctr.z - half_spacing;
        vn[1] =  sourcePos.z - cube_ctr.z - half_spacing;
        vn[2] = -sourcePos.x + cube_ctr.x - half_spacing;
        vn[3] =  sourcePos.x - cube_ctr.x - half_spacing;
        vn[4] = -sourcePos.y + cube_ctr.y - half_spacing;
        vn[5] =  sourcePos.y - cube_ctr.y - half_spacing;

        // smallest positive -vn/vd among outward faces
        double min_tfar = 1e300;
        double t_out[6];
        for (int k=0;k<n_out;++k){
            double t = -vn[ind_out[k]] * inv_vd[k];
            t_out[k] = t;
            if (t>0.0 && t<min_tfar) min_tfar = t;
        }
        t_far = min_tfar;

        int n_new=0;
        for (int k=0;k<n_out;++k){
            if (fabs(t_out[k]-min_tfar) < 1e-9) ind_far[n_new++] = ind_out[k];
        }
        n_far = n_new;

        if (t_far < tTarget){

            double seglen = t_far - t_prev;

            int ix = cube_ind[0];
            int iy = cube_ind[1];
            int iz = cube_ind[2];
            int matv = d_map[idx3(ix,iy,iz)];
            
            double mu = mu_table[matv];
            sum_muL += mu * seglen;

        }
        t_last_valid = t_prev;
        t_prev = t_far;
    }
    double Ldet = tTarget - t_last_valid;
    sum_muL += Ldet*MU_GAGG;
}

__global__ void create_uniform_phantom(double* d_f, double val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NVOX) d_f[i] = val;
}

__global__ void inv_proj(const double* __restrict__ d_g, double* __restrict__ d_w, int row_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row_count) d_w[i] = 1.0/d_g[i];
}

__global__ void scale_sysmat(double* __restrict__ d_H, const double* __restrict__ d_w, int row_count, int col_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = row_count * col_count;
    if (idx >= total) return;

    int i = idx % row_count;  // row index
    double s = sqrt(d_w[i]);
    d_H[idx] *= s;
}

// ---------------- Kernel: one thread per voxel ----------------
__global__ void sysmat_kernel(
    const int*    __restrict__ d_map,     
    const double3*__restrict__ cube_centers, 
    int Ncubes,
    double*       __restrict__ H,            // (Ncubes*NANG) x NVOX, column-major by voxel
    int v_start,
    int v_count,
    unsigned long long seed_base)
{

    // load map to shared memory

    int block_size = blockDim.y * blockDim.x;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    const int map_size = 35*35*17;

    extern __shared__ int ds_map[];

    for (int i = tid; i<map_size; i+=block_size) {
       ds_map[i] = d_map[i];
    }

    __syncthreads();

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
    double sum_muL;

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
            ray_tracing_sci_spect(Vec3(sx,sy,sz), p, ZTOP, ds_map, sum_muL);

            // survival
            double survive;
            survive = exp(-sum_muL);

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
    for(int k=0; k<GZ; ++k)
        for(int i=0; i<GX; ++i)
            for(int j=0; j<GY; ++j)
                m3d[idx3(i,j,k)] = m2d[(k*35 + i)*35 + j];
    return m3d;
}

// -------------------- Per-GPU worker --------------------
void run_gpu_chunk(GpuWorkspace& ws,
                   int v_start,
                   const std::vector<int>& h_map,
                   const std::vector<double3>& h_cubes,
                   std::vector<double>& F)
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

    size_t smem_bytes = 83300;

    cudaFuncSetAttribute(
        sysmat_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );

    unsigned long long seed_base = 1337ULL + (unsigned long long)ws.dev;
    sysmat_kernel<<<grd,blk,smem_bytes>>>(ws.d_map,ws.d_cubes,ws.Ncubes,ws.d_H,v_start,ws.v_count,seed_base);
    CUDA_CHECK(cudaGetLastError());   
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        int t = 256;
        int b = (int)((col_count + t - 1) / t);
        create_uniform_phantom<<<b, t>>>(ws.d_f, 40.0);
        CUDA_CHECK(cudaGetLastError());
    }

    // y = H * x, H is MxN, col-major, lda = M
    {
        const double alpha = 1.0;
        const double beta  = 0.0;
        CUBLAS_CHECK(cublasDgemv(ws.handle,
                                 CUBLAS_OP_N,  
                                 (int)row_count, (int)col_count,
                                 &alpha,
                                 ws.d_H, (int)row_count,
                                 ws.d_f, 1,
                                 &beta,
                                 ws.d_g, 1));
    }

    // w = 1./y (elementwise)
    {
        int t = 256;
        int b = (int)((row_count + t - 1) / t);
        inv_proj<<<b, t>>>(ws.d_g, ws.d_w, (int)row_count);
        CUDA_CHECK(cudaGetLastError());
    }

    // Scale rows of H in-place: H(i,j) *= sqrt(w[i])
    {
        int t = 256;
        int b = (int)((matrix_elems + t - 1) / t);
        scale_sysmat<<<b, t>>>(ws.d_H, ws.d_w, (int)row_count, (int)col_count);
        CUDA_CHECK(cudaGetLastError());
    }

    // F = H'^H  (where H is already row-scaled => H'^H = H'^diag(w)H)
    {
        const double alpha = 1.0;
        const double beta  = 0.0;
        CUBLAS_CHECK(cublasDgemm(ws.handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 (int)col_count, (int)col_count, (int)row_count,
                                 &alpha,
                                 ws.d_H, (int)row_count,   // A: MxN
                                 ws.d_H, (int)row_count,   // B: MxN
                                 &beta,
                                 ws.d_F, (int)col_count)); // C: NxN
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(F.data(),ws.d_F,col_count*col_count*sizeof(double),cudaMemcpyDeviceToHost));
}

// Compute average FWHM (over x & y) from Fisher matrix F
double compute_avg_fim_fwhm(const std::vector<double>& F)
{
    // F is Nvox x Nvox -> Nvox^2 elements
    const size_t Fsize = F.size();

    const int Nxy = static_cast<int>(std::sqrt(static_cast<double>(NVOX)));

    std::vector<double> xy(Nxy);
    const double xy_min = -74.0;
    const double xy_max =  74.0;
    const double step   = (xy_max - xy_min)/static_cast<double>(Nxy - 1);
    for (int i = 0; i < Nxy; ++i) {
        xy[i] = xy_min + step * static_cast<double>(i);
    }

    // Build support indices: sup_ind = find(X.^2 + Y.^2 < R^2)
    std::vector<int> sup_lin;
    std::vector<int> sup_ix;
    std::vector<int> sup_iy;
    sup_lin.reserve(Nxy * Nxy);

    const double R2_max = 60.0 * 60.0;
    for (int iy = 0; iy < Nxy; ++iy) {
        for (int ix = 0; ix < Nxy; ++ix) {
            double X = xy[ix]; 
            double Y = xy[iy]; 
            if (X*X + Y*Y < R2_max) {
                int lin = iy + ix * Nxy; // column-major linear index
                sup_lin.push_back(lin);
                sup_ix.push_back(ix);
                sup_iy.push_back(iy);
            }
        }
    }

    const int NSUP = static_cast<int>(sup_lin.size());

    // Local helper to compute FWHM for one voxel (one column of F)
    auto fwhm_for_voxel = [&](const double* col, int x_ind, int y_ind, double& x_fwhm, double& y_fwhm) {

        // Central value at (y_ind, x_ind)
        const int center_idx = y_ind + x_ind * Nxy;
        const double center_val = col[center_idx];
        const double half_val   = center_val / 2.0;

        // --- x_profile = img(y_ind, :) ---
        // Find left index in [0, x_ind] where |x_profile - half| is minimal
        int left_idx  = x_ind;
        double best_l = std::numeric_limits<double>::infinity();
        for (int i = 0; i <= x_ind; ++i) {
            double val  = col[y_ind + i * Nxy];
            double diff = std::fabs(val - half_val);
            if (diff < best_l) {
                best_l   = diff;
                left_idx = i;
            }
        }

        // Find right index in [x_ind+1, Nxy-1] where |x_profile - half| is minimal
        int right_idx  = x_ind;
        double best_r  = std::numeric_limits<double>::infinity();
        for (int i = x_ind + 1; i < Nxy; ++i) {
            double val  = col[y_ind + i * Nxy];
            double diff = std::fabs(val - half_val);
            if (diff < best_r) {
                best_r   = diff;
                right_idx = i;
            }
        }

        x_fwhm = (right_idx - left_idx) * step; 

        // --- y_profile = img(:, x_ind) ---
        int left_idy = y_ind;
        best_l = std::numeric_limits<double>::infinity();
        for (int j = 0; j <= y_ind; ++j) {
            double val  = col[j + x_ind * Nxy];
            double diff = std::fabs(val - half_val);
            if (diff < best_l) {
                best_l   = diff;
                left_idy = j;
            }
        }

        int right_idy = y_ind;
        best_r = std::numeric_limits<double>::infinity();
        for (int j = y_ind + 1; j < Nxy; ++j) {
            double val  = col[j + x_ind * Nxy];
            double diff = std::fabs(val - half_val);
            if (diff < best_r) {
                best_r   = diff;
                right_idy = j;
            }
        }

        y_fwhm = (right_idy - left_idy) * step;
    };

    // Loop over support voxels and accumulate FWHMs
    double sum_fwhm = 0.0;
    int count_fwhm  = 0;

    for (int k = 0; k < NSUP; ++k) {
        int vox_idx = sup_lin[k];   
        int x_ind   = sup_ix[k];    
        int y_ind   = sup_iy[k];    

        const double* col = &F[static_cast<size_t>(vox_idx) * static_cast<size_t>(NVOX)];

        double x_fwhm = 0.0;
        double y_fwhm = 0.0;
        fwhm_for_voxel(col, x_ind, y_ind, x_fwhm, y_fwhm);

        sum_fwhm += x_fwhm + y_fwhm;
        count_fwhm += 2;  // x & y
    }

    double avg_fim_fwhm = sum_fwhm/static_cast<double>(count_fwhm);
    return avg_fim_fwhm;
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

        std::vector<double> avg_fim_fwhm(N, 0.0);
        std::mutex io_mutex;

        auto worker = [&](int dev_id) {

            int Ncubes_fixed = 162;
            int v_count      = NVOX;

            GpuWorkspace ws = create_gpu_workspace(dev_id, Ncubes_fixed, v_count);
            std::vector<double> F((size_t)v_count * v_count);

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
                    // const int Ncubes_bottom = static_cast<int>(h_bottom_cubes.size());
                    std::vector<int> h_map = load_map_txt_35x35x17(map_path.c_str());

                    std::vector<double> F(static_cast<size_t>(NVOX) * NVOX, 0.0);

                    run_gpu_chunk(ws, 0, h_map, h_bottom_cubes, F);

                    double fwhm_val = compute_avg_fim_fwhm(F);
                    avg_fim_fwhm[i] = fwhm_val;

                    {
                        std::lock_guard<std::mutex> lock(io_mutex);
                        std::cout << "[GPU " << dev_id << "] design " << tag << " -> Average FIM FWHM = " << fwhm_val << "\n";
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
        out.write(reinterpret_cast<const char*>(avg_fim_fwhm.data()), (std::streamsize)(avg_fim_fwhm.size()*sizeof(double)));
        out.close();
    } 
    catch (const std::exception& e){
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
