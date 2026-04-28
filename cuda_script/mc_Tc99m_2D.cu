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

static constexpr double SPACING = 3.0;   // mm
static constexpr double FIRST_X = -51.0; // mm (centers)
static constexpr double FIRST_Y = -51.0; // mm
static constexpr double FIRST_Z = -42.0; // mm
static constexpr double ZTOP    = 7.5;   // mm (top plane)
static constexpr int    NSAMPLES = 1000; // random samples per cube

// FOV (same as your CPU)
static constexpr double VX0=-60.0, VXSTEP=2.0, VX1=60.0;
static constexpr double VY0=0.0, VYSTEP=2.0, VY1=0.0;
static constexpr double VZ0=-60.0, VZSTEP=2.0, VZ1=60.0, VZOFF=150.0;
static constexpr int NX=int((VX1-VX0)/VXSTEP+1); // 75
static constexpr int NY=int((VY1-VY0)/VYSTEP+1); // 75
static constexpr int NZ=int((VZ1-VZ0)/VZSTEP+1); // 75
static constexpr int NVOX = NX*NY*NZ;

// Angles
static constexpr int NANG = 48;
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

__global__ void frob_loss_kernel(const double* F,
                                     int N,
                                     double* loss)
{
    unsigned int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned long long total = (unsigned long long)N * (unsigned long long)N;

    double local_sum = 0.0;

    for (unsigned long long k = idx; k < total; k += stride) {
        int i = (int)(k % N);   // column index
        int j = (int)(k / N);   // row index
        if (i != j) {
            double v = F[k];    // F is N x N in column-major
            local_sum += v * v;
        }
    }

    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss, sdata[0]);
    }
}

// -------------------- Per-GPU worker --------------------
void run_gpu_chunk(int dev,int v_start,int v_count, int Ncubes,
                   const std::vector<int>& h_map,
                   const std::vector<double3>& h_cubes,
                   double& h_loss)
{
    CUDA_CHECK(cudaSetDevice(dev));
    size_t row_count=(size_t)Ncubes*NANG;
    size_t col_count=(size_t)v_count;
    size_t matrix_elems=row_count*col_count;

    // std::cout<<"[GPU "<<dev<<"] voxels "<<v_start<<"-"<<v_start+v_count-1<<std::endl;

    int* d_map; double3* d_cubes; double* d_H; double* d_f; double* d_g; double* d_w; double* d_F; double*  d_loss;
    CUDA_CHECK(cudaMalloc(&d_map,GX*GY*GZ*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cubes,Ncubes*sizeof(double3)));
    CUDA_CHECK(cudaMalloc(&d_H,matrix_elems*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f,NVOX*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_g,row_count*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w,row_count*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_F,col_count*col_count*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_loss,  sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_map,h_map.data(),GX*GY*GZ*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cubes,h_cubes.data(),Ncubes*sizeof(double3),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_H,0,matrix_elems*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(double)));

    int vox_tiles = (v_count + VOXELS_PER_BLOCK - 1) / VOXELS_PER_BLOCK;
    dim3 blk(VOXELS_PER_BLOCK, CUBE_WORKERS); 
    dim3 grd(vox_tiles, NANG);

    unsigned long long seed_base = 1337ULL + (unsigned long long)dev;
    sysmat_kernel<<<grd,blk>>>(d_map,d_cubes,Ncubes,d_H,v_start,v_count,seed_base);
    CUDA_CHECK(cudaGetLastError());   
    CUDA_CHECK(cudaDeviceSynchronize());

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    for (int j = 0; j < (int)col_count; ++j) {
        double norm_j = 0.0;
        CUBLAS_CHECK(cublasDnrm2(handle,
                                 (int)row_count,
                                 d_H + (size_t)j * row_count,  // column j
                                 1,
                                 &norm_j));

        double scale = 1.0 / norm_j;
        CUBLAS_CHECK(cublasDscal(handle,
                                    (int)row_count,
                                    &scale,
                                    d_H + (size_t)j * row_count,
                                    1));
    }

    {
        const double alpha = 1.0;
        const double beta  = 0.0;
        CUBLAS_CHECK(cublasDgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 (int)col_count, (int)col_count, (int)row_count,
                                 &alpha,
                                 d_H, (int)row_count,   // A: MxN
                                 d_H, (int)row_count,   // B: MxN
                                 &beta,
                                 d_F, (int)col_count)); // C: NxN
    }
    // CUDA_CHECK(cudaDeviceSynchronize());

    {
        int threads = 256;
        int blocks  = (int)std::min<size_t>((col_count * col_count + threads - 1) / threads, 1024ULL);
        size_t shmem = threads * sizeof(double);

        frob_loss_kernel<<<blocks, threads, shmem>>>(d_F, (int)col_count, d_loss);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost));
    }

    cudaFree(d_map); 
    cudaFree(d_cubes); 
    cudaFree(d_H); 
    cudaFree(d_f);
    cudaFree(d_g);
    cudaFree(d_w);
    cudaFree(d_F);
}

// ---------------- main ----------------
int main(int argc, char** argv){
    try{

        const std::string cube_pos_path = argv[1];
        const std::string map_path      = argv[2];
        const std::string out_path      = argv[3];
        const std::string tar_dev       = argv[4];

        int dev = std::stoi(tar_dev);

        // Load inputs
        std::vector<double3> h_cubes = load_cube_pos(cube_pos_path.c_str());
        const int Ncubes_all = (int)h_cubes.size();
        // std::cout<<"Current design has "<<Ncubes_all<<" crystals.\n";

        // Filter to bottom-layer cubes: z <= -30 
        std::vector<double3> h_bottom_cubes;
        h_bottom_cubes.reserve(Ncubes_all);
        for (int i = 0; i < Ncubes_all; ++i) {
            if (h_cubes[i].z <= -30.0) {       // bottom layer criterion
                h_bottom_cubes.push_back(h_cubes[i]);
            }
        }
        const int Ncubes_bottom = (int)h_bottom_cubes.size();
        // std::cout<<"Using "<<Ncubes_bottom<<" bottom-layer crystals (z < -30).\n";

        std::vector<int> h_map = load_map_txt_35x35x17(map_path.c_str());

        // ---------------- Query GPUs ----------------
        double h_loss;

        // Launch GPU Computation
        int v_start = 0;
        int v_count = NVOX;
        run_gpu_chunk(dev, v_start, v_count, Ncubes_bottom, h_map, h_bottom_cubes, h_loss);

        // Compute Mutual Coherence
        std::cout << "Mutual Coherence = " << h_loss << std::endl;

        std::ofstream out(out_path,std::ios::binary);
        out.write(reinterpret_cast<const char*>(&h_loss), (std::streamsize)(sizeof(double)));
        out.close();
    } 
    
    catch (const std::exception& e){
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

}
