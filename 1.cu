#include <iostream>

using namespace std;

#define TYPE float
typedef TYPE T;

__constant__ float dev_box[4];
__constant__ int dev_threads[1];
__constant__ int dev_blocks[1];
__constant__ int dev_n_of_ints[1];
__constant__ int dev_n_of_func = 4;

template<class T>
class interval_gpu
{
    public:
        __device__ __host__ interval_gpu();
        __device__ __host__ interval_gpu(T const &v);
        __device__ __host__ interval_gpu(T const &l, T const &u);
        __device__ __host__ T const &lower() const;
        __device__ __host__ T const &upper() const;
        static __device__ __host__ interval_gpu empty();
friend ostream& operator<<(ostream& os, const interval_gpu<T> &x){
os<<"["<<x.lower()<<":"<<x.upper()<<"]";return os;
}
    private: T low; T up;
};
// Constructors
template<class T> inline __device__ __host__
interval_gpu<T>::interval_gpu(){}
template<class T> inline __device__ __host__
interval_gpu<T>::interval_gpu(T const &v) :
    low(v), up(v){}
template<class T> inline __device__ __host__
interval_gpu<T>::interval_gpu(T const &l, T const &u) :
    low(l), up(u){}

template<class T> inline __device__ __host__
T const &interval_gpu<T>::lower() const
{return low;}

template<class T> inline __device__ __host__
T const &interval_gpu<T>::upper() const
{return up;}
//OVERLOAD OVERLOAD OVERLOAD OVERLOAD OVERLOAD OVERLOAD OVERLOAD OVERLOAD OVERLOAD
template<class T> inline __host__ __device__
interval_gpu<T> operator+(interval_gpu<T> const &x, interval_gpu<T> const &y)
{
  return interval_gpu<T>(x.lower() + y.lower(), x.upper() + y.upper());
}
template<class T> inline __host__ __device__
interval_gpu<T> operator-(interval_gpu<T> const &x, interval_gpu<T> const &y)
{return interval_gpu<T>(x.lower() - y.upper(), x.upper() - y.lower());}
template<class T> inline __host__ __device__
interval_gpu<T> operator*(interval_gpu<T> const &x, interval_gpu<T> const &y)
{return interval_gpu<T>(min(min(x.lower()*y.lower(),x.lower()*y.upper()),
          min(x.upper()*y.lower(),x.upper()*y.upper())),
        max(max(x.lower()*y.lower(),x.lower()*y.upper()),
            max(x.upper()*y.lower(),x.upper()*y.upper())));}
template<class T> inline __host__ __device__
interval_gpu<T> operator/(interval_gpu<T> const &x, interval_gpu<T> const &y)
{return interval_gpu<T>(min(min(x.lower()/y.lower(),x.lower()/y.upper()),
          min(x.upper()/y.lower(),x.upper()/y.upper())),
        max(max(x.lower()/y.lower(),x.lower()/y.upper()),
            max(x.upper()/y.lower(),x.upper()/y.upper())));}



__device__ __forceinline__ int g1(interval_gpu<T> *x){
interval_gpu<T> lmax(12);
interval_gpu<T> f(x[0]*x[0] + x[1]*x[1] - lmax*lmax);
return int(bool(f.upper() < 0) + bool(f.lower() < 0));
}
__device__ __forceinline__ int g2(interval_gpu<T> *x){
interval_gpu<T> l(8);
interval_gpu<T> f(l*l - x[0]*x[0] - x[1]*x[1]);
return int(bool(f.upper() < 0) + bool(f.lower() < 0));
}

__device__ __forceinline__ int g3(interval_gpu<T> *x){
interval_gpu<T> lmax(12);
interval_gpu<T> l0(5);
interval_gpu<T> f((x[0]-l0)*(x[0]-l0) + x[1]*x[1] - lmax*lmax);
return int(bool(f.upper() < 0) + bool(f.lower() < 0));
}
__device__ __forceinline__ int g4(interval_gpu<T> *x){
interval_gpu<T> l(8);
interval_gpu<T> l0(5);
interval_gpu<T> f(l*l  - (x[0]-l0)*(x[0]-l0) - x[1]*x[1]);
return int(bool(f.upper() < 0) + bool(f.lower() < 0));
}


__constant__ int(*dev_func_pp[4])(interval_gpu<T>*) = {&g1,&g2,&g3,&g4};

template<class T>
__global__ void second_grid(int* detail_res,int* corner){
  double x1_low = dev_box[0] + int(corner[0] % dev_threads[0])*(dev_box[1] - dev_box[0])/dev_threads[0];
  double x2_low = dev_box[2] + int(corner[0] / dev_threads[0])*(dev_box[3] - dev_box[2])/dev_blocks[0];
  interval_gpu<T>* x = new interval_gpu<T>[dev_n_of_ints[0]];
  x[0] = interval_gpu<T>(x1_low  +  (threadIdx.x) * ((dev_box[1] - dev_box[0])/dev_threads[0])/blockDim.x,
                         x1_low  +(1+threadIdx.x) * ((dev_box[1] - dev_box[0])/dev_threads[0])/blockDim.x);
  x[1] = interval_gpu<T>(x2_low +   (blockIdx.x) * ((dev_box[3] - dev_box[2])/dev_blocks[0])/gridDim.x,
                         x2_low + (1+blockIdx.x) * ((dev_box[3] - dev_box[2])/dev_blocks[0])/gridDim.x);


  detail_res[(blockIdx.x*blockDim.x + threadIdx.x)] = 1;
  for(int i = 0; i < dev_n_of_func; i++){
    detail_res[(blockIdx.x*blockDim.x + threadIdx.x)] *= (*dev_func_pp[i])(x);
  }
  if((blockIdx.x*blockDim.x + threadIdx.x)==0){
    printf("corner = %d\n",corner[0]);
  }
}


//1 thread to up, in for loop to the end


template<class T>
__global__ void large_grid(int* res){
  interval_gpu<T>* x = new interval_gpu<T>[dev_n_of_ints[0]];
  x[0] = interval_gpu<T>(dev_box[0] +  (threadIdx.x) * (dev_box[1] - dev_box[0])/blockDim.x,
                         dev_box[0] +(1+threadIdx.x) * (dev_box[1] - dev_box[0])/blockDim.x);
  x[1] = interval_gpu<T>(dev_box[2] +   (blockIdx.x) * (dev_box[3] - dev_box[2])/gridDim.x,
                         dev_box[2] + (1+blockIdx.x) * (dev_box[3] - dev_box[2])/gridDim.x);

  res[(blockIdx.x*blockDim.x + threadIdx.x)] = 1;

  for(int i = 0; i < dev_n_of_func; i++){
    res[(blockIdx.x*blockDim.x + threadIdx.x)] *= (*dev_func_pp[i])(x);
  }

  // if( (blockIdx.x*blockDim.x + threadIdx.x) == 2926){printf("[%f:%f]:[%f:%f]\n",
  // dev_box[0] +  (threadIdx.x) * (dev_box[1] - dev_box[0])/blockDim.x,
  // dev_box[0] +(1+threadIdx.x) * (dev_box[1] - dev_box[0])/blockDim.x,
  // dev_box[2] +   (blockIdx.x) * (dev_box[3] - dev_box[2])/gridDim.x,
  // dev_box[2] + (1+blockIdx.x) * (dev_box[3] - dev_box[2])/gridDim.x);}

  // if(res[(blockIdx.x*blockDim.x + threadIdx.x)]%16>0){
  //   //call
  // }
}

//в уточнении нуждаются только граничные ячейки.
//возвращается 2048 индексов номеров крупной сетки
//launch kernell fromkernell cudalaunchkernel

#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

int main(){
    int n_of_ints = 2;

    float host_box[4] = {-15.0,0.0,0.0,7.5};

    int lb = 64;
    int lt = lb*2;
    int * res;
    int * detail_res;
    int*corner;

    //cout<<fixed;
    //cout.precision(4);
    cudaMallocManaged(&corner, sizeof(int));
    cudaMallocManaged(&res, sizeof(int)*lb*lt);
    cudaMallocManaged(&detail_res, sizeof(int)*lb*lb);
    cudaMemcpyToSymbol(dev_n_of_ints, &n_of_ints, sizeof(int));
    cudaMemcpyToSymbol(dev_threads, &lt, sizeof(int));
    cudaMemcpyToSymbol(dev_blocks, &lb, sizeof(int));
    cudaMemcpyToSymbol(dev_box, &host_box, sizeof(float)*4);


    large_grid<T><<<lb, lt>>>(res);
    cudaDeviceSynchronize();
    int counter = 0;
    for(int i = 0; i < lb; i++){
      for(int j = 0; j < lt; j++){
        if(int(res[(i*lt+j)])%16>0){
        interval_gpu<T> xb1(host_box[0] + (j) * (host_box[1] - host_box[0])/lt ,host_box[0]+(1+j) * (host_box[1] - host_box[0])/lt);
        interval_gpu<T> xb2(host_box[2] + (i) * (host_box[3] - host_box[2])/lb ,host_box[2]+(1+i) * (host_box[3] - host_box[2])/lb);
        // cout<<xb1<<":"<<xb2<<"\n";
      }

      if(int(res[(i*lt+j)])%16>0){
        counter++;
        corner[0] = (i*lt+j);//
        // corner[0] = 2926;
        //cout<<corner[0]<<"\n";
        // break;
        // //cout<<"x1_low = "<<((i*lt+j)% lb)*(host_box[1] - host_box[0])/lt<<"\n";
        // //cout<<"x2_low = "<<((i*lt+j)/ lb)*(host_box[3] - host_box[2])/lb<<"\n";
        cout<<"counter = "<<counter<<"\n";
        second_grid<T><<<lb,lb>>>(detail_res,corner);
        CudaCheckError();
        cudaDeviceSynchronize();
        for(int k = 0; k < lb; k++){
          for(int m = 0; m < lb; m++){
            if(int(detail_res[k*lb+m])%16>0){
                double x1_low = host_box[0] + (j) * (host_box[1] - host_box[0])/lt ; //host_box[0]+(1+j) * (host_box[1] - host_box[0])/lt
                double x2_low = host_box[2] + (i) * (host_box[3] - host_box[2])/lb ; //host_box[2]+(1+i) * (host_box[3] - host_box[2])/lb
                interval_gpu<T> x3(x1_low + m*(host_box[1] - host_box[0])/lt/lb,x1_low + (m+1)*(host_box[1] - host_box[0])/lt/lb);
                interval_gpu<T> x4(x2_low + k*(host_box[3] - host_box[2])/lb/lb,x2_low + (k+1)*(host_box[3] - host_box[2])/lb/lb);
                // cout<<x3<<":"<<x4<<"\n";
            }
          detail_res[k*lb+m] = 0;
          }
        }

        cudaDeviceSynchronize();
        // if(counter == 21){i = lb; j = lt; break;}
      }
    }
  }
// cout<<"dick"<<"\n";
//   cudaFree(res);
//   for(int i = 0; i < lb; i++){
//     for(int j = 0; j < lt; j++){
//       if(int(res[(i*lt+j)])%16>0){
//       interval_gpu<T> xb1(host_box[0] + (j) * (host_box[1] - host_box[0])/lt ,host_box[0]+(1+j) * (host_box[1] - host_box[0])/lt);
//       interval_gpu<T> xb2(host_box[2] + (i) * (host_box[3] - host_box[2])/lb ,host_box[2]+(1+i) * (host_box[3] - host_box[2])/lb);
//       cout<<xb1<<":"<<xb2<<"\n";
//     }
//   }
// }

    cudaFree(res);
    cudaFree(detail_res);
    cudaFree(dev_blocks);
    cudaFree(dev_threads);
    cudaFree(corner);
    cudaFree(dev_n_of_ints);
    cudaFree(dev_box);
    return 0;
}
