#include <iostream>

using namespace std;

#define TYPE float
typedef TYPE T;

__constant__ float dev_box[4];
__constant__ int dev_threads[1];
__constant__ int dev_blocks[1];
__constant__ int dev_n_of_ints[1];
__constant__ int dev_n_of_func[1];

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



__device__ int g1(interval_gpu<T> *x){
interval_gpu<T> lmax(12);
interval_gpu<T> f(x[0]*x[0] + x[1]*x[1] - lmax*lmax);
return int(bool(f.upper() < 0) + bool(f.lower() < 0));
}
__device__ int g2(interval_gpu<T> *x){
interval_gpu<T> l(8);
interval_gpu<T> f(l*l - x[0]*x[0] - x[1]*x[1]);
return int(bool(f.upper() < 0) + bool(f.lower() < 0));
}

__device__ int g3(interval_gpu<T> *x){
interval_gpu<T> lmax(12);
interval_gpu<T> l0(5);
interval_gpu<T> f((x[0]-l0)*(x[0]-l0) + x[1]*x[1] - lmax*lmax);
return int(bool(f.upper() < 0) + bool(f.lower() < 0));
}
__device__ int g4(interval_gpu<T> *x){
interval_gpu<T> l(8);
interval_gpu<T> l0(5);
interval_gpu<T> f(l*l  - (x[0]-l0)*(x[0]-l0) - x[1]*x[1]);
return int(bool(f.upper() < 0) + bool(f.lower() < 0));
}


__constant__ int(*dev_func_pp[4])(interval_gpu<T>*) = {&g1,&g2,&g3,&g4};

template<class T>
__global__ void first_grid(int* res){
  interval_gpu<T>* x = new interval_gpu<T>[dev_n_of_ints[0]];
  x[0] = interval_gpu<T>(dev_box[0] +  (threadIdx.x) * (dev_box[1] - dev_box[0])/64,
                         dev_box[0] +(1+threadIdx.x) * (dev_box[1] - dev_box[0])/64);
  x[1] = interval_gpu<T>(dev_box[2] +   (blockIdx.x) * (dev_box[3] - dev_box[2])/32,
                         dev_box[2] + (1+blockIdx.x) * (dev_box[3] - dev_box[2])/32);


  res[(blockIdx.x*blockDim.x + threadIdx.x)] = 1;

  for(int i = 0; i < dev_n_of_func[0]; i++){
    res[(blockIdx.x*blockDim.x + threadIdx.x)] *= (*dev_func_pp[i])(x);
  }
}


int main(){
    int n_of_ints = 2;
    int n_of_func = 4;

    float host_box[4] = {-15.0,15.0,0.0,15.0};

    int ithreads = 256;
    int iblocks = 128;
    int * res;
    cout<<fixed;
    cout.precision(4);
    cudaMallocManaged(&res, sizeof(int)*iblocks*ithreads);
    cudaMemcpyToSymbol(dev_n_of_ints, &n_of_ints, sizeof(int));
    cudaMemcpyToSymbol(dev_n_of_func, &n_of_func, sizeof(int));
    cudaMemcpyToSymbol(dev_threads, &ithreads, sizeof(int));
    cudaMemcpyToSymbol(dev_blocks, &iblocks, sizeof(int));
    cudaMemcpyToSymbol(dev_box, &host_box, sizeof(float)*4);

    first_grid<T><<<32, 64>>>(res);
    cudaDeviceSynchronize();
      for(int i = 0; i < 32; i++){
        for(int j = 0; j < 64; j++){
        if(int(res[(i*64+j)]) > 0){
                interval_gpu<T> x1(host_box[0] + (j) * (host_box[1] - host_box[0])/64 ,host_box[0]+(1+j) * (host_box[1] - host_box[0])/64);
                interval_gpu<T> x2(host_box[2] + (i) * (host_box[3] - host_box[2])/32 ,host_box[2]+(1+i) * (host_box[3] - host_box[2])/32);
                cout<<x1<<":"<<x2<<"\n";
        }
      }
  }

    cudaFree(res);
    cudaFree(dev_blocks);
    cudaFree(dev_threads);

    return 0;
}
