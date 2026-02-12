/**
 * Top-K 完整前 K 个输出 - one-block 版本（中文详注版）
 * 输入: N=50000 FP32, K=2048
 * 流程:
 *   1) 预处理: 将 float 映射到可比较的 uint32_t key（twiddle）
 *   2) 基数选择: 8 轮 * 4bit，逐轮收敛第 K 大值的 bit 前缀
 *   3) 最终过滤: 输出所有 > kth 的值 + 需要数量的 == kth 值，得到恰好 K 个
 */

 #include <hip/hip_runtime.h>
 #include <iostream>
 #include <algorithm>
 #include <cstdlib>
 #include <iomanip>
 #include <cstdint>
 #include <vector>
 #include <cstdio>
 #include <limits>
 #include <cmath>
 #include <random>
 
 #define N 50000
 #define K 2048
 #define BLOCK_SIZE 256
 #define VECTOR_SIZE 4
 
 #define NumPasses 8
 #define NumBuckets 16
 #define BitsPerPass 4
 
 using namespace std;
 
 using u32x4  = __attribute__((__ext_vector_type__(4))) uint32_t;
 
 /**
  * HIP_CHECK:
  * - 用于 main()（返回 int）中的 HIP API 错误检查
  * - 一旦失败，打印错误位置与原因并退出程序
  */
 #define HIP_CHECK(call)                                                       \
   do {                                                                        \
     hipError_t _err = (call);                                                 \
     if (_err != hipSuccess) {                                                 \
       fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,            \
               hipGetErrorString(_err));                                       \
       return 1;                                                               \
     }                                                                         \
   } while (0)
 
 /**
  * HIP_CHECK_BOOL:
  * - 用于返回 bool 的函数/lambda（本文件中主要用于 run_case）
  * - 一旦失败，打印错误并返回 false，方便上层按 case 失败处理
  */
 #define HIP_CHECK_BOOL(call)                                                  \
   do {                                                                        \
     hipError_t _err = (call);                                                 \
     if (_err != hipSuccess) {                                                 \
       fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,            \
               hipGetErrorString(_err));                                       \
       return false;                                                           \
     }                                                                         \
   } while (0)
 
 /**
  * twiddle_float (device):
  * 将 IEEE754 float 映射到 uint32_t，使“无符号整数比较”与“float 数值大小规律”一致。
  *
  * 设计要点:
  * - 正数: 翻转符号位（xor 0x80000000），保持有序
  * - 负数: 全位取反（xor 0xffffffff），使更大的负数映射后也更大
  * - NaN: 统一映射到 0xffffffff，放到序列最前（最大）
  */
 __device__ __forceinline__ uint32_t twiddle_float(float key) {
   uint32_t x = __float_as_uint(key);
   uint32_t mask = (x & 0x80000000u) ? 0xffffffffu : 0x80000000u;
   return (key == key) ? (x ^ mask) : 0xffffffffu;
 }
 
 /**
  * get_start_bit:
  * 给定第 pass 轮，返回该轮从 bit 的哪个起点开始取 4 bit 桶号。
  * 例如 8 轮时依次是: 28, 24, 20, 16, 12, 8, 4, 0
  */
 __device__ __forceinline__ int get_start_bit(int pass) {
   return 32 - (pass + 1) * BitsPerPass;
 }
 
 /**
  * calc_bucket:
  * 从映射后的 32bit key 中提取当前 pass 负责的 4 bit，范围 [0, 15]。
  */
 __device__ __forceinline__ int calc_bucket(uint32_t bits, int pass) {
   int start = get_start_bit(pass);
   return (bits >> start) & (NumBuckets - 1);
 }
 
 /**
  * preprocess_bits_kernel:
  * 预处理 kernel，将输入 float 数组一次性转换到 bits 数组。
  * 后续 kernel 直接消费 bits，避免重复执行 twiddle_float。
  */
 __global__ void preprocess_bits_kernel(const float* __restrict__ data,
                                        uint32_t* __restrict__ bits,
                                        int n) {
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
     bits[i] = twiddle_float(data[i]);
   }
 }
 
 /**
  * bits_of_float (host):
  * 在 CPU 端获取 float 的原始 bit 模式。
  * 用于测试时的“位级比较”，可正确处理 NaN、+0/-0 等特殊值。
  */
 static inline uint32_t bits_of_float(float v) {
   union {
     float f;
     uint32_t u;
   } x;
   x.f = v;
   return x.u;
 }
 
 /**
  * twiddle_float_host:
  * CPU 版本 twiddle，与 device 端逻辑保持一致。
  * 用于构造参考答案时保证排序规则与 GPU 完全一致。
  */
 static inline uint32_t twiddle_float_host(float key) {
   uint32_t x = bits_of_float(key);
   uint32_t mask = (x & 0x80000000u) ? 0xffffffffu : 0x80000000u;
   return (key == key) ? (x ^ mask) : 0xffffffffu;
 }
 
 static inline bool greater_by_twiddle(float a, float b) {
   uint32_t ta = twiddle_float_host(a);
   uint32_t tb = twiddle_float_host(b);
   if (ta != tb) return ta > tb;
   return bits_of_float(a) > bits_of_float(b);
 }
 
 /**
  * Counter:
  * 存放跨 kernel 共享的选择状态与输出计数器。
  * - kth_value_bits: 当前收敛到的“第 K 大值”bit 前缀/最终值
  * - num_of_kth_needed: 还需要补多少个 == kth 的值
  * - out_cnt/out_back_cnt: final filter 阶段的前向与回填计数
  */
 struct Counter {
   uint32_t kth_value_bits;
   int num_of_kth_needed;
   unsigned int out_cnt;
   unsigned int out_back_cnt;
 };
 
 /**
  * reset_kernel:
  * 每轮完整 TopK 前调用，重置 Counter 与全局直方图。
  * 这里用 kernel 清零，避免 host 侧频繁 memset 调度。
  */
 __global__ void reset_kernel(Counter* counter, unsigned int* global_hist) {
   if (threadIdx.x == 0) {
     counter->kth_value_bits = 0;
     counter->num_of_kth_needed = 0;
     counter->out_cnt = 0;
     counter->out_back_cnt = 0;
   }
   if (threadIdx.x < NumBuckets) {
     global_hist[threadIdx.x] = 0;
   }
 }
 
 /**
  * radix_hist_kernel:
  * 统计当前 pass 的桶计数。
  *
  * 关键逻辑:
  * - pass==0: 全量数据参与
  * - pass>0 : 仅统计“高位前缀与当前 kth 前缀相同”的候选集合
  * - one-block 版本中 global_hist 只有一个 block 写入
  *
  * 实现细节:
  * - 先在 shared memory 累计桶计数，最后写回 global_hist
  * - 主体按 u32x4 向量读取，尾部再处理不足 4 的剩余元素
  */
 __global__ __launch_bounds__(BLOCK_SIZE) void radix_hist_kernel(
     const uint32_t* __restrict__ bits_data,
     int n,
     int pass,
     const Counter* __restrict__ counter,
     unsigned int* __restrict__ global_hist) {
   __shared__ unsigned int hist[NumBuckets];
 
   uint32_t kth_bits = counter->kth_value_bits;
   uint32_t prev_mask = 0u;
   if (pass > 0) {
     int prev_start_bit = get_start_bit(pass - 1);
     prev_mask = (0xffffffffu << prev_start_bit);
   }
 
   for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x) hist[i] = 0;
   __syncthreads();
 
   const int num_vecs = n / VECTOR_SIZE;
   const u32x4* in_vec = reinterpret_cast<const u32x4*>(bits_data);
 
   for (int vec_idx = blockIdx.x * blockDim.x + threadIdx.x; vec_idx < num_vecs;
        vec_idx += gridDim.x * blockDim.x) {
     u32x4 v = in_vec[vec_idx];
 #pragma unroll
     for (int j = 0; j < VECTOR_SIZE; j++) {
       uint32_t bits = v[j];
       bool count_me;
       if (pass == 0) {
         count_me = true;
       } else {
         // 非首轮仅统计“已确定高位前缀”一致的候选元素
         count_me = ((bits & prev_mask) == (kth_bits & prev_mask));
       }
       if (count_me) {
         int b = calc_bucket(bits, pass);
         atomicAdd(hist + b, 1u);
       }
     }
   }
 
   int tail_start = num_vecs * VECTOR_SIZE;
   for (int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < n;
        i += gridDim.x * blockDim.x) {
     uint32_t bits = bits_data[i];
     bool count_me;
     if (pass == 0) {
       count_me = true;
     } else {
       count_me = ((bits & prev_mask) == (kth_bits & prev_mask));
     }
     if (count_me) {
       int b = calc_bucket(bits, pass);
       atomicAdd(hist + b, 1u);
     }
   }
   __syncthreads();
 
   for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x) {
     global_hist[i] = hist[i];
   }
 }
 
 /**
  * radix_scan_choose_kernel:
  * 对 16 个桶做前缀和，并据此更新第 K 大值在当前 pass 的 bit。
  *
  * 思路:
  * - hist 做前缀和后，hist[b] 表示 <= b 的元素数量（在候选集合内）
  * - 从小桶到大桶查找满足“落入第 K 大区间”的桶 b
  * - 将 b 写入 kth_value_bits 对应的 4 bit
  * - 更新下一轮需要追踪的剩余 rank（num_of_kth_needed）
  */
 __global__ void radix_scan_choose_kernel(
     int pass,
     int k,
     Counter* __restrict__ counter,
     unsigned int* __restrict__ global_hist) {
   __shared__ unsigned int hist[NumBuckets];
 
   int start_bit = get_start_bit(pass);
 
   for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x) hist[i] = global_hist[i];
   __syncthreads();
 
   if (threadIdx.x == 0) {
     for (int i = 1; i < NumBuckets; i++) hist[i] += hist[i - 1];
   }
   __syncthreads();
 
   if (threadIdx.x == 0) {
     const uint32_t bucket_mask = ((1u << BitsPerPass) - 1u) << start_bit;
     uint32_t kth_bits = (pass == 0) ? 0u : counter->kth_value_bits;
     int k_remaining = (pass == 0) ? k : counter->num_of_kth_needed;
     unsigned int total = hist[NumBuckets - 1];
     for (int b = 0; b < NumBuckets; b++) {
       if ((int)total - k_remaining < (int)hist[b]) {
         kth_bits = (kth_bits & ~bucket_mask) | ((uint32_t)b << start_bit);
         k_remaining = k_remaining - (int)(total - hist[b]);
         break;
       }
     }
     counter->kth_value_bits = kth_bits;
     counter->num_of_kth_needed = k_remaining;
     if (pass == NumPasses - 1) {
       counter->out_cnt = 0;
       counter->out_back_cnt = 0;
     }
   }
 
   __syncthreads();
   if (pass < NumPasses - 1) {
     for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x) global_hist[i] = 0;
   }
 }
 
 /**
  * last_filter_kernel:
  * 根据最终 kth_value_bits 输出前 K 个元素。
  *
  * 策略:
  * - bits > kth: 一定属于 top-k，按 out_cnt 从前往后写
  * - bits == kth: 只取 num_of_kth_needed 个，按 out_back_cnt 从后往前补位
  * 这样可在并发场景下稳定填满恰好 K 个位置。
  */
 __global__ __launch_bounds__(BLOCK_SIZE) void last_filter_kernel(
     const float* __restrict__ in,
     const uint32_t* __restrict__ bits_data,
     float* __restrict__ out,
     unsigned int* __restrict__ out_idx,
     int n,
     int k,
     Counter* __restrict__ counter) {
   uint32_t kth_value_bits = counter->kth_value_bits;
   int num_of_kth_needed = counter->num_of_kth_needed;
   unsigned int* p_out_cnt = &counter->out_cnt;
   unsigned int* p_out_back = &counter->out_back_cnt;
 
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
     float value = in[i];
     uint32_t bits = bits_data[i];
     if (bits > kth_value_bits) {
       unsigned int pos = atomicAdd(p_out_cnt, 1u);
       if (pos < (unsigned int)k) {
         out[pos] = value;
         out_idx[pos] = (unsigned int)i;
       }
     } else if (bits == kth_value_bits) {
       // 双计数器前后夹逼写入，保证并发下恰好填满 K 个
       unsigned int back_pos = atomicAdd(p_out_back, 1u);
       if (back_pos < (unsigned int)num_of_kth_needed) {
         unsigned int pos = (unsigned int)k - 1 - back_pos;
         out[pos] = value;
         out_idx[pos] = (unsigned int)i;
       }
     }
   }
 }
 
 /**
  * main:
  * 程序入口，包含:
  * - GPU 资源申请与释放
  * - one-block TopK 执行函数 run_all
  * - 多组测试用例 run_case（正确性 + 可选性能）
  * - 汇总输出
  */
 int main() {
   float* out = new float[K];
   unsigned int* out_idx = new unsigned int[K];
   std::vector<float> host_input(N, 0.0f);
   std::vector<float> last_input(N, 0.0f);
   float measured_latency_us = 0.0f;
 
   float *data_dev, *out_dev;
   uint32_t* bits_dev;
   unsigned int* out_idx_dev;
   Counter* counter_dev;
   unsigned int* global_hist_dev;
 
   HIP_CHECK(hipMalloc((void**)&data_dev, N * sizeof(float)));
   HIP_CHECK(hipMalloc((void**)&bits_dev, N * sizeof(uint32_t)));
   HIP_CHECK(hipMalloc((void**)&out_dev, K * sizeof(float)));
   HIP_CHECK(hipMalloc((void**)&out_idx_dev, K * sizeof(unsigned int)));
   HIP_CHECK(hipMalloc((void**)&counter_dev, sizeof(Counter)));
   HIP_CHECK(hipMalloc((void**)&global_hist_dev, NumBuckets * sizeof(unsigned int)));
 
   /**
    * run_all:
    * 对当前 data_dev 执行一次完整 one-block TopK。
    * 调用方负责在外部做同步与结果拷贝。
    */
   auto run_all = [&]() {
     hipLaunchKernelGGL(reset_kernel, 1, NumBuckets, 0, 0, counter_dev, global_hist_dev);
     HIP_CHECK_BOOL(hipGetLastError());
     for (int pass = 0; pass < NumPasses; pass++) {
       hipLaunchKernelGGL(radix_hist_kernel, 1, BLOCK_SIZE, 0, 0,
                          bits_dev, N, pass, counter_dev, global_hist_dev);
       HIP_CHECK_BOOL(hipGetLastError());
       hipLaunchKernelGGL(radix_scan_choose_kernel, 1, BLOCK_SIZE, 0, 0,
                          pass, K, counter_dev, global_hist_dev);
       HIP_CHECK_BOOL(hipGetLastError());
     }
     hipLaunchKernelGGL(last_filter_kernel, 1, BLOCK_SIZE, 0, 0,
                        data_dev, bits_dev, out_dev, out_idx_dev, N, K, counter_dev);
     HIP_CHECK_BOOL(hipGetLastError());
     return true;
   };
 
   /**
    * run_case:
    * 执行单个测试用例。
    *
    * 参数:
    * - name: 用例名称
    * - input: 输入数据
    * - do_bench: 是否执行基准计时
    *
    * 返回:
    * - true: 该 case 正确通过
    * - false: HIP 调用失败或结果不匹配
    */
   auto run_case = [&](const std::vector<float>& input, bool do_bench) -> bool {
     last_input = input;
     HIP_CHECK_BOOL(hipMemcpy(data_dev, input.data(), N * sizeof(float), hipMemcpyHostToDevice));
     hipLaunchKernelGGL(preprocess_bits_kernel, 1, BLOCK_SIZE, 0, 0, data_dev, bits_dev, N);
     HIP_CHECK_BOOL(hipGetLastError());
 
     // 执行并取回结果
     if (!run_all()) return false;
     HIP_CHECK_BOOL(hipDeviceSynchronize());
     HIP_CHECK_BOOL(hipMemcpy(out, out_dev, K * sizeof(float), hipMemcpyDeviceToHost));
     HIP_CHECK_BOOL(hipMemcpy(out_idx, out_idx_dev, K * sizeof(unsigned int), hipMemcpyDeviceToHost));
 
     // 参考答案（CPU）按与 GPU 完全一致的 twiddle 规则排序
     std::vector<float> ref_sorted(input.begin(), input.end());
     std::sort(ref_sorted.begin(), ref_sorted.end(), greater_by_twiddle);
 
     // GPU 输出也按同规则排序，避免输出顺序差异导致误判
     std::vector<float> out_sorted(out, out + K);
     std::sort(out_sorted.begin(), out_sorted.end(), greater_by_twiddle);
 
     // 采用“位级相等”比较，准确处理 NaN、+0/-0
     int correct = 0;
     for (int i = 0; i < K; i++) {
       if (bits_of_float(out_sorted[i]) == bits_of_float(ref_sorted[i])) correct++;
     }
     bool pass = (correct == K);
     if (!pass) return false;
 
     int idx_match = 0;
     for (int i = 0; i < K; i++) {
       unsigned int idx = out_idx[i];
       if (idx < (unsigned int)N &&
           bits_of_float(out[i]) == bits_of_float(input[idx])) {
         idx_match++;
       }
     }
     bool idx_pass = (idx_match == K);
     if (!idx_pass) return false;
 
     if (!do_bench) return true;
 
     // 可选性能测试: 先预热再计时
     for (int i = 0; i < 100; i++) run_all();
     HIP_CHECK_BOOL(hipDeviceSynchronize());
 
     hipEvent_t start, stop;
     HIP_CHECK_BOOL(hipEventCreate(&start));
     HIP_CHECK_BOOL(hipEventCreate(&stop));
     const int iterations = 200;
     HIP_CHECK_BOOL(hipEventRecord(start));
     for (int iter = 0; iter < iterations; iter++) {
       if (!run_all()) return false;
     }
     HIP_CHECK_BOOL(hipEventRecord(stop));
     HIP_CHECK_BOOL(hipEventSynchronize(stop));
     float ms = 0.0f;
     HIP_CHECK_BOOL(hipEventElapsedTime(&ms, start, stop));
     HIP_CHECK_BOOL(hipEventDestroy(start));
     HIP_CHECK_BOOL(hipEventDestroy(stop));
 
     measured_latency_us = (ms * 1000.0f) / iterations;
     return true;
   };
 
   /**
    * fill_uniform:
    * 生成 [lo, hi] 的均匀随机输入，使用固定种子确保可复现。
    */
   auto fill_uniform = [&](uint32_t seed, float lo, float hi) {
     std::mt19937 gen(seed);
     std::uniform_real_distribution<float> dist(lo, hi);
     for (int i = 0; i < N; i++) host_input[i] = dist(gen);
   };
 
   // case 1: 均匀随机（带基准）
   fill_uniform(123, -50.0f, 50.0f);
   if (!run_case(host_input, true)) return 1;
 
   // case 2: 全相等
   for (int i = 0; i < N; i++) host_input[i] = 3.1415926f;
   if (!run_case(host_input, false)) return 1;
 
   // case 3: 升序
   for (int i = 0; i < N; i++) host_input[i] = (float)i * 0.001f - 25.0f;
   if (!run_case(host_input, false)) return 1;
 
   // case 4: 降序
   for (int i = 0; i < N; i++) host_input[i] = (float)(N - i) * 0.001f - 25.0f;
   if (!run_case(host_input, false)) return 1;
 
   // case 5: 大量重复值
   for (int i = 0; i < N; i++) host_input[i] = (float)(i % 17) - 8.0f;
   if (!run_case(host_input, false)) return 1;
 
   // case 6: 特殊值（NaN/Inf/+0/-0/极值/次正规）
   fill_uniform(7, -20.0f, 20.0f);
   host_input[0] = std::numeric_limits<float>::infinity();
   host_input[1] = -std::numeric_limits<float>::infinity();
   host_input[2] = std::numeric_limits<float>::quiet_NaN();
   host_input[3] = -0.0f;
   host_input[4] = 0.0f;
   host_input[5] = std::numeric_limits<float>::max();
   host_input[6] = -std::numeric_limits<float>::max();
   host_input[7] = std::numeric_limits<float>::denorm_min();
   if (!run_case(host_input, false)) return 1;
 
   // 额外做一次最终一致性检查
   HIP_CHECK(hipDeviceSynchronize());
 
   std::vector<float> ref_sorted(last_input.begin(), last_input.end());
   std::sort(ref_sorted.begin(), ref_sorted.end(), greater_by_twiddle);
 
   int correct = 0;
   std::vector<float> out_sorted(out, out + K);
   std::sort(out_sorted.begin(), out_sorted.end(), greater_by_twiddle);
   for (int i = 0; i < K; i++) {
     if (bits_of_float(out_sorted[i]) == bits_of_float(ref_sorted[i])) correct++;
   }
   float accuracy = (float)correct / K * 100.0f;
   cout << fixed << setprecision(2);
   cout << "Final accuracy check: " << accuracy << "% (" << correct << "/" << K << ")" << endl;
   cout << "latency: " << measured_latency_us << " us" << endl;
 
   // 资源释放
   delete[] out;
   delete[] out_idx;
   HIP_CHECK(hipFree(data_dev));
   HIP_CHECK(hipFree(bits_dev));
   HIP_CHECK(hipFree(out_dev));
   HIP_CHECK(hipFree(out_idx_dev));
   HIP_CHECK(hipFree(counter_dev));
   HIP_CHECK(hipFree(global_hist_dev));
   return 0;
 }
 