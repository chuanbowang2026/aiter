# RadixSortSelect one-block 

本文档对应文件：`topk_hip_oneblock.cu`，给出一个**正确性优先、可复现**的 one-block TopK 基准程序。

## 1. 问题与配置

- 输入：`N=50000` 个 `float`
- 输出：Top-`K=2048` 的值和索引
- 线程块：`BLOCK_SIZE=256`
- 基数选择参数：`NumPasses=8`、`BitsPerPass=4`、`NumBuckets=16`

## 2. 核心算法（两阶段）

### 阶段 A：Radix 逐轮锁定第 K 大阈值

1. `preprocess_bits_kernel`  
   将 `float` 映射为可比较的 `uint32_t bits`（`twiddle_float`），避免后续重复转换。
2. 对 `pass=0..7` 循环：
   - `radix_hist_kernel`：统计当前候选在 16 个桶里的数量
   - `radix_scan_choose_kernel`：前缀和选桶，更新
     - `kth_value_bits`（已确定前缀）
     - `num_of_kth_needed`（候选内剩余 rank）

### 阶段 B：`last_filter_kernel` 输出完整 TopK

- `bits > kth_value_bits`：从前往后写入 `out/out_idx`
- `bits == kth_value_bits`：按需要数量从后往前补位
- 最终得到恰好 `K` 个元素（并发下用双计数器保证不冲突）

## 3. 数据流与关键状态

- `data_dev`：原始输入
- `bits_dev`：twiddle 后 key
- `global_hist_dev[16]`：每轮直方图
- `counter_dev`：
  - `kth_value_bits`
  - `num_of_kth_needed`
  - `out_cnt / out_back_cnt`

流程摘要：  
`input -> data_dev -> bits_dev -> (8轮hist+choose) -> last_filter -> out/out_idx`

## 4. 正确性验证方式

程序内置多组 case（随机、全相等、升降序、重复值、特殊值），每组都做：

1. **值集合校验**：  
   CPU 参考与 GPU 输出都按同一 `twiddle` 比较器排序，做位级对比（兼容 NaN/+0/-0）。
2. **索引一致性校验**：  
   检查 `out[i]` 是否与 `input[out_idx[i]]` 完全一致（位级比较）。

最终输出：

- `Final accuracy check: ...`
- `latency: ... us`

## 5. 编译与运行

在本目录执行：

```bash
hipcc topk_hip_opt_oneblock.cu -O3 -o topk_hip_opt_oneblock
./topk_hip_opt_oneblock
```

典型输出：

```text
Final accuracy check: 100.00% (2048/2048)
latency: 4xx.xx us
```

## 5.1 运行环境
 GPU 型号：`AMD Instinct MI308X`（gfx942）
- ROCm 版本：6.4.2
- 驱动版本：6.12.12
- 环境：rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0

## 6. 额外说明

- 本代码实现是“最后全量 `last_filter` 扫描输出”的路线。
- 另一类常用的实现不是等到最后才全量扫一遍，而是每轮把“已确定入选”和“待继续筛选”分流存储，最后再用 last_filter 做边界补齐，效果可能会更好。

## 7. 历史数据记录

- `Final accuracy check: 100.00%`
- `latency: 401.22 us`
- `Final accuracy check: 100.00%`
- `latency: 401.42 us`
- `Final accuracy check: 100.00%`
- `latency: 401.65 us`

补充记录（原multiple-blocks版本，oneblock只改了launch 和直方图）：

- `accuracy: 100%`
- `latency: 63.37 us / 64.16 us / 64.44 us`