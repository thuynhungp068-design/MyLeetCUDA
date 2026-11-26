__global__ void sgemm_vec4(float *a, float *b, float *c, int M, int N, int K) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 32;
  __shared__ float s_a[BM][BK]; // 128*32*4 = 16KB, float = 4 bytes
  __shared__ float s_b[BK][BN]; // 32*128*4 = 16KB

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int tid = threadIdx.y * blockDim.x + tx; // 0~1023

  // 加载 A: 每线程加载 s_a[a_m][a_k..a_k+3] 共 4 个元素
  int load_smem_a_m = tid / 8;        // 0~127, 8 线程/行
  int load_smem_a_k = (tid % 8) * 4;  // 0,4,...,28
  // 加载 B: 每线程加载 s_b[b_k][b_n..b_n+3] 共 4 个元素
  int load_smem_b_k = tid / 32;       // 0~31, 32 线程/行
  int load_smem_b_n = (tid % 32) * 4; // 0,4,...,124

  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  // 4×4 Thread Tile 基址（独立于加载映射），这里compute索引的计算逻辑要和
  // load索引的计算逻辑分开，load/compute是可以独立索引的，理解这点很重要。
  // 目标C Tile为[BM,BN]=[128x128], 有32x32线程，则每个线程处理4x4 tile
  // 那么，就可以不重不漏地覆盖[32x4,32x4]=[128x128]的大小
  int comp_smem_a_m_base = (tid / 32) * 4; // 0,4,8,...,124
  int comp_smem_b_n_base = (tid % 32) * 4; // 0,4,8,...,124

  float sum[4][4] = {0.f};
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int load_gmem_a_k = bk * BK + load_smem_a_k; // A [M, K]
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]); // s_a [BM,BK]
    int load_gmem_b_k = bk * BK + load_smem_b_k; // B [K, N]
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]); // s_b [BK,BN]
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      // 每次迭代加载 4 个 A 元素 + 4 个 B 元素，再做 4×4=16 次 FMA
      float a_vals[4] = {s_a[comp_smem_a_m_base + 0][k],
                         s_a[comp_smem_a_m_base + 1][k],
                         s_a[comp_smem_a_m_base + 2][k],
                         s_a[comp_smem_a_m_base + 3][k]};
      float b_vals[4] = {s_b[k][comp_smem_b_n_base + 0],
                         s_b[k][comp_smem_b_n_base + 1],
                         s_b[k][comp_smem_b_n_base + 2],
                         s_b[k][comp_smem_b_n_base + 3]};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          sum[i][j] += a_vals[i] * b_vals[j];
        }
      }
    }
    __syncthreads();
  }

  // 存储 4×4：每行 4 个元素连续 → 可用 float4 store（要求 N 为 4 的倍数以保证对齐）
  int store_gmem_c_m = by * BM + comp_smem_a_m_base;
  int store_gmem_c_n = bx * BN + comp_smem_b_n_base;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int store_gmem_c_addr = (store_gmem_c_m + i) * N + store_gmem_c_n;
    float4 reg_c;
    reg_c.x = sum[i][0];
    reg_c.y = sum[i][1];
    reg_c.z = sum[i][2];
    reg_c.w = sum[i][3];
    FLOAT4(c[store_gmem_c_addr]) = reg_c;
  }
}

// 矩阵形状：A[M][K] × B[K][N] → C[M][N]
// 每个 block 有 32×32 = 1024 个线程（32 个 warp），负责 C 的 128×128 小块。
// BM=128 是输出行数，BN=128 是输出列数，BK=32 是每轮沿 K 处理的元素数。
//
// 每轮 bk 循环的共享内存布局：
//   s_a[BM][BK] = 128×32  个 float，占 16 KiB
//   s_b[BK][BN] = 32×128  个 float，占 16 KiB
//
// 每个线程每轮从 A、B 各搬 4 个连续 float，共搬 8 个 float。
//   A 每行有 32 个 float，需要 32/4=8 个线程：行号 tid/8，起始列 (tid%8)*4。
//   B 每行有 128 个 float，需要 128/4=32 个线程：行号 tid/32，起始列 (tid%32)*4。
//
// 每个线程固定负责 C 的一个 4×4 小块，用 sum[4][4] 跨所有 bk 轮累加。
// 1024 个线程 × 每线程 16 个输出 = 128×128 个输出，整个 K 算完后写回。

// SGEMM Vec4：共享内存分块 + float4 向量化读写 + 4×4 Thread Tile
//
// 【矩阵形状与线程分工】
//   A[M][K] × B[K][N] → C[M][N]，三个矩阵都按行连续存放。
//   K 是完整点积的长度，即 A 的列数、B 的行数。
//   Block = (32, 32, 1)，共 1024 个线程、32 个 warp。
//   BM=128、BN=128：一个 block 负责 C 的一个 128×128 输出 tile。
//   BK=32：每轮 bk 处理完整 K 维中的 32 项。
//   每个线程固定负责 C 中的 4×4 个输出，用 sum[4][4] 跨轮累加。
//
// 【每轮搬运：全局内存 → 共享内存】
//   s_a[BM][BK] = 128×32，存 A 的 128 行和当前 K 段的 32 列。
//   s_b[BK][BN] = 32×128，存 B 当前 K 段的 32 行和 128 列。
//   每块输入 tile 有 4096 个 float，占 16 KiB，两块共占 32 KiB。
//   每个线程从 A、B 各搬一包 float4，共搬 8 个 float。
//   A 和 B 都沿同一行搬 4 个连续元素。
//
//   tid = threadIdx.y * blockDim.x + threadIdx.x。
//   A 每行有 32 个元素，8 个线程各搬 4 个：
//     s_a 的局部行号 = tid / 8，局部起始列 = (tid % 8) * 4。
//   B 每行有 128 个元素，32 个线程各搬 4 个：
//     s_b 的局部行号 = tid / 32，局部起始列 = (tid % 32) * 4。
//
//   全局下标 = 对应 tile 的全局起点 + 线程的局部偏移。
//   A 的行号以 by * BM 为起点，B 的列号以 bx * BN 为起点。
//   这两个起点在整个计算中固定。每轮 bk 用 bk * BK 更新 A 的列
//   和 B 的行，保证两边取的是同一段 K。
//   第一次 __syncthreads() 等两个输入 tile 都搬齐，再开始计算。
//
// 【每轮计算：共享内存 → 线程的 16 个累加值】
//   计算坐标和搬运坐标分别分配。一个线程可以用其他线程搬来的数据。
//   线程的 C 小块起点为 ((tid / 32) * 4, (tid % 32) * 4)，
//   两个坐标都是相对于当前 block 的 C tile。
//   每一步 k，从 s_a 的 4 行各取一个值，从 s_b 的 4 列各取一个值。
//   这 4 个 A 值与 4 个 B 值两两相乘，更新 sum[4][4] 的 16 个位置。
//   sum[i][j] 对应线程小块的第 i 行、第 j 列，跨所有 bk 轮保留。
//   #pragma unroll 展开固定次数的 k、i、j 循环。
//   第二次 __syncthreads() 等所有线程读完，下一轮才能覆盖共享内存。
//
// 【写回】
//   整个 K 算完后，把 block 的全局起点与线程的小块起点相加，
//   得到该线程写 C 的全局行列号。每行用一次 float4 写 4 个值，
//   共写 4 行。1024 个线程各写 16 个值，正好覆盖 128×128 个输出。
//
// 【使用条件】
//   这份实现按完整 tile 工作，要求 M、N 是 128 的倍数，K 是 32 的倍数。
//   float4 访问的起始地址需要按 16 字节对齐。
// 每个线程负责 C 的一个 4×4 输出小块。
// sum[4][4] 在外层 bk 循环之前清零，保存这 16 个输出格子的部分和。
//
// 外层 bk 循环每次搬入一段 K 的数据到共享内存。
// 内层 k 循环遍历这一段里的 32 个位置（BK = 32）。
// 每一步 k，从 A 的 4 行各取一个值，从 B 的 4 列各取一个值，
// 两两相乘，让 sum[4][4] 的 16 个格子各累加一项。
//
// 内层 k 循环结束时，当前 K 切片的 32 项已经累加完毕。
// 下一轮 bk 换一批输入，继续累加同一组 sum[4][4]。
// 例如总 K = 64，第一轮累加第 0～31 项，第二轮累加第 32～63 项。
//
// 外层 bk 循环结束后，每个格子的完整点积才算完。
// 此时把 sum[4][4] 的 16 个结果写回全局内存中对应的 C 位置。