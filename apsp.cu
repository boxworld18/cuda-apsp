// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#include "cuda_utils.h"

constexpr int THREAD_SIZE = 2;
constexpr int BLOCK_SIZE = 64;
constexpr int SM_SIZE = BLOCK_SIZE * BLOCK_SIZE;
constexpr int MAX_LEN = 100001;

namespace {

__global__ void phase1(int n, int phr, int *graph) {
    __shared__ int res[SM_SIZE];

    int thr_x = THREAD_SIZE * threadIdx.x;
    int thr_y = THREAD_SIZE * threadIdx.y;

    int res_id1 = thr_y * BLOCK_SIZE + thr_x;
    int res_id2 = thr_y * BLOCK_SIZE + thr_x + 1;
    int res_id3 = (thr_y + 1) * BLOCK_SIZE + thr_x;
    int res_id4 = (thr_y + 1) * BLOCK_SIZE + thr_x + 1;

    int abs_x = phr * BLOCK_SIZE + thr_x;
    int abs_y = phr * BLOCK_SIZE + thr_y;
    int abs_id1 = abs_y * n + abs_x;
    int abs_id2 = abs_y * n + abs_x + 1;
    int abs_id3 = (abs_y + 1) * n + abs_x;
    int abs_id4 = (abs_y + 1) * n + abs_x + 1;

    int len1, len2, len3, len4;
    res[res_id1] = len1 = (abs_x < n && abs_y < n) ? graph[abs_id1] : MAX_LEN;
    res[res_id2] = len2 = (abs_x + 1 < n && abs_y < n) ? graph[abs_id2] : MAX_LEN;
    res[res_id3] = len3 = (abs_x < n && abs_y + 1 < n) ? graph[abs_id3] : MAX_LEN;
    res[res_id4] = len4 = (abs_x + 1 < n && abs_y + 1 < n) ? graph[abs_id4] : MAX_LEN;

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
        int yk1 = res[thr_y * BLOCK_SIZE + k];
        int yk2 = res[(thr_y + 1) * BLOCK_SIZE + k];
        int xk1 = res[k * BLOCK_SIZE + thr_x];
        int xk2 = res[k * BLOCK_SIZE + thr_x + 1];
        len1 = min(len1, yk1 + xk1);
        len2 = min(len2, yk1 + xk2);
        len3 = min(len3, yk2 + xk1);
        len4 = min(len4, yk2 + xk2);
        __syncthreads();
        res[res_id1] = len1;
        res[res_id2] = len2;
        res[res_id3] = len3;
        res[res_id4] = len4;
        __syncthreads();
    }

    if (abs_x + 1 < n) {
        if (abs_y + 1 < n) {
            graph[abs_id1] = len1;
            graph[abs_id2] = len2;
            graph[abs_id3] = len3;
            graph[abs_id4] = len4;
        } else if (abs_y < n) {
            graph[abs_id1] = len1;
            graph[abs_id2] = len2;
        }
    } else if (abs_x < n) {
        if (abs_y + 1 < n) {
            graph[abs_id1] = len1;
            graph[abs_id3] = len3;
        } else if (abs_y < n) {
            graph[abs_id1] = len1;
        }
    }
}

__global__ void phase2(int n, int phr, int *graph) {
    if (blockIdx.x == phr) return;
    
    __shared__ int res[SM_SIZE];
    __shared__ int cen[SM_SIZE];
    
    int thr_x = THREAD_SIZE * threadIdx.x;
    int thr_y = THREAD_SIZE * threadIdx.y;

    int res_id1 = thr_y * BLOCK_SIZE + thr_x;
    int res_id2 = thr_y * BLOCK_SIZE + thr_x + 1;
    int res_id3 = (thr_y + 1) * BLOCK_SIZE + thr_x;
    int res_id4 = (thr_y + 1) * BLOCK_SIZE + thr_x + 1;

    // get center block
    int cen_x = phr * BLOCK_SIZE + thr_x;
    int cen_y = phr * BLOCK_SIZE + thr_y;
    int cen_id1 = cen_y * n + cen_x;
    int cen_id2 = cen_y * n + cen_x + 1;
    int cen_id3 = (cen_y + 1) * n + cen_x;
    int cen_id4 = (cen_y + 1) * n + cen_x + 1;

    cen[res_id1] = (cen_x < n && cen_y < n) ? graph[cen_id1] : MAX_LEN;
    cen[res_id2] = (cen_x + 1 < n && cen_y < n) ? graph[cen_id2] : MAX_LEN;
    cen[res_id3] = (cen_x < n && cen_y + 1 < n) ? graph[cen_id3] : MAX_LEN;
    cen[res_id4] = (cen_x + 1 < n && cen_y + 1 < n) ? graph[cen_id4] : MAX_LEN;

    // get own block
    int abs_x, abs_y;
    if (blockIdx.y) {
        abs_x = blockIdx.x * BLOCK_SIZE + thr_x;
        abs_y = phr * BLOCK_SIZE + thr_y;
    } else {
        abs_x = phr * BLOCK_SIZE + thr_x;
        abs_y = blockIdx.x * BLOCK_SIZE + thr_y;
    }

    int abs_id1 = abs_y * n + abs_x;
    int abs_id2 = abs_y * n + abs_x + 1;
    int abs_id3 = (abs_y + 1) * n + abs_x;
    int abs_id4 = (abs_y + 1) * n + abs_x + 1;

    int len1, len2, len3, len4;
    res[res_id1] = len1 = (abs_x < n && abs_y < n) ? graph[abs_id1] : MAX_LEN;
    res[res_id2] = len2 = (abs_x + 1 < n && abs_y < n) ? graph[abs_id2] : MAX_LEN;
    res[res_id3] = len3 = (abs_x < n && abs_y + 1 < n) ? graph[abs_id3] : MAX_LEN;
    res[res_id4] = len4 = (abs_x + 1 < n && abs_y + 1 < n) ? graph[abs_id4] : MAX_LEN;
    __syncthreads();

    if (blockIdx.y) {
        for (int k = 0; k < BLOCK_SIZE; k++) {
            int path1 = cen[thr_y * BLOCK_SIZE + k] + res[k * BLOCK_SIZE + thr_x];
            int path2 = cen[thr_y * BLOCK_SIZE + k] + res[k * BLOCK_SIZE + thr_x + 1];
            int path3 = cen[(thr_y + 1) * BLOCK_SIZE + k] + res[k * BLOCK_SIZE + thr_x];
            int path4 = cen[(thr_y + 1) * BLOCK_SIZE + k] + res[k * BLOCK_SIZE + thr_x + 1];
            len1 = min(len1, path1);
            len2 = min(len2, path2);
            len3 = min(len3, path3);
            len4 = min(len4, path4);
            __syncthreads();
            res[res_id1] = len1;
            res[res_id2] = len2;
            res[res_id3] = len3;
            res[res_id4] = len4;
            __syncthreads();
        }
    } else {
        for (int k = 0; k < BLOCK_SIZE; k++) {
            int path1 = res[thr_y * BLOCK_SIZE + k] + cen[k * BLOCK_SIZE + thr_x];
            int path2 = res[thr_y * BLOCK_SIZE + k] + cen[k * BLOCK_SIZE + thr_x + 1];
            int path3 = res[(thr_y + 1) * BLOCK_SIZE + k] + cen[k * BLOCK_SIZE + thr_x];
            int path4 = res[(thr_y + 1) * BLOCK_SIZE + k] + cen[k * BLOCK_SIZE + thr_x + 1];
            len1 = min(len1, path1);
            len2 = min(len2, path2);
            len3 = min(len3, path3);
            len4 = min(len4, path4);
            __syncthreads();
            res[res_id1] = len1;
            res[res_id2] = len2;
            res[res_id3] = len3;
            res[res_id4] = len4;
            __syncthreads();
        }
    }

    if (abs_x + 1 < n) {
        if (abs_y + 1 < n) {
            graph[abs_id1] = len1;
            graph[abs_id2] = len2;
            graph[abs_id3] = len3;
            graph[abs_id4] = len4;
        } else if (abs_y < n) {
            graph[abs_id1] = len1;
            graph[abs_id2] = len2;
        }
    } else if (abs_x < n) {
        if (abs_y + 1 < n) {
            graph[abs_id1] = len1;
            graph[abs_id3] = len3;
        } else if (abs_y < n) {
            graph[abs_id1] = len1;
        }
    }
}

__global__ void phase3(int n, int phr, int *graph) {
    if (blockIdx.x == phr || blockIdx.y == phr) return;
    
    __shared__ int ver[SM_SIZE];
    __shared__ int hor[SM_SIZE];

    int thr_x = THREAD_SIZE * threadIdx.x;
    int thr_y = THREAD_SIZE * threadIdx.y;

    int res_id1 = thr_y * BLOCK_SIZE + thr_x;
    int res_id2 = thr_y * BLOCK_SIZE + thr_x + 1;
    int res_id3 = (thr_y + 1) * BLOCK_SIZE + thr_x;
    int res_id4 = (thr_y + 1) * BLOCK_SIZE + thr_x + 1;

    // get horizontal block
    int hor_x = phr * BLOCK_SIZE + thr_x;
    int hor_y = blockIdx.y * BLOCK_SIZE + thr_y;
    int hor_id1 = hor_y * n + hor_x;
    int hor_id2 = hor_y * n + hor_x + 1;
    int hor_id3 = (hor_y + 1) * n + hor_x;
    int hor_id4 = (hor_y + 1) * n + hor_x + 1;

    hor[res_id1] = (hor_x < n && hor_y < n) ? graph[hor_id1] : MAX_LEN;
    hor[res_id2] = (hor_x + 1 < n && hor_y < n) ? graph[hor_id2] : MAX_LEN;
    hor[res_id3] = (hor_x < n && hor_y + 1 < n) ? graph[hor_id3] : MAX_LEN;
    hor[res_id4] = (hor_x + 1 < n && hor_y + 1 < n) ? graph[hor_id4] : MAX_LEN;

    // get vertical block
    int ver_x = blockIdx.x * BLOCK_SIZE + thr_x;
    int ver_y = phr * BLOCK_SIZE + thr_y;
    int ver_id1 = ver_y * n + ver_x;
    int ver_id2 = ver_y * n + ver_x + 1;
    int ver_id3 = (ver_y + 1) * n + ver_x;
    int ver_id4 = (ver_y + 1) * n + ver_x + 1;

    ver[res_id1] = (ver_x < n && ver_y < n) ? graph[ver_id1] : MAX_LEN;
    ver[res_id2] = (ver_x + 1 < n && ver_y < n) ? graph[ver_id2] : MAX_LEN;
    ver[res_id3] = (ver_x < n && ver_y + 1 < n) ? graph[ver_id3] : MAX_LEN;
    ver[res_id4] = (ver_x + 1 < n && ver_y + 1 < n) ? graph[ver_id4] : MAX_LEN;

    // get own block
    int abs_x = blockIdx.x * BLOCK_SIZE + thr_x;
    int abs_y = blockIdx.y * BLOCK_SIZE + thr_y;
    int abs_id1 = abs_y * n + abs_x;
    int abs_id2 = abs_y * n + abs_x + 1;
    int abs_id3 = (abs_y + 1) * n + abs_x;
    int abs_id4 = (abs_y + 1) * n + abs_x + 1;

    int len1, len2, len3, len4;
    len1 = (abs_x < n && abs_y < n) ? graph[abs_id1] : MAX_LEN;
    len2 = (abs_x + 1 < n && abs_y < n) ? graph[abs_id2] : MAX_LEN;
    len3 = (abs_x < n && abs_y + 1 < n) ? graph[abs_id3] : MAX_LEN;
    len4 = (abs_x + 1 < n && abs_y + 1 < n) ? graph[abs_id4] : MAX_LEN;

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
        int path1 = hor[thr_y * BLOCK_SIZE + k] + ver[k * BLOCK_SIZE + thr_x];
        int path2 = hor[thr_y * BLOCK_SIZE + k] + ver[k * BLOCK_SIZE + thr_x + 1];
        int path3 = hor[(thr_y + 1) * BLOCK_SIZE + k] + ver[k * BLOCK_SIZE + thr_x];
        int path4 = hor[(thr_y + 1) * BLOCK_SIZE + k] + ver[k * BLOCK_SIZE + thr_x + 1];
        len1 = min(len1, path1);
        len2 = min(len2, path2);
        len3 = min(len3, path3);
        len4 = min(len4, path4);
    }

    if (abs_x + 1 < n) {
        if (abs_y + 1 < n) {
            graph[abs_id1] = len1;
            graph[abs_id2] = len2;
            graph[abs_id3] = len3;
            graph[abs_id4] = len4;
        } else if (abs_y < n) {
            graph[abs_id1] = len1;
            graph[abs_id2] = len2;
        }
    } else if (abs_x < n) {
        if (abs_y + 1 < n) {
            graph[abs_id1] = len1;
            graph[abs_id3] = len3;
        } else if (abs_y < n) {
            graph[abs_id1] = len1;
        }
    }

}

}

void apsp(int n, /* device */ int *graph) {
    int num_blocks = (n - 1) / BLOCK_SIZE + 1;
    for (int phr = 0; phr < num_blocks; phr++) {
        dim3 thr(BLOCK_SIZE / THREAD_SIZE, BLOCK_SIZE / THREAD_SIZE);
        dim3 blk1(1, 1);
        dim3 blk2(num_blocks, 2);
        dim3 blk3(num_blocks, num_blocks + 1);

        phase1<<<blk1, thr>>>(n, phr, graph);
        phase2<<<blk2, thr>>>(n, phr, graph);
        phase3<<<blk3, thr>>>(n, phr, graph);
    }
}