#include "16_header.h"

// ベンチマーク: fp16_inv を iters 回呼び出してナノ秒を返す
long bench_fp16_inv(int iters){
    fp16_t a, inv;
    fp16_init(&a); fp16_init(&inv);
    fp16_random(&a);

    // ウォームアップ
    fp16_inv(&inv, &a);

    volatile uint32_t sink = 0; // 最適化抑止用
    struct timespec st, ed;
    clock_gettime(CLOCK_MONOTONIC, &st);
    for(int i=0;i<iters;i++){
        fp16_inv(&inv, &a);
        sink ^= inv.x0.x0.x0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ed);
    long ns = (ed.tv_sec - st.tv_sec) * 1000000000L + (ed.tv_nsec - st.tv_nsec);

    if (sink == 0xFFFFFFFF) { // 実際は起きないが最適化を抑止
        fp16_set(&a, &inv);
    }

    fp16_clear(&a); fp16_clear(&inv);
    return ns;
}

// ベンチマーク: fp4_mul を iters 回呼び出してナノ秒を返す
long bench_fp4_mul(int iters){
    fp4_t a,b,acc;
    fp4_init(&a); fp4_init(&b); fp4_init(&acc);
    fp4_random(&a); fp4_random(&b);

    // ウォームアップ
    fp4_mul(&acc, &a, &b);

    struct timespec st, ed;
    clock_gettime(CLOCK_MONOTONIC, &st);
    for(int i=0;i<iters;i++){
        fp4_mul(&acc, &acc, &a);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed);
    long ns = (ed.tv_sec - st.tv_sec) * 1000000000L + (ed.tv_nsec - st.tv_nsec);

    fp4_clear(&a); fp4_clear(&b); fp4_clear(&acc);
    return ns;
}
