#include "16_header.h"

// ベンチマーク: fp16_inv を iters 回呼び出してナノ秒を返す
long bench_fp16_inv(int iters){
    fp16_t a, inv;
    fp16_init(&a); fp16_init(&inv);
    fp16_random(&a);

    // ウォームアップ
    fp16_inv(&inv, &a);

    volatile uint32_t sink = 0; // 最適化抑止用
    uint64_t before = fp_mul_count;
    uint64_t before_fp4 = fp4_mul_count;
    struct timespec st, ed;
    clock_gettime(CLOCK_MONOTONIC, &st);
    for(int i=0;i<iters;i++){
        fp16_inv(&inv, &a);
        sink ^= inv.x0.x0.x0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ed);
    uint64_t after = fp_mul_count;
    uint64_t after_fp4 = fp4_mul_count;
    long ns = (ed.tv_sec - st.tv_sec) * 1000000000L + (ed.tv_nsec - st.tv_nsec);
    uint64_t muls = after - before;
    uint64_t muls_fp4 = after_fp4 - before_fp4;

    printf("[Bench] fp16_inv: %d iters -> %ld ns (%.2f ns/op), muls: %llu total, %.2f per op, fp4_mul calls: %llu total, %.2f per op\n",
        iters, ns, (double)ns / iters,
        (unsigned long long)muls, (double)muls / iters,
        (unsigned long long)muls_fp4, (double)muls_fp4 / iters);

    if (sink == 0xFFFFFFFF) { // 実際は起きないが最適化を抑止
        fp16_set(&a, &inv);
    }

    fp16_clear(&a); fp16_clear(&inv);
    return ns;
}

long bench_fp16_inv_karatsuba(int iters){
    fp16_t a, inv;
    fp16_init(&a); fp16_init(&inv);
    fp16_random(&a);

    // ウォームアップ
    fp16_inv_karatsuba(&inv, &a);

    volatile uint32_t sink = 0; // 最適化抑止用
    uint64_t before_fp = fp_mul_count;
    uint64_t before_fp4k = fp4_mul_karatsuba_count;
    struct timespec st, ed;
    clock_gettime(CLOCK_MONOTONIC, &st);
    for(int i=0;i<iters;i++){
        fp16_inv_karatsuba(&inv, &a);
        sink ^= inv.x0.x0.x0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ed);
    uint64_t after_fp = fp_mul_count;
    uint64_t after_fp4k = fp4_mul_karatsuba_count;
    long ns = (ed.tv_sec - st.tv_sec) * 1000000000L + (ed.tv_nsec - st.tv_nsec);
    uint64_t muls_fp = after_fp - before_fp;
    uint64_t muls_fp4k = after_fp4k - before_fp4k;

    printf("[Bench] fp16_inv_karatsuba: %d iters -> %ld ns (%.2f ns/op), fp muls: %llu total, %.2f per op, fp4_mul_karatsuba calls: %llu total, %.2f per op\n",
        iters, ns, (double)ns / iters,
        (unsigned long long)muls_fp, (double)muls_fp / iters,
        (unsigned long long)muls_fp4k, (double)muls_fp4k / iters);

    if (sink == 0xFFFFFFFF) { // 実際は起きないが最適化を抑止
        fp16_set(&a, &inv);
    }

    fp16_clear(&a); fp16_clear(&inv);
    return ns;
}

long bench_fp16_inv_slow(int iters){
    fp16_t a, inv;
    fp16_init(&a); fp16_init(&inv);
    fp16_random(&a);

    // ウォームアップ
    fp16_inv_slow(&inv, &a);

    volatile uint32_t sink = 0; // 最適化抑止用
    uint64_t before = fp_mul_count;
    uint64_t before_fp4 = fp4_mul_count;
    uint64_t before_fp4_slow2 = fp4_mul_slow_count;
    struct timespec st, ed;
    clock_gettime(CLOCK_MONOTONIC, &st);
    for(int i=0;i<iters;i++){
        fp16_inv_slow(&inv, &a);
        sink ^= inv.x0.x0.x0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ed);
    uint64_t after = fp_mul_count;
    uint64_t after_fp4 = fp4_mul_count;
    uint64_t after_fp4_slow2 = fp4_mul_slow_count;
    long ns = (ed.tv_sec - st.tv_sec) * 1000000000L + (ed.tv_nsec - st.tv_nsec);
    uint64_t muls = after - before;
    uint64_t muls_fp4 = after_fp4 - before_fp4;
    uint64_t muls_fp4_slow2 = after_fp4_slow2 - before_fp4_slow2;

    printf("[Bench] fp16_inv_slow: %d iters -> %ld ns (%.2f ns/op), muls: %llu total, %.2f per op, fp4_mul_slow calls: %llu total, %.2f per op\n",
        iters, ns, (double)ns / iters,
        (unsigned long long)muls, (double)muls / iters,
        (unsigned long long)muls_fp4_slow2, (double)muls_fp4_slow2 / iters);

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

    uint64_t before = fp_mul_count;
    struct timespec st, ed;
    clock_gettime(CLOCK_MONOTONIC, &st);
    for(int i=0;i<iters;i++){
        fp4_mul(&acc, &acc, &a);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed);
    uint64_t after = fp_mul_count;
    long ns = (ed.tv_sec - st.tv_sec) * 1000000000L + (ed.tv_nsec - st.tv_nsec);
    uint64_t muls = after - before;

    printf("[Bench] fp4_mul: %d iters -> %ld ns (%.2f ns/op), muls: %llu total, %.2f per op\n",
        iters, ns, (double)ns / iters,
        (unsigned long long)muls, (double)muls / iters);

    fp4_clear(&a); fp4_clear(&b); fp4_clear(&acc);
    return ns;
}

long bench_fp4_mul_karatsuba(int iters){
    fp4_t a,b,acc;
    fp4_init(&a); fp4_init(&b); fp4_init(&acc);
    fp4_random(&a); fp4_random(&b);

    // ウォームアップ
    fp4_mul_karatsuba(&acc, &a, &b);

    uint64_t before_fp = fp_mul_count;
    struct timespec st, ed;
    clock_gettime(CLOCK_MONOTONIC, &st);
    for(int i=0;i<iters;i++){
        fp4_mul_karatsuba(&acc, &acc, &a);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed);
    uint64_t after_fp = fp_mul_count;
    long ns = (ed.tv_sec - st.tv_sec) * 1000000000L + (ed.tv_nsec - st.tv_nsec);
    uint64_t muls = after_fp - before_fp;

    printf("[Bench] fp4_mul_karatsuba: %d iters -> %ld ns (%.2f ns/op), fp muls: %llu total, %.2f per op\n",
        iters, ns, (double)ns / iters,
        (unsigned long long)muls, (double)muls / iters);

    fp4_clear(&a); fp4_clear(&b); fp4_clear(&acc);
    return ns;
}

long bench_fp4_mul_slow(int iters){
    fp4_t a,b,acc;
    fp4_init(&a); fp4_init(&b); fp4_init(&acc);
    fp4_random(&a); fp4_random(&b);

    // ウォームアップ
    fp4_mul_slow(&acc, &a, &b);

    uint64_t before = fp_mul_count;
    struct timespec st, ed;
    clock_gettime(CLOCK_MONOTONIC, &st);
    for(int i=0;i<iters;i++){
        fp4_mul_slow(&acc, &acc, &a);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed);
    uint64_t after = fp_mul_count;
    long ns = (ed.tv_sec - st.tv_sec) * 1000000000L + (ed.tv_nsec - st.tv_nsec);
    uint64_t muls = after - before;

    printf("[Bench] fp4_mul_slow: %d iters -> %ld ns (%.2f ns/op), muls: %llu total, %.2f per op\n",
        iters, ns, (double)ns / iters,
        (unsigned long long)muls, (double)muls / iters);

    fp4_clear(&a); fp4_clear(&b); fp4_clear(&acc);
    return ns;
}
