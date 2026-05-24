#include "16_header.h"

typedef struct {
    uint64_t muls;
    uint64_t adds;
    uint64_t subs;
    uint64_t fp4;
    uint64_t fp4k;
    uint64_t fp4slow;
} bench_counts_t;

static fp_t bench_ab_a;
static fp_t bench_ab_b;

#define BENCH_UNROLL 8

static void bench_fp_init_ab(void){
    fp_random(&bench_ab_a);
    fp_random(&bench_ab_b);
}

// ベンチマーク: fp16_inv を iters 回呼び出してナノ秒を返す
static long bench_fp16_inv_ns(int iters, bench_counts_t *counts){
    fp16_t a, inv;
    fp16_init(&a); fp16_init(&inv);
    fp16_random(&a);

    // ウォームアップ
    fp16_inv(&inv, &a);

    volatile uint32_t sink = 0; // 最適化抑止用
    uint64_t before = fp_mul_count;
    uint64_t before_add = fp_add_count;
    uint64_t before_sub = fp_sub_count;
    uint64_t before_fp4 = fp4_mul_count;
    struct timespec st1, ed1, st2, ed2;
    clock_gettime(CLOCK_MONOTONIC, &st1);
    for(int i=0;i<iters;i++){
        fp16_random(&a);
        fp16_inv(&inv, &a);
        sink ^= inv.x0.x0.x0; // prevent dead-code elimination
    }
    clock_gettime(CLOCK_MONOTONIC, &ed1);
    uint64_t after = fp_mul_count;
    uint64_t after_add = fp_add_count;
    uint64_t after_sub = fp_sub_count;
    uint64_t after_fp4 = fp4_mul_count;

    clock_gettime(CLOCK_MONOTONIC, &st2);
    for(int i=0;i<iters;i++){
        fp16_random(&a);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed2);
    long ns_ops = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);
    long ns_init = (ed2.tv_sec - st2.tv_sec) * 1000000000L + (ed2.tv_nsec - st2.tv_nsec);
    long ns = ns_ops - ns_init;
    if (ns < 0) ns = 0;
    uint64_t muls = after - before;
    uint64_t adds = after_add - before_add;
    uint64_t subs = after_sub - before_sub;
    uint64_t muls_fp4 = after_fp4 - before_fp4;

    if (counts) {
        counts->muls = muls;
        counts->adds = adds;
        counts->subs = subs;
        counts->fp4 = muls_fp4;
    }

    if (sink == 0xFFFFFFFF) { // 実際は起きないが最適化を抑止
        fp16_set(&a, &inv);
    }

    fp16_clear(&a); fp16_clear(&inv);
    return ns;
}

static long bench_fp16_inv_run_ns(int iters, uint64_t *muls, uint64_t *adds, uint64_t *subs, uint64_t *fp4s){
    bench_counts_t counts = {0};
    long ns = bench_fp16_inv_ns(iters, &counts);
    if (muls) *muls = counts.muls;
    if (adds) *adds = counts.adds;
    if (subs) *subs = counts.subs;
    if (fp4s) *fp4s = counts.fp4;
    return ns;
}

long bench_fp16_inv_avg(int iters, int count){
    long long total_ns = 0;
    long long total_ops = (long long)iters * (long long)count;
    unsigned long long total_muls = 0;
    unsigned long long total_adds = 0;
    unsigned long long total_subs = 0;
    unsigned long long total_fp4 = 0;
    for (int i = 0; i < count; i++) {
        uint64_t muls = 0, adds = 0, subs = 0, fp4s = 0;
        total_ns += bench_fp16_inv_run_ns(iters, &muls, &adds, &subs, &fp4s);
        total_muls += muls;
        total_adds += adds;
        total_subs += subs;
        total_fp4 += fp4s;
    }
    printf("[Bench] fp16_inv: %d cases x %d iters, %.2f ns/op, muls/op: %.2f, adds/op: %.2f, subs/op: %.2f, fp4_mul/op: %.2f\n",
        count, iters, (double)total_ns / (double)total_ops,
        (double)total_muls / (double)total_ops,
        (double)total_adds / (double)total_ops,
        (double)total_subs / (double)total_ops,
        (double)total_fp4 / (double)total_ops);
    return (long)(total_ns / count);
}

static long bench_fp16_inv_new_ns(int iters, bench_counts_t *counts){
    fp16_t a, inv;
    fp16_init(&a); fp16_init(&inv);
    fp16_random(&a);

    // ウォームアップ
    fp16_inv_new(&inv, &a);

    volatile uint32_t sink = 0; // 最適化抑止用
    uint64_t before = fp_mul_count;
    uint64_t before_add = fp_add_count;
    uint64_t before_sub = fp_sub_count;
    uint64_t before_fp4 = fp4_mul_count;
    struct timespec st1, ed1, st2, ed2;
    clock_gettime(CLOCK_MONOTONIC, &st1);
    for (int i = 0; i < iters; i++) {
        fp16_random(&a);
        fp16_inv_new(&inv, &a);
        sink ^= inv.x0.x0.x0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ed1);
    uint64_t after = fp_mul_count;
    uint64_t after_add = fp_add_count;
    uint64_t after_sub = fp_sub_count;
    uint64_t after_fp4 = fp4_mul_count;

    clock_gettime(CLOCK_MONOTONIC, &st2);
    for (int i = 0; i < iters; i++) {
        fp16_random(&a);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed2);
    long ns_ops = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);
    long ns_init = (ed2.tv_sec - st2.tv_sec) * 1000000000L + (ed2.tv_nsec - st2.tv_nsec);
    long ns = ns_ops - ns_init;
    if (ns < 0) ns = 0;
    uint64_t muls = after - before;
    uint64_t adds = after_add - before_add;
    uint64_t subs = after_sub - before_sub;
    uint64_t muls_fp4 = after_fp4 - before_fp4;

    if (counts) {
        counts->muls = muls;
        counts->adds = adds;
        counts->subs = subs;
        counts->fp4 = muls_fp4;
    }

    if (sink == 0xFFFFFFFF) { // 実際は起きないが最適化を抑止
        fp16_set(&a, &inv);
    }

    fp16_clear(&a); fp16_clear(&inv);
    return ns;
}

static long bench_fp16_inv_new_run_ns(int iters, uint64_t *muls, uint64_t *adds, uint64_t *subs, uint64_t *fp4s){
    bench_counts_t counts = {0};
    long ns = bench_fp16_inv_new_ns(iters, &counts);
    if (muls) *muls = counts.muls;
    if (adds) *adds = counts.adds;
    if (subs) *subs = counts.subs;
    if (fp4s) *fp4s = counts.fp4;
    return ns;
}

long bench_fp16_inv_new_avg(int iters, int count){
    long long total_ns = 0;
    long long total_ops = (long long)iters * (long long)count;
    unsigned long long total_muls = 0;
    unsigned long long total_adds = 0;
    unsigned long long total_subs = 0;
    unsigned long long total_fp4 = 0;
    for (int i = 0; i < count; i++) {
        uint64_t muls = 0, adds = 0, subs = 0, fp4s = 0;
        total_ns += bench_fp16_inv_new_run_ns(iters, &muls, &adds, &subs, &fp4s);
        total_muls += muls;
        total_adds += adds;
        total_subs += subs;
        total_fp4 += fp4s;
    }
    printf("[Bench] fp16_inv_new: %d cases x %d iters, %.2f ns/op, muls/op: %.2f, adds/op: %.2f, subs/op: %.2f, fp4_mul/op: %.2f\n",
        count, iters, (double)total_ns / (double)total_ops,
        (double)total_muls / (double)total_ops,
        (double)total_adds / (double)total_ops,
        (double)total_subs / (double)total_ops,
        (double)total_fp4 / (double)total_ops);
    return (long)(total_ns / count);
}

static long bench_fp16_inv_karatsuba_ns(int iters, bench_counts_t *counts){
    fp16_t a, inv;
    fp16_init(&a); fp16_init(&inv);
    fp16_random(&a);

    // ウォームアップ
    fp16_inv_karatsuba(&inv, &a);

    volatile uint32_t sink = 0; // 最適化抑止用
    uint64_t before_fp = fp_mul_count;
    uint64_t before_add = fp_add_count;
    uint64_t before_sub = fp_sub_count;
    uint64_t before_fp4k = fp4_mul_karatsuba_count;
    struct timespec st1, ed1, st2, ed2;
    clock_gettime(CLOCK_MONOTONIC, &st1);
    for(int i=0;i<iters;i++){
        fp16_random(&a);
        fp16_inv_karatsuba(&inv, &a);
        sink ^= inv.x0.x0.x0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ed1);
    uint64_t after_fp = fp_mul_count;
    uint64_t after_add = fp_add_count;
    uint64_t after_sub = fp_sub_count;
    uint64_t after_fp4k = fp4_mul_karatsuba_count;

    clock_gettime(CLOCK_MONOTONIC, &st2);
    for(int i=0;i<iters;i++){
        fp16_random(&a);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed2);
    long ns_ops = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);
    long ns_init = (ed2.tv_sec - st2.tv_sec) * 1000000000L + (ed2.tv_nsec - st2.tv_nsec);
    long ns = ns_ops - ns_init;
    if (ns < 0) ns = 0;
    uint64_t muls_fp = after_fp - before_fp;
    uint64_t adds = after_add - before_add;
    uint64_t subs = after_sub - before_sub;
    uint64_t muls_fp4k = after_fp4k - before_fp4k;

    if (counts) {
        counts->muls = muls_fp;
        counts->adds = adds;
        counts->subs = subs;
        counts->fp4k = muls_fp4k;
    }

    if (sink == 0xFFFFFFFF) { // 実際は起きないが最適化を抑止
        fp16_set(&a, &inv);
    }

    fp16_clear(&a); fp16_clear(&inv);
    return ns;
}

static long bench_fp16_inv_karatsuba_run_ns(int iters, uint64_t *muls, uint64_t *adds, uint64_t *subs, uint64_t *fp4ks){
    bench_counts_t counts = {0};
    long ns = bench_fp16_inv_karatsuba_ns(iters, &counts);
    if (muls) *muls = counts.muls;
    if (adds) *adds = counts.adds;
    if (subs) *subs = counts.subs;
    if (fp4ks) *fp4ks = counts.fp4k;
    return ns;
}

long bench_fp16_inv_karatsuba_avg(int iters, int count){
    long long total_ns = 0;
    long long total_ops = (long long)iters * (long long)count;
    unsigned long long total_muls = 0;
    unsigned long long total_adds = 0;
    unsigned long long total_subs = 0;
    unsigned long long total_fp4k = 0;
    for (int i = 0; i < count; i++) {
        uint64_t muls = 0, adds = 0, subs = 0, fp4ks = 0;
        total_ns += bench_fp16_inv_karatsuba_run_ns(iters, &muls, &adds, &subs, &fp4ks);
        total_muls += muls;
        total_adds += adds;
        total_subs += subs;
        total_fp4k += fp4ks;
    }
    printf("[Bench] fp16_inv_karatsuba: %d cases x %d iters, %.2f ns/op, fp muls/op: %.2f, adds/op: %.2f, subs/op: %.2f, fp4_mul_karatsuba/op: %.2f\n",
        count, iters, (double)total_ns / (double)total_ops,
        (double)total_muls / (double)total_ops,
        (double)total_adds / (double)total_ops,
        (double)total_subs / (double)total_ops,
        (double)total_fp4k / (double)total_ops);
    return (long)(total_ns / count);
}

static long bench_fp16_inv_slow_ns(int iters, bench_counts_t *counts){
    fp16_t a, inv;
    fp16_init(&a); fp16_init(&inv);
    fp16_random(&a);

    // ウォームアップ
    fp16_inv_slow(&inv, &a);

    volatile uint32_t sink = 0; // 最適化抑止用
    uint64_t before = fp_mul_count;
    uint64_t before_add = fp_add_count;
    uint64_t before_sub = fp_sub_count;
    uint64_t before_fp4 = fp4_mul_count;
    uint64_t before_fp4_slow2 = fp4_mul_slow_count;
    struct timespec st1, ed1, st2, ed2;
    clock_gettime(CLOCK_MONOTONIC, &st1);
    for(int i=0;i<iters;i++){
        fp16_random(&a);
        fp16_inv_slow(&inv, &a);
        sink ^= inv.x0.x0.x0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ed1);
    uint64_t after = fp_mul_count;
    uint64_t after_add = fp_add_count;
    uint64_t after_sub = fp_sub_count;
    uint64_t after_fp4 = fp4_mul_count;
    uint64_t after_fp4_slow2 = fp4_mul_slow_count;

    clock_gettime(CLOCK_MONOTONIC, &st2);
    for(int i=0;i<iters;i++){
        fp16_random(&a);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed2);
    long ns_ops = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);
    long ns_init = (ed2.tv_sec - st2.tv_sec) * 1000000000L + (ed2.tv_nsec - st2.tv_nsec);
    long ns = ns_ops - ns_init;
    if (ns < 0) ns = 0;
    uint64_t muls = after - before;
    uint64_t adds = after_add - before_add;
    uint64_t subs = after_sub - before_sub;
    uint64_t muls_fp4 = after_fp4 - before_fp4;
    uint64_t muls_fp4_slow2 = after_fp4_slow2 - before_fp4_slow2;

    if (counts) {
        counts->muls = muls;
        counts->adds = adds;
        counts->subs = subs;
        counts->fp4 = muls_fp4;
        counts->fp4slow = muls_fp4_slow2;
    }

    if (sink == 0xFFFFFFFF) { // 実際は起きないが最適化を抑止
        fp16_set(&a, &inv);
    }

    fp16_clear(&a); fp16_clear(&inv);
    return ns;
}

static long bench_fp16_inv_slow_run_ns(int iters, uint64_t *muls, uint64_t *adds, uint64_t *subs, uint64_t *fp4slows){
    bench_counts_t counts = {0};
    long ns = bench_fp16_inv_slow_ns(iters, &counts);
    if (muls) *muls = counts.muls;
    if (adds) *adds = counts.adds;
    if (subs) *subs = counts.subs;
    if (fp4slows) *fp4slows = counts.fp4slow;
    return ns;
}

long bench_fp16_inv_slow_avg(int iters, int count){
    long long total_ns = 0;
    long long total_ops = (long long)iters * (long long)count;
    unsigned long long total_muls = 0;
    unsigned long long total_adds = 0;
    unsigned long long total_subs = 0;
    unsigned long long total_fp4slow = 0;
    for (int i = 0; i < count; i++) {
        uint64_t muls = 0, adds = 0, subs = 0, fp4slows = 0;
        total_ns += bench_fp16_inv_slow_run_ns(iters, &muls, &adds, &subs, &fp4slows);
        total_muls += muls;
        total_adds += adds;
        total_subs += subs;
        total_fp4slow += fp4slows;
    }
    printf("[Bench] fp16_inv_slow: %d cases x %d iters, %.2f ns/op, muls/op: %.2f, adds/op: %.2f, subs/op: %.2f, fp4_mul_slow/op: %.2f\n",
        count, iters, (double)total_ns / (double)total_ops,
        (double)total_muls / (double)total_ops,
        (double)total_adds / (double)total_ops,
        (double)total_subs / (double)total_ops,
        (double)total_fp4slow / (double)total_ops);
    return (long)(total_ns / count);
}

// ベンチマーク: fp4_mul を iters 回呼び出してナノ秒を返す
static long bench_fp4_mul_run_ns(int iters, uint64_t *muls, uint64_t *adds, uint64_t *subs){
    fp4_t a, b, acc;
    fp4_init(&a); fp4_init(&b); fp4_init(&acc);
    fp4_random(&a); fp4_random(&b);

    // ウォームアップ
    fp4_mul(&acc, &a, &b);

    uint64_t before_mul = fp_mul_count;
    uint64_t before_add = fp_add_count;
    uint64_t before_sub = fp_sub_count;
    struct timespec st1, ed1, st2, ed2;
    clock_gettime(CLOCK_MONOTONIC, &st1);
    for (int i = 0; i < iters; i++) {
        fp4_random(&a); fp4_random(&b);
        fp4_mul(&acc, &a, &b);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed1);
    uint64_t after_mul = fp_mul_count;
    uint64_t after_add = fp_add_count;
    uint64_t after_sub = fp_sub_count;
    long ns_ops = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);

    clock_gettime(CLOCK_MONOTONIC, &st2);
    for (int i = 0; i < iters; i++) {
        fp4_random(&a); fp4_random(&b);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed2);
    long ns_init = (ed2.tv_sec - st2.tv_sec) * 1000000000L + (ed2.tv_nsec - st2.tv_nsec);
    long ns = ns_ops - ns_init;
    if (ns < 0) ns = 0;

    if (muls) *muls = after_mul - before_mul;
    if (adds) *adds = after_add - before_add;
    if (subs) *subs = after_sub - before_sub;

    fp4_clear(&a); fp4_clear(&b); fp4_clear(&acc);
    return ns;
}

long bench_fp4_mul_avg(int iters, int count){
    long long total_ns = 0;
    long long total_ops = (long long)iters * (long long)count;
    unsigned long long total_muls = 0;
    unsigned long long total_adds = 0;
    unsigned long long total_subs = 0;
    for (int i = 0; i < count; i++) {
        uint64_t muls = 0, adds = 0, subs = 0;
        total_ns += bench_fp4_mul_run_ns(iters, &muls, &adds, &subs);
        total_muls += muls;
        total_adds += adds;
        total_subs += subs;
    }
    printf("[Bench] fp4_mul: %d cases x %d iters, %.2f ns/op, muls/op: %.2f, adds/op: %.2f, subs/op: %.2f\n",
        count, iters, (double)total_ns / (double)total_ops,
        (double)total_muls / (double)total_ops,
        (double)total_adds / (double)total_ops,
        (double)total_subs / (double)total_ops);
    return (long)(total_ns / count);
}

// ベンチマーク: fp4_mul_new を iters 回呼び出してナノ秒を返す
static long bench_fp4_mul_new_run_ns(int iters, uint64_t *muls, uint64_t *adds, uint64_t *subs){
    fp4_t a, b, acc;
    fp4_init(&a); fp4_init(&b); fp4_init(&acc);
    fp4_random(&a); fp4_random(&b);

    // ウォームアップ
    fp4_mul_new(&acc, &a, &b);

    uint64_t before_mul = fp_mul_count;
    uint64_t before_add = fp_add_count;
    uint64_t before_sub = fp_sub_count;
    struct timespec st1, ed1, st2, ed2;
    clock_gettime(CLOCK_MONOTONIC, &st1);
    for (int i = 0; i < iters; i++) {
        fp4_random(&a); fp4_random(&b);
        fp4_mul_new(&acc, &a, &b);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed1);
    uint64_t after_mul = fp_mul_count;
    uint64_t after_add = fp_add_count;
    uint64_t after_sub = fp_sub_count;
    long ns_ops = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);

    clock_gettime(CLOCK_MONOTONIC, &st2);
    for (int i = 0; i < iters; i++) {
        fp4_random(&a); fp4_random(&b);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed2);
    long ns_init = (ed2.tv_sec - st2.tv_sec) * 1000000000L + (ed2.tv_nsec - st2.tv_nsec);
    long ns = ns_ops - ns_init;
    if (ns < 0) ns = 0;

    if (muls) *muls = after_mul - before_mul;
    if (adds) *adds = after_add - before_add;
    if (subs) *subs = after_sub - before_sub;

    fp4_clear(&a); fp4_clear(&b); fp4_clear(&acc);
    return ns;
}

long bench_fp4_mul_new_avg(int iters, int count){
    long long total_ns = 0;
    long long total_ops = (long long)iters * (long long)count;
    unsigned long long total_muls = 0;
    unsigned long long total_adds = 0;
    unsigned long long total_subs = 0;
    for (int i = 0; i < count; i++) {
        uint64_t muls = 0, adds = 0, subs = 0;
        total_ns += bench_fp4_mul_new_run_ns(iters, &muls, &adds, &subs);
        total_muls += muls;
        total_adds += adds;
        total_subs += subs;
    }
    printf("[Bench] fp4_mul_new: %d cases x %d iters, %.2f ns/op, muls/op: %.2f, adds/op: %.2f, subs/op: %.2f\n",
        count, iters, (double)total_ns / (double)total_ops,
        (double)total_muls / (double)total_ops,
        (double)total_adds / (double)total_ops,
        (double)total_subs / (double)total_ops);
    return (long)(total_ns / count);
}

static long bench_fp4_mul_karatsuba_run_ns(int iters, uint64_t *muls, uint64_t *adds, uint64_t *subs){
    fp4_t a, b, acc;
    fp4_init(&a); fp4_init(&b); fp4_init(&acc);
    fp4_random(&a); fp4_random(&b);

    // ウォームアップ
    fp4_mul_karatsuba(&acc, &a, &b);

    uint64_t before_mul = fp_mul_count;
    uint64_t before_add = fp_add_count;
    uint64_t before_sub = fp_sub_count;
    struct timespec st1, ed1, st2, ed2;
    clock_gettime(CLOCK_MONOTONIC, &st1);
    for (int i = 0; i < iters; i++) {
        fp4_random(&a); fp4_random(&b);
        fp4_mul_karatsuba(&acc, &a, &b);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed1);
    uint64_t after_mul = fp_mul_count;
    uint64_t after_add = fp_add_count;
    uint64_t after_sub = fp_sub_count;
    long ns_ops = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);

    clock_gettime(CLOCK_MONOTONIC, &st2);
    for (int i = 0; i < iters; i++) {
        fp4_random(&a); fp4_random(&b);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed2);
    long ns_init = (ed2.tv_sec - st2.tv_sec) * 1000000000L + (ed2.tv_nsec - st2.tv_nsec);
    long ns = ns_ops - ns_init;
    if (ns < 0) ns = 0;

    if (muls) *muls = after_mul - before_mul;
    if (adds) *adds = after_add - before_add;
    if (subs) *subs = after_sub - before_sub;

    fp4_clear(&a); fp4_clear(&b); fp4_clear(&acc);
    return ns;
}

long bench_fp4_mul_karatsuba_avg(int iters, int count){
    long long total_ns = 0;
    long long total_ops = (long long)iters * (long long)count;
    unsigned long long total_muls = 0;
    unsigned long long total_adds = 0;
    unsigned long long total_subs = 0;
    for (int i = 0; i < count; i++) {
        uint64_t muls = 0, adds = 0, subs = 0;
        total_ns += bench_fp4_mul_karatsuba_run_ns(iters, &muls, &adds, &subs);
        total_muls += muls;
        total_adds += adds;
        total_subs += subs;
    }
    printf("[Bench] fp4_mul_karatsuba: %d cases x %d iters, %.2f ns/op, muls/op: %.2f, adds/op: %.2f, subs/op: %.2f\n",
        count, iters, (double)total_ns / (double)total_ops,
        (double)total_muls / (double)total_ops,
        (double)total_adds / (double)total_ops,
        (double)total_subs / (double)total_ops);
    return (long)(total_ns / count);
}

static long bench_fp4_mul_slow_run_ns(int iters, uint64_t *muls, uint64_t *adds, uint64_t *subs){
    fp4_t a, b, acc;
    fp4_init(&a); fp4_init(&b); fp4_init(&acc);
    fp4_random(&a); fp4_random(&b);

    // ウォームアップ
    fp4_mul_slow(&acc, &a, &b);

    uint64_t before_mul = fp_mul_count;
    uint64_t before_add = fp_add_count;
    uint64_t before_sub = fp_sub_count;
    struct timespec st1, ed1, st2, ed2;
    clock_gettime(CLOCK_MONOTONIC, &st1);
    for (int i = 0; i < iters; i++) {
        fp4_random(&a); fp4_random(&b);
        fp4_mul_slow(&acc, &a, &b);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed1);
    uint64_t after_mul = fp_mul_count;
    uint64_t after_add = fp_add_count;
    uint64_t after_sub = fp_sub_count;
    long ns_ops = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);

    clock_gettime(CLOCK_MONOTONIC, &st2);
    for (int i = 0; i < iters; i++) {
        fp4_random(&a); fp4_random(&b);
    }
    clock_gettime(CLOCK_MONOTONIC, &ed2);
    long ns_init = (ed2.tv_sec - st2.tv_sec) * 1000000000L + (ed2.tv_nsec - st2.tv_nsec);
    long ns = ns_ops - ns_init;
    if (ns < 0) ns = 0;

    if (muls) *muls = after_mul - before_mul;
    if (adds) *adds = after_add - before_add;
    if (subs) *subs = after_sub - before_sub;

    fp4_clear(&a); fp4_clear(&b); fp4_clear(&acc);
    return ns;
}

long bench_fp4_mul_slow_avg(int iters, int count){
    long long total_ns = 0;
    long long total_ops = (long long)iters * (long long)count;
    unsigned long long total_muls = 0;
    unsigned long long total_adds = 0;
    unsigned long long total_subs = 0;
    for (int i = 0; i < count; i++) {
        uint64_t muls = 0, adds = 0, subs = 0;
        total_ns += bench_fp4_mul_slow_run_ns(iters, &muls, &adds, &subs);
        total_muls += muls;
        total_adds += adds;
        total_subs += subs;
    }
    printf("[Bench] fp4_mul_slow: %d cases x %d iters, %.2f ns/op, muls/op: %.2f, adds/op: %.2f, subs/op: %.2f\n",
        count, iters, (double)total_ns / (double)total_ops,
        (double)total_muls / (double)total_ops,
        (double)total_adds / (double)total_ops,
        (double)total_subs / (double)total_ops);
    return (long)(total_ns / count);
}

// ベンチマーク: fp_add を iters 回呼び出してナノ秒を返す
static long bench_fp_add_run_ns(int iters){
    fp_t acc;
    fp_init(&acc);
    size_t total_ops = (size_t)iters * (size_t)BENCH_UNROLL;
    fp_t *as = (fp_t *)malloc(total_ops * sizeof(fp_t));
    fp_t *bs = (fp_t *)malloc(total_ops * sizeof(fp_t));
    if (!as || !bs) {
        free(as);
        free(bs);
        fp_clear(&acc);
        return 0;
    }
    for (size_t i = 0; i < total_ops; i++) {
        fp_random(&as[i]);
        fp_random(&bs[i]);
    }

    // ウォームアップ
    fp_add(&acc, &as[0], &bs[0]);

    volatile uint32_t sink = 0;
    struct timespec st1, ed1;
    clock_gettime(CLOCK_MONOTONIC, &st1);
    for (int i = 0; i < iters; i++) {
        size_t j = (size_t)i * (size_t)BENCH_UNROLL;
        fp_add(&acc, &as[j + 0], &bs[j + 0]);
        fp_add(&acc, &as[j + 1], &bs[j + 1]);
        fp_add(&acc, &as[j + 2], &bs[j + 2]);
        fp_add(&acc, &as[j + 3], &bs[j + 3]);
        fp_add(&acc, &as[j + 4], &bs[j + 4]);
        fp_add(&acc, &as[j + 5], &bs[j + 5]);
        fp_add(&acc, &as[j + 6], &bs[j + 6]);
        fp_add(&acc, &as[j + 7], &bs[j + 7]);
        sink ^= acc.x0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ed1);

    if (sink == 0xFFFFFFFF) {
        fp_set(&acc, &as[0]);
    }

    long ns_add = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);
    long ns = ns_add;
    if (ns < 0) ns = 0;

    free(as);
    free(bs);
    fp_clear(&acc);
    return ns;
}

long bench_fp_add_avg(int iters, int count){
    long long total_ns = 0;
    long long total_ops = (long long)iters * (long long)count * (long long)BENCH_UNROLL;
    for (int i = 0; i < count; i++) {
        total_ns += bench_fp_add_run_ns(iters);
    }
    printf("[Bench] fp_add: %d cases x %d iters, %.2f ns/op\n",
        count, iters, (double)total_ns / (double)total_ops);
    return (long)(total_ns / count);
}

// ベンチマーク: fp_sub を iters 回呼び出してナノ秒を返す（毎回乱数）
static long bench_fp_sub_run_ns(int iters){
    fp_t acc;
    fp_init(&acc);
    size_t total_ops = (size_t)iters * (size_t)BENCH_UNROLL;
    fp_t *as = (fp_t *)malloc(total_ops * sizeof(fp_t));
    fp_t *bs = (fp_t *)malloc(total_ops * sizeof(fp_t));
    if (!as || !bs) {
        free(as);
        free(bs);
        fp_clear(&acc);
        return 0;
    }
    for (size_t i = 0; i < total_ops; i++) {
        fp_random(&as[i]);
        fp_random(&bs[i]);
    }

    // ウォームアップ
    fp_sub(&acc, &as[0], &bs[0]);

    volatile uint32_t sink = 0;
    struct timespec st1, ed1;
    clock_gettime(CLOCK_MONOTONIC, &st1);
    for (int i = 0; i < iters; i++) {
        size_t j = (size_t)i * (size_t)BENCH_UNROLL;
        fp_sub(&acc, &as[j + 0], &bs[j + 0]);
        fp_sub(&acc, &as[j + 1], &bs[j + 1]);
        fp_sub(&acc, &as[j + 2], &bs[j + 2]);
        fp_sub(&acc, &as[j + 3], &bs[j + 3]);
        fp_sub(&acc, &as[j + 4], &bs[j + 4]);
        fp_sub(&acc, &as[j + 5], &bs[j + 5]);
        fp_sub(&acc, &as[j + 6], &bs[j + 6]);
        fp_sub(&acc, &as[j + 7], &bs[j + 7]);
        sink ^= acc.x0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ed1);

    if (sink == 0xFFFFFFFF) {
        fp_set(&acc, &as[0]);
    }

    long ns_sub = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);
    long ns = ns_sub;
    if (ns < 0) ns = 0;

    free(as);
    free(bs);
    fp_clear(&acc);
    return ns;
}

long bench_fp_sub_avg(int iters, int count){
    long long total_ns = 0;
    long long total_ops = (long long)iters * (long long)count * (long long)BENCH_UNROLL;
    for (int i = 0; i < count; i++) {
        total_ns += bench_fp_sub_run_ns(iters);
    }
    printf("[Bench] fp_sub: %d cases x %d iters, %.2f ns/op\n",
        count, iters, (double)total_ns / (double)total_ops);
    return (long)(total_ns / count);
}

// ベンチマーク: fp_mul を iters 回呼び出してナノ秒を返す（毎回乱数）
static long bench_fp_mul_run_ns(int iters){
    fp_t acc;
    fp_init(&acc);
    size_t total_ops = (size_t)iters * (size_t)BENCH_UNROLL;
    fp_t *as = (fp_t *)malloc(total_ops * sizeof(fp_t));
    fp_t *bs = (fp_t *)malloc(total_ops * sizeof(fp_t));
    if (!as || !bs) {
        free(as);
        free(bs);
        fp_clear(&acc);
        return 0;
    }
    for (size_t i = 0; i < total_ops; i++) {
        fp_random(&as[i]);
        fp_random(&bs[i]);
    }

    // ウォームアップ
    fp_mul(&acc, &as[0], &bs[0]);

    volatile uint32_t sink = 0;
    struct timespec st1, ed1;
    clock_gettime(CLOCK_MONOTONIC, &st1);
    for (int i = 0; i < iters; i++) {
        size_t j = (size_t)i * (size_t)BENCH_UNROLL;
        fp_mul(&acc, &as[j + 0], &bs[j + 0]);
        fp_mul(&acc, &as[j + 1], &bs[j + 1]);
        fp_mul(&acc, &as[j + 2], &bs[j + 2]);
        fp_mul(&acc, &as[j + 3], &bs[j + 3]);
        fp_mul(&acc, &as[j + 4], &bs[j + 4]);
        fp_mul(&acc, &as[j + 5], &bs[j + 5]);
        fp_mul(&acc, &as[j + 6], &bs[j + 6]);
        fp_mul(&acc, &as[j + 7], &bs[j + 7]);
        sink ^= acc.x0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ed1);

    if (sink == 0xFFFFFFFF) {
        fp_set(&acc, &as[0]);
    }

    long ns_mul = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);
    long ns = ns_mul;
    if (ns < 0) ns = 0;

    free(as);
    free(bs);
    fp_clear(&acc);
    return ns;
}

long bench_fp_mul_avg(int iters, int count){
    long long total_ns = 0;
    long long total_ops = (long long)iters * (long long)count * (long long)BENCH_UNROLL;
    for (int i = 0; i < count; i++) {
        total_ns += bench_fp_mul_run_ns(iters);
    }
    printf("[Bench] fp_mul: %d cases x %d iters, %.2f ns/op\n",
        count, iters, (double)total_ns / (double)total_ops);
    return (long)(total_ns / count);
}

// ベンチマーク: fp_add_plus を iters 回呼び出してナノ秒を返す
static long bench_fp_add_plus_run_ns(int iters){
    fp_t acc;
    fp_init(&acc);
    size_t total_ops = (size_t)iters * (size_t)BENCH_UNROLL;
    fp_t *as = (fp_t *)malloc(total_ops * sizeof(fp_t));
    fp_t *bs = (fp_t *)malloc(total_ops * sizeof(fp_t));
    if (!as || !bs) {
        free(as);
        free(bs);
        fp_clear(&acc);
        return 0;
    }
    for (size_t i = 0; i < total_ops; i++) {
        fp_random(&as[i]);
        fp_random(&bs[i]);
    }

    // ウォームアップ
    fp_add_plus(&acc, &as[0], &bs[0]);

    volatile uint32_t sink = 0;
    struct timespec st1, ed1;
    clock_gettime(CLOCK_MONOTONIC, &st1);
    for (int i = 0; i < iters; i++) {
        size_t j = (size_t)i * (size_t)BENCH_UNROLL;
        fp_add_plus(&acc, &as[j + 0], &bs[j + 0]);
        fp_add_plus(&acc, &as[j + 1], &bs[j + 1]);
        fp_add_plus(&acc, &as[j + 2], &bs[j + 2]);
        fp_add_plus(&acc, &as[j + 3], &bs[j + 3]);
        fp_add_plus(&acc, &as[j + 4], &bs[j + 4]);
        fp_add_plus(&acc, &as[j + 5], &bs[j + 5]);
        fp_add_plus(&acc, &as[j + 6], &bs[j + 6]);
        fp_add_plus(&acc, &as[j + 7], &bs[j + 7]);
        sink ^= acc.x0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ed1);

    if (sink == 0xFFFFFFFF) {
        fp_set(&acc, &as[0]);
    }

    long ns_add = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);
    long ns = ns_add;
    if (ns < 0) ns = 0;

    free(as);
    free(bs);
    fp_clear(&acc);
    return ns;
}

long bench_fp_add_plus_avg(int iters, int count){
    long long total_ns = 0;
    long long total_ops = (long long)iters * (long long)count * (long long)BENCH_UNROLL;
    for (int i = 0; i < count; i++) {
        total_ns += bench_fp_add_plus_run_ns(iters);
    }
    printf("[Bench] fp_add_plus: %d cases x %d iters, %.2f ns/op\n",
        count, iters, (double)total_ns / (double)total_ops);
    return (long)(total_ns / count);
}

// ベンチマーク: fp_sub_plus を iters 回呼び出してナノ秒を返す（毎回乱数）
static long bench_fp_sub_plus_run_ns(int iters){
    fp_t acc;
    fp_init(&acc);
    size_t total_ops = (size_t)iters * (size_t)BENCH_UNROLL;
    fp_t *as = (fp_t *)malloc(total_ops * sizeof(fp_t));
    fp_t *bs = (fp_t *)malloc(total_ops * sizeof(fp_t));
    if (!as || !bs) {
        free(as);
        free(bs);
        fp_clear(&acc);
        return 0;
    }
    for (size_t i = 0; i < total_ops; i++) {
        fp_random(&as[i]);
        fp_random(&bs[i]);
    }

    // ウォームアップ
    fp_sub_plus(&acc, &as[0], &bs[0]);

    volatile uint32_t sink = 0;
    struct timespec st1, ed1;
    clock_gettime(CLOCK_MONOTONIC, &st1);
    for (int i = 0; i < iters; i++) {
        size_t j = (size_t)i * (size_t)BENCH_UNROLL;
        fp_sub_plus(&acc, &as[j + 0], &bs[j + 0]);
        fp_sub_plus(&acc, &as[j + 1], &bs[j + 1]);
        fp_sub_plus(&acc, &as[j + 2], &bs[j + 2]);
        fp_sub_plus(&acc, &as[j + 3], &bs[j + 3]);
        fp_sub_plus(&acc, &as[j + 4], &bs[j + 4]);
        fp_sub_plus(&acc, &as[j + 5], &bs[j + 5]);
        fp_sub_plus(&acc, &as[j + 6], &bs[j + 6]);
        fp_sub_plus(&acc, &as[j + 7], &bs[j + 7]);
        sink ^= acc.x0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ed1);

    if (sink == 0xFFFFFFFF) {
        fp_set(&acc, &as[0]);
    }

    long ns_sub = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);
    long ns = ns_sub;
    if (ns < 0) ns = 0;

    free(as);
    free(bs);
    fp_clear(&acc);
    return ns;
}

long bench_fp_sub_plus_avg(int iters, int count){
    long long total_ns = 0;
    long long total_ops = (long long)iters * (long long)count * (long long)BENCH_UNROLL;
    for (int i = 0; i < count; i++) {
        total_ns += bench_fp_sub_plus_run_ns(iters);
    }
    printf("[Bench] fp_sub_plus: %d cases x %d iters, %.2f ns/op\n",
        count, iters, (double)total_ns / (double)total_ops);
    return (long)(total_ns / count);
}

// ベンチマーク: fp_mul_plus を iters 回呼び出してナノ秒を返す（毎回乱数）
static long bench_fp_mul_plus_run_ns(int iters){
    fp_t acc;
    fp_init(&acc);
    size_t total_ops = (size_t)iters * (size_t)BENCH_UNROLL;
    fp_t *as = (fp_t *)malloc(total_ops * sizeof(fp_t));
    fp_t *bs = (fp_t *)malloc(total_ops * sizeof(fp_t));
    if (!as || !bs) {
        free(as);
        free(bs);
        fp_clear(&acc);
        return 0;
    }
    for (size_t i = 0; i < total_ops; i++) {
        fp_random(&as[i]);
        fp_random(&bs[i]);
    }

    // ウォームアップ
    fp_mul_plus(&acc, &as[0], &bs[0]);

    volatile uint32_t sink = 0;
    struct timespec st1, ed1;
    clock_gettime(CLOCK_MONOTONIC, &st1);
    for (int i = 0; i < iters; i++) {
        size_t j = (size_t)i * (size_t)BENCH_UNROLL;
        fp_mul_plus(&acc, &as[j + 0], &bs[j + 0]);
        fp_mul_plus(&acc, &as[j + 1], &bs[j + 1]);
        fp_mul_plus(&acc, &as[j + 2], &bs[j + 2]);
        fp_mul_plus(&acc, &as[j + 3], &bs[j + 3]);
        fp_mul_plus(&acc, &as[j + 4], &bs[j + 4]);
        fp_mul_plus(&acc, &as[j + 5], &bs[j + 5]);
        fp_mul_plus(&acc, &as[j + 6], &bs[j + 6]);
        fp_mul_plus(&acc, &as[j + 7], &bs[j + 7]);
        sink ^= acc.x0;
    }
    clock_gettime(CLOCK_MONOTONIC, &ed1);

    if (sink == 0xFFFFFFFF) {
        fp_set(&acc, &as[0]);
    }

    long ns_mul = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);
    long ns = ns_mul;
    if (ns < 0) ns = 0;

    free(as);
    free(bs);
    fp_clear(&acc);
    return ns;
}

long bench_fp_mul_plus_avg(int iters, int count){
    long long total_ns = 0;
    long long total_ops = (long long)iters * (long long)count * (long long)BENCH_UNROLL;
    for (int i = 0; i < count; i++) {
        total_ns += bench_fp_mul_plus_run_ns(iters);
    }
    printf("[Bench] fp_mul_plus: %d cases x %d iters, %.2f ns/op\n",
        count, iters, (double)total_ns / (double)total_ops);
    return (long)(total_ns / count);
}

// static long bench_matrix_round_p_ns(int iters, int inv_mode, bench_counts_t *counts){
//     state_t s;
//     state_init(&s);
//     state_random(&s);

//     matrix_round_p(&s, 0, inv_mode);

//     volatile uint32_t sink = 0;
//     uint64_t before = fp_mul_count;
//     uint64_t before_add = fp_add_count;
//     uint64_t before_sub = fp_sub_count;
//     uint64_t before_fp4 = fp4_mul_count;
//     struct timespec st1, ed1, st2, ed2;
//     clock_gettime(CLOCK_MONOTONIC, &st1);
//     for(int i=0;i<iters;i++){
//         state_random(&s);
//         matrix_round_p(&s, (uint8_t)i, inv_mode);
//         sink ^= s.m[0][0].x0;
//     }
//     clock_gettime(CLOCK_MONOTONIC, &ed1);
//     uint64_t after = fp_mul_count;
//     uint64_t after_add = fp_add_count;
//     uint64_t after_sub = fp_sub_count;
//     uint64_t after_fp4 = fp4_mul_count;

//     clock_gettime(CLOCK_MONOTONIC, &st2);
//     for(int i=0;i<iters;i++){
//         state_random(&s);
//     }
//     clock_gettime(CLOCK_MONOTONIC, &ed2);
//     long ns_ops = (ed1.tv_sec - st1.tv_sec) * 1000000000L + (ed1.tv_nsec - st1.tv_nsec);
//     long ns_init = (ed2.tv_sec - st2.tv_sec) * 1000000000L + (ed2.tv_nsec - st2.tv_nsec);
//     long ns = ns_ops - ns_init;
//     if (ns < 0) ns = 0;

//     if (counts) {
//         counts->muls = after - before;
//         counts->adds = after_add - before_add;
//         counts->subs = after_sub - before_sub;
//         counts->fp4 = after_fp4 - before_fp4;
//     }

//     if (sink == 0xFFFFFFFF) {
//         state_set_zero(&s);
//     }
//     return ns;
// }

// static long bench_matrix_round_p_run_ns(int iters, int inv_mode,
//                                         uint64_t *muls, uint64_t *adds,
//                                         uint64_t *subs, uint64_t *fp4s){
//     bench_counts_t counts = {0};
//     long ns = bench_matrix_round_p_ns(iters, inv_mode, &counts);
//     if (muls) *muls = counts.muls;
//     if (adds) *adds = counts.adds;
//     if (subs) *subs = counts.subs;
//     if (fp4s) *fp4s = counts.fp4;
//     return ns;
// }

// static long bench_matrix_round_p_avg(int iters, int count, int inv_mode, const char *label){
//     long long total_ns = 0;
//     long long total_ops = (long long)iters * (long long)count;
//     unsigned long long total_muls = 0;
//     unsigned long long total_adds = 0;
//     unsigned long long total_subs = 0;
//     unsigned long long total_fp4 = 0;
//     for (int i = 0; i < count; i++) {
//         uint64_t muls = 0, adds = 0, subs = 0, fp4s = 0;
//         total_ns += bench_matrix_round_p_run_ns(iters, inv_mode, &muls, &adds, &subs, &fp4s);
//         total_muls += muls;
//         total_adds += adds;
//         total_subs += subs;
//         total_fp4 += fp4s;
//     }
//     printf("[Bench] MATRIX round P (%s): %d cases x %d iters, %.2f ns/op, muls/op: %.2f, adds/op: %.2f, subs/op: %.2f, fp4_mul/op: %.2f\n",
//         label, count, iters, (double)total_ns / (double)total_ops,
//         (double)total_muls / (double)total_ops,
//         (double)total_adds / (double)total_ops,
//         (double)total_subs / (double)total_ops,
//         (double)total_fp4 / (double)total_ops);
//     return (long)(total_ns / count);
// }

// long bench_matrix_round_p_inv_avg(int iters, int count){
//     return bench_matrix_round_p_avg(iters, count, 0, "inv");
// }

// long bench_matrix_round_p_inv_new_avg(int iters, int count){
//     return bench_matrix_round_p_avg(iters, count, 1, "inv_new");
// }

// long bench_matrix_round_p_inv_karatsuba_avg(int iters, int count){
//     return bench_matrix_round_p_avg(iters, count, 2, "inv_karatsuba");
// }
