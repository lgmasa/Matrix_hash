/*
 * bench.c --- MATRIX / SHA-256 / Grøstl-256 を同一条件で測る測定ハーネス
 *
 *  公平さのため、全ハッシュを共通シグネチャ hash_fn に包み、
 *  同じ bench_one() に通す(同一CPU・同一コンパイラ・同一測定コード・同一入力)。
 *
 *  【第1段階】まず MATRIX だけ測る。SHA/Grøstl は ENABLE_* を 0 のままにしておく。
 *  【第2段階】参照実装を入手したら ENABLE_SHA256 / ENABLE_GROESTL を 1 にして
 *             ラッパ内の関数呼び出しを実装に合わせる。
 *
 * build(第1段階: MATRIX のみ):
 *   gcc -O2 bench.c 16_fp.c 16_fp4.c 16_fp16.c 16_matrix.c -lgmp -o bench
 *   ※ 16_main.c は main 重複になるので含めない
 *
 * build(第2段階: 3つ揃えたら):
 *   gcc -O2 bench.c 16_fp.c 16_fp4.c 16_fp16.c 16_matrix.c \
 *       sha256_ref.c groestl_ref.c -lgmp -o bench
 *
 * 測定前におすすめ(周波数変動を抑える):
 *   sudo cpupower frequency-set -g performance
 * コンパイルはこれ
 * gcc -O2 bench.c 16_fp.c 16_fp4.c 16_fp16.c 16_matrix.c \
    sha256_ref.c groestl_ref.c -lgmp -o bench
 */
#include "16_header.h"
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <time.h>

/* ===== 比較対象の有効/無効(参照実装を入れたら 1 にする) ===== */
#define ENABLE_SHA256   0
#define ENABLE_GROESTL  0

#if ENABLE_SHA256
/* 参照実装のヘッダ or プロトタイプ。実装に合わせて修正すること */
void sha256(const uint8_t *msg, size_t len, uint8_t *digest);
#endif
#if ENABLE_GROESTL
void groestl256(const uint8_t *msg, size_t len, uint8_t *digest);
#endif

/* ===== 測定対象の共通シグネチャ =====
 * 全ハッシュをこの形に包むことで、全く同じループで測れる = 公平 */
typedef void (*hash_fn)(uint8_t *digest, size_t out_len,
                        const uint8_t *msg, size_t len);

/* ===== 計測ユーティリティ ===== */
static double elapsed_sec(struct timespec a, struct timespec b){
    return (double)(b.tv_sec - a.tv_sec) + (double)(b.tv_nsec - a.tv_nsec) * 1e-9;
}

typedef struct {
    double sec_per_hash;     /* 1回あたりの秒 */
    double cycles_per_byte;  /* cycles/byte(cpu_ghz を渡したとき) */
    double mb_per_sec;       /* スループット */
} bench_result;

/* 最適化で呼び出しごと消されるのを防ぐための吸い込み先 */
static volatile uint8_t g_sink = 0;

static bench_result bench_one(hash_fn f, size_t out_len,
                              const uint8_t *msg, size_t len,
                              int warmup, int iters, double cpu_ghz){
    uint8_t digest[512];                 /* MATRIX の最大127byteでも余裕 */
    struct timespec t0, t1;

    /* ウォームアップ(キャッシュ・分岐予測を温める。測定に含めない) */
    for(int i = 0; i < warmup; i++){
        f(digest, out_len, msg, len);
        g_sink ^= digest[0];
    }

    /* 本測定 */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for(int i = 0; i < iters; i++){
        f(digest, out_len, msg, len);
        g_sink ^= digest[0];             /* 結果を使う = dead code 化を防ぐ */
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double total = elapsed_sec(t0, t1);
    bench_result r;
    r.sec_per_hash    = total / (double)iters;
    r.mb_per_sec      = (double)len * (double)iters / total / 1e6;
    r.cycles_per_byte = (cpu_ghz > 0.0)
                      ? r.sec_per_hash * (cpu_ghz * 1e9) / (double)len
                      : 0.0;
    return r;
}

/* 複数のハッシュを同一条件で測って表にする */
static void bench_all(const char *names[], hash_fn fns[], size_t out_lens[],
                      int n, const uint8_t *msg, size_t len,
                      int warmup, int iters, double cpu_ghz){
    printf("--- 入力長 %zu byte / %d回平均(warmup %d) / CPU %.2f GHz ---\n",
           len, iters, warmup, cpu_ghz);
    printf("%-14s %10s %12s %14s %10s\n",
           "hash", "out(byte)", "us/hash", "cycles/byte", "MB/s");
    for(int i = 0; i < n; i++){
        if(fns[i] == NULL) continue;     /* 無効化されている対象は飛ばす */
        bench_result r = bench_one(fns[i], out_lens[i], msg, len,
                                   warmup, iters, cpu_ghz);
        printf("%-14s %10zu %12.3f %14.2f %10.1f\n",
               names[i], out_lens[i], r.sec_per_hash * 1e6,
               r.cycles_per_byte, r.mb_per_sec);
    }
    printf("\n");
}

/* ===== MATRIX の準備 ===== */
static state_t   g_MDS;
static affine16_t g_AFF;

static void setup_MDS(state_t *MDS){
    state_init(MDS);
    int row[4] = {1, 1, 2, 8};           /* circ(1,1,2,8) */
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            fp_set_ui(&MDS->m[i][j], (u128)row[(j - i + 4) & 3]);
}

/* ===== 各ハッシュのラッパ(共通シグネチャに合わせる) ===== */
static void wrap_matrix(uint8_t *d, size_t out, const uint8_t *m, size_t len){
    matrix_hash(d, out, m, len, MATRIX_ROUNDS, &g_MDS, &g_AFF);
}

#if ENABLE_SHA256
static void wrap_sha256(uint8_t *d, size_t out, const uint8_t *m, size_t len){
    (void)out;                            /* SHA-256 は 32byte 固定 */
    sha256(m, len, d);
}
#endif

#if ENABLE_GROESTL
static void wrap_groestl(uint8_t *d, size_t out, const uint8_t *m, size_t len){
    (void)out;                            /* Grøstl-256 は 32byte 固定 */
    groestl256(m, len, d);
}
#endif

/* ===== main ===== */
int main(void){
    /* --- 測定条件(必要に応じて調整) --- */
    const double cpu_ghz = 2.60;          /* i5-11400 定格。performance 固定推奨 */
    const int    warmup  = 1000;
    const int    iters   = 20000;         /* 結果がブレるなら増やす */

    /* --- MATRIX の MDS / AFF を準備 --- */
    setup_MDS(&g_MDS);
    affine16_init(&g_AFF);
    affine16_set_A(&g_AFF);
    affine16_set_b(&g_AFF);

    /* --- 入力バッファ --- */
    static uint8_t buf[16384];
    for(size_t i = 0; i < sizeof(buf); i++) buf[i] = (uint8_t)i;

    printf("=== ハッシュ処理時間の比較(同一条件) ===\n\n");

    /* ===== (1) 256bit 出力どうしの比較 ===== */
    {
        const char *names[3] = {"MATRIX-256", "SHA-256", "Groestl-256"};
        hash_fn     fns[3]   = {wrap_matrix, NULL, NULL};
        size_t      outs[3]  = {32, 32, 32};   /* 全部 256bit に揃える */
#if ENABLE_SHA256
        fns[1] = wrap_sha256;
#endif
#if ENABLE_GROESTL
        fns[2] = wrap_groestl;
#endif
        size_t lens[] = {64, 1024, 16384};
        for(int k = 0; k < 3; k++)
            bench_all(names, fns, outs, 3, buf, lens[k], warmup, iters, cpu_ghz);
    }

    /* ===== (2) MATRIX の素数ごとの比較(出力長が違うので参考値) ===== */
    {
        printf("=== MATRIX: 素数ごと(出力長が異なるので直接比較ではなく参考) ===\n\n");
        const char *names[3] = {"MATRIX(p=2^7-1)", "MATRIX(p=2^31-1)", "MATRIX(p=2^127-1)"};
        hash_fn     fns[3]   = {wrap_matrix, wrap_matrix, wrap_matrix};
        size_t      outs[3]  = {8, 32, 64};    /* 8->q=7, 32->q=31, 64->q=127 */
        size_t lens[] = {1024, 16384};
        for(int k = 0; k < 2; k++)
            bench_all(names, fns, outs, 3, buf, lens[k], warmup, iters, cpu_ghz);
    }

    return 0;
}