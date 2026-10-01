/*
 * 16_xof_test.c --- MATRIX-XOF(16_xof.c)が正しく動いているかを確認する
 *
 *  xof_31 と xof_127 の両方に、同じ検査を同じ条件で行う。
 *  参照 test vector は無いので、XOF が満たすべき性質と、
 *  既存の関数だけで組んだ手計算との一致で確認する。
 *
 *   1) パラメータ       : xof_*_info の値が設計どおりか
 *   2) 引数チェック     : 不正な引数で 0 を返すか
 *   3) 手計算との一致   : 1ブロックの吸収 -> P -> 絞り出し を既存関数で組んだ結果と一致するか
 *   4) 入口の一致       : xof_31 / xof_127 が matrix_xof(q, cap) と同じ出力か
 *   5) 決定性           : 同じ入力 -> 同じ出力
 *   6) 前方一致         : 短い出力が長い出力の先頭と一致するか(複数回の絞り出しをまたぐ)
 *   7) 出力長 0         : 何も書き込まずに成功するか
 *   8) パディング境界   : 入力長 0, r-1, r, r+1, 2r-1, 2r, 2r+1 がすべて別の出力か
 *   9) パディングの単射 : M と M||0x00、M||0x1F、M||0x80 が別の出力か
 *  10) 入力感度         : 1bit 違いの入力が別の出力か
 *  11) アバランシェ     : 入力1bit反転で出力ビットの約半分が反転するか
 *  12) 出力の偏り       : 出力全体の1の割合が約半分か
 *  13) 絞り出しの更新   : 長い出力の各ブロックが互いに異なるか(P で状態が進んでいるか)
 *  14) グローバル変数   : 別の素数の計算を挟んでも出力が変わらないか
 *  15) モードの区別     : xof_31 / xof_127 / matrix_hash の出力が互いに異なるか
 *
 * build:
 *   gcc -O2 16_xof_test.c 16_xof.c 16_fp.c 16_fp4.c 16_fp16.c 16_matrix.c -lgmp -o 16_xof_test
 *   (16_main.c は main 重複を避けるため含めない)
 * run:
 *   ./16_xof_test
 */
#include "16_header.h"

typedef int (*xof_fn)(uint8_t *out, size_t out_len,
                      const uint8_t *msg, size_t msg_len,
                      const state_t *MDS, const affine16_t *AFF);

static state_t    g_MDS;
static affine16_t g_AFF;

static int g_pass = 0;
static int g_fail = 0;

/* ===================== 共通の小道具 ===================== */

static void report(const char *name, int ok, const char *detail){
    printf("   [%s] %s", ok ? "OK" : "NG", name);
    if(detail && detail[0]) printf("  (%s)", detail);
    printf("\n");
    if(ok) g_pass++; else g_fail++;
}

static int popcount_diff(const uint8_t *a, const uint8_t *b, size_t n){
    int c = 0;
    for(size_t i = 0; i < n; i++){
        uint8_t x = a[i] ^ b[i];
        while(x){ c += x & 1; x >>= 1; }
    }
    return c;
}

static int popcount_bytes(const uint8_t *a, size_t n){
    int c = 0;
    for(size_t i = 0; i < n; i++){
        uint8_t x = a[i];
        while(x){ c += x & 1; x >>= 1; }
    }
    return c;
}

static void fill_msg(uint8_t *m, size_t n, unsigned seed){
    for(size_t i = 0; i < n; i++) m[i] = (uint8_t)(i * 31u + seed * 7u + 3u);
}

/* ===================== 個々のテスト ===================== */

/* 1) パラメータ */
static void test_info(xof_info in, unsigned q, unsigned cap){
    char buf[160];
    size_t lb = (size_t)((q - 1) / 8);
    int ok = in.q == q
          && in.cap_elems == cap
          && in.rate_elems == 16 - cap
          && in.rate_bytes == (16 - cap) * lb
          && in.capacity_bits == cap * q
          && in.security_bits == cap * q / 2
          && in.state_bits == 16 * q;
    snprintf(buf, sizeof(buf), "rate=%zubyte, capacity=%ubit, 強度上限=%ubit",
             in.rate_bytes, in.capacity_bits, in.security_bits);
    report("1) パラメータ", ok, buf);
}

/* 2) 引数チェック: 不正な引数では 0 を返す */
static void test_invalid_args(unsigned q){
    uint8_t out[16];
    const uint8_t m[4] = {1, 2, 3, 4};
    int ok = 1;
    ok &= matrix_xof(out, 16, m, 4, 7,   9, MATRIX_ROUNDS, &g_MDS, &g_AFF) == 0;  /* 2^7-1 は非対応 */
    ok &= matrix_xof(out, 16, m, 4, 5,   9, MATRIX_ROUNDS, &g_MDS, &g_AFF) == 0;  /* 対応していない素数 */
    ok &= matrix_xof(out, 16, m, 4, q,   0, MATRIX_ROUNDS, &g_MDS, &g_AFF) == 0;  /* キャパシティ 0 */
    ok &= matrix_xof(out, 16, m, 4, q,  16, MATRIX_ROUNDS, &g_MDS, &g_AFF) == 0;  /* キャパシティ 16 */
    ok &= matrix_xof(NULL, 16, m, 4, q,  3, MATRIX_ROUNDS, &g_MDS, &g_AFF) == 0;  /* 出力先 NULL */
    ok &= matrix_xof(out, 16, NULL, 4, q, 3, MATRIX_ROUNDS, &g_MDS, &g_AFF) == 0; /* 入力 NULL(長さ>0) */
    ok &= matrix_xof(out, 16, m, 4, q,  3, MATRIX_ROUNDS, NULL, &g_AFF) == 0;    /* MDS NULL */
    ok &= matrix_xof(out, 16, m, 4, q,  3, MATRIX_ROUNDS, &g_MDS, NULL) == 0;    /* AFF NULL */
    ok &= matrix_xof(out, 16, NULL, 0, q, 3, MATRIX_ROUNDS, &g_MDS, &g_AFF) == 1; /* 空入力は NULL でも可 */
    report("2) 引数チェック", ok, "");
}

/* 3) 手計算との一致
 *    16_xof.c を使わず、既存の関数だけで「パディングした1ブロックを吸収 -> P -> レート部を出力」
 *    を組み、matrix_xof の出力(1ブロックぶん)と比べる。
 *    レート部の位置(行優先の先頭 r_e 個)、読み込みの順序、パディング、出力の並びがすべて検査される。 */
static void manual_xof_one_block(uint8_t *out, const uint8_t *msg, size_t msg_len,
                                 unsigned q, unsigned cap){
    field_select_for_output((size_t)8 * q);         /* load_bits を q に合わせる */
    unsigned r_e   = 16 - cap;
    size_t   lb    = load_bits / 8;
    size_t   rate  = r_e * lb;

    uint8_t blk[256];
    memset(blk, 0, rate);
    memcpy(blk, msg, msg_len);                      /* msg_len < rate を前提 */
    blk[msg_len]  ^= 0x1F;
    blk[rate - 1] ^= 0x80;

    state_t S; state_init(&S); state_set_zero(&S);
    for(unsigned k = 0; k < r_e; k++){
        u128 v = 0;
        for(size_t b = 0; b < lb; b++) v = (v << 8) | blk[k * lb + b];
        fp_t t; fp_init(&t); fp_set_ui(&t, v);
        fp_add(&S.m[k / 4][k % 4], &S.m[k / 4][k % 4], &t);
    }
    matrix_permutation_P(&S, &S, MATRIX_ROUNDS, &g_MDS, &g_AFF);

    for(unsigned k = 0; k < r_e; k++){
        u128 v = S.m[k / 4][k % 4].x0;
        for(size_t b = 0; b < lb; b++)
            out[k * lb + (lb - 1 - b)] = (uint8_t)(v >> (8 * b));
    }
    state_clear(&S);
}

static void test_manual(unsigned q, unsigned cap, size_t rate){
    uint8_t m[256], a[256], b[256];
    size_t lens[3] = {0, 5, rate - 1};
    int ok = 1;
    for(int i = 0; i < 3; i++){
        fill_msg(m, lens[i], (unsigned)i);
        matrix_xof(a, rate, m, lens[i], q, cap, MATRIX_ROUNDS, &g_MDS, &g_AFF);
        manual_xof_one_block(b, m, lens[i], q, cap);
        ok &= memcmp(a, b, rate) == 0;
    }
    report("3) 手計算との一致", ok, "入力長 0 / 5 / rate-1");
}

/* 4) 入口の一致 */
static void test_entry(xof_fn f, unsigned q, unsigned cap){
    uint8_t m[100], a[300], b[300];
    fill_msg(m, sizeof(m), 1);
    f(a, sizeof(a), m, sizeof(m), &g_MDS, &g_AFF);
    matrix_xof(b, sizeof(b), m, sizeof(m), q, cap, MATRIX_ROUNDS, &g_MDS, &g_AFF);
    report("4) 入口の一致", memcmp(a, b, sizeof(a)) == 0, "");
}

/* 5) 決定性 */
static void test_deterministic(xof_fn f){
    uint8_t m[400], a[100], b[100];
    int fail = 0;
    for(int t = 0; t < 20; t++){
        size_t ml = (size_t)(rand() % 400);
        fill_msg(m, ml, (unsigned)t);
        int r1 = f(a, sizeof(a), m, ml, &g_MDS, &g_AFF);
        int r2 = f(b, sizeof(b), m, ml, &g_MDS, &g_AFF);
        if(r1 != 1 || r2 != 1 || memcmp(a, b, sizeof(a))) fail++;
    }
    report("5) 決定性", fail == 0, "");
}

/* 6) 前方一致: 32 / rate / rate+1 / 300 byte の出力が 2*rate+50 byte の出力の先頭と一致 */
static void test_prefix(xof_fn f, size_t rate){
    uint8_t m[50], full[600], part[600];
    size_t full_len = 2 * rate + 50;              /* 絞り出しを3回またぐ */
    size_t lens[4]  = {32, rate, rate + 1, 300 < full_len ? 300 : full_len};
    int ok = 1;
    fill_msg(m, sizeof(m), 2);
    f(full, full_len, m, sizeof(m), &g_MDS, &g_AFF);
    for(int i = 0; i < 4; i++){
        f(part, lens[i], m, sizeof(m), &g_MDS, &g_AFF);
        ok &= memcmp(part, full, lens[i]) == 0;
    }
    report("6) 前方一致", ok, "");
}

/* 7) 出力長 0: 成功し、出力先に何も書き込まない */
static void test_zero_output(xof_fn f){
    uint8_t m[10], out[8];
    fill_msg(m, sizeof(m), 3);
    memset(out, 0xAA, sizeof(out));
    int r = f(out, 0, m, sizeof(m), &g_MDS, &g_AFF);
    int untouched = 1;
    for(size_t i = 0; i < sizeof(out); i++) if(out[i] != 0xAA) untouched = 0;
    report("7) 出力長 0", r == 1 && untouched, "");
}

/* 8) パディング境界 */
static void test_padding_boundary(xof_fn f, size_t rate){
    static uint8_t m[600];
    size_t lens[7] = {0, rate - 1, rate, rate + 1, 2 * rate - 1, 2 * rate, 2 * rate + 1};
    uint8_t outs[7][64];
    int same = 0;
    fill_msg(m, sizeof(m), 4);
    for(int i = 0; i < 7; i++) f(outs[i], 64, m, lens[i], &g_MDS, &g_AFF);
    for(int i = 0; i < 7; i++)
        for(int j = i + 1; j < 7; j++)
            if(memcmp(outs[i], outs[j], 64) == 0) same++;
    report("8) パディング境界", same == 0, "0, r-1, r, r+1, 2r-1, 2r, 2r+1");
}

/* 9) パディングの単射: M と M にパディングと紛らわしいバイトを付けたものが別の出力
 *    M の長さは、付けたバイトがブロック末尾に来る rate-2 と、普通の長さ 10 の両方で試す */
static void test_padding_injective(xof_fn f, size_t rate){
    uint8_t m[256], o0[64], o1[64];
    const uint8_t tails[3] = {0x00, 0x1F, 0x80};
    size_t bases[2] = {10, rate - 2};
    int same = 0;
    for(int b = 0; b < 2; b++){
        fill_msg(m, bases[b], 5);
        f(o0, 64, m, bases[b], &g_MDS, &g_AFF);
        for(int t = 0; t < 3; t++){
            m[bases[b]] = tails[t];
            f(o1, 64, m, bases[b] + 1, &g_MDS, &g_AFF);
            if(memcmp(o0, o1, 64) == 0) same++;
        }
    }
    report("9) パディングの単射", same == 0, "M と M||00, M||1F, M||80");
}

/* 10) 入力感度 + 11) アバランシェ */
static void test_sensitivity(xof_fn f){
    const int    N   = 200;
    const size_t OUT = 64;
    uint8_t m[400], m2[400], a[64], b[64];
    int same = 0;
    long flipped = 0;
    for(int t = 0; t < N; t++){
        size_t ml = 1 + (size_t)(rand() % 399);
        for(size_t i = 0; i < ml; i++) m2[i] = m[i] = (uint8_t)rand();
        int bit = rand() % (int)(ml * 8);
        m2[bit / 8] ^= (uint8_t)(1u << (bit % 8));
        f(a, OUT, m,  ml, &g_MDS, &g_AFF);
        f(b, OUT, m2, ml, &g_MDS, &g_AFF);
        if(memcmp(a, b, OUT) == 0) same++;
        flipped += popcount_diff(a, b, OUT);
    }
    double frac = (double)flipped / ((double)N * OUT * 8);
    char buf[64];
    report("10) 入力感度", same == 0, "");
    snprintf(buf, sizeof(buf), "反転率 %.4f, 理想 0.5", frac);
    report("11) アバランシェ", frac > 0.48 && frac < 0.52, buf);
}

/* 12) 出力の偏り: 長い出力の1の割合 */
static void test_bit_balance(xof_fn f){
    static uint8_t out[4096];
    const uint8_t m[] = "MATRIX-XOF";
    f(out, sizeof(out), m, sizeof(m) - 1, &g_MDS, &g_AFF);
    double frac = (double)popcount_bytes(out, sizeof(out)) / (sizeof(out) * 8.0);
    char buf[64];
    snprintf(buf, sizeof(buf), "1の割合 %.4f(4096byte)", frac);
    report("12) 出力の偏り", frac > 0.49 && frac < 0.51, buf);
}

/* 13) 絞り出しの更新: 長い出力をブロックごとに区切り、すべてのブロックが互いに異なる */
static void test_squeeze_blocks(xof_fn f, size_t rate){
    static uint8_t out[8 * 256];
    const size_t nblk = 8;
    const uint8_t m[] = "squeeze";
    int same = 0;
    f(out, nblk * rate, m, sizeof(m) - 1, &g_MDS, &g_AFF);
    for(size_t i = 0; i < nblk; i++)
        for(size_t j = i + 1; j < nblk; j++)
            if(memcmp(out + i * rate, out + j * rate, rate) == 0) same++;
    report("13) 絞り出しの更新", same == 0, "8ブロックが互いに異なる");
}

/* 14) グローバル変数: 別の素数の計算を挟んでも同じ出力 */
static void test_global_state(xof_fn f){
    uint8_t m[40], a[100], b[100], tmp[64];
    fill_msg(m, sizeof(m), 6);
    f(a, sizeof(a), m, sizeof(m), &g_MDS, &g_AFF);

    matrix_hash(tmp, 8,  m, sizeof(m), MATRIX_ROUNDS, &g_MDS, &g_AFF);   /* 2^7-1 に切り替わる */
    xof_31 (tmp, 64, m, sizeof(m), &g_MDS, &g_AFF);
    xof_127(tmp, 64, m, sizeof(m), &g_MDS, &g_AFF);
    matrix_hash(tmp, 64, m, sizeof(m), MATRIX_ROUNDS, &g_MDS, &g_AFF);   /* 2^127-1 に切り替わる */

    f(b, sizeof(b), m, sizeof(m), &g_MDS, &g_AFF);
    report("14) グローバル変数", memcmp(a, b, sizeof(a)) == 0, "");
}

/* ===================== まとめて実行 ===================== */

static void run_all(const char *name, xof_fn f, xof_info in){
    printf("== %s ==\n", name);
    test_info(in, in.q, in.cap_elems);
    test_invalid_args(in.q);
    test_manual(in.q, in.cap_elems, in.rate_bytes);
    test_entry(f, in.q, in.cap_elems);
    test_deterministic(f);
    test_prefix(f, in.rate_bytes);
    test_zero_output(f);
    test_padding_boundary(f, in.rate_bytes);
    test_padding_injective(f, in.rate_bytes);
    test_sensitivity(f);
    test_bit_balance(f);
    test_squeeze_blocks(f, in.rate_bytes);
    test_global_state(f);
    printf("\n");
}

/* 15) モードの区別 */
static void test_modes_distinct(void){
    uint8_t a[32], b[32], h[32];
    const uint8_t m[] = "MATRIX";
    xof_31 (a, 32, m, 6, &g_MDS, &g_AFF);
    xof_127(b, 32, m, 6, &g_MDS, &g_AFF);
    matrix_hash(h, 32, m, 6, MATRIX_ROUNDS, &g_MDS, &g_AFF);
    int ok = memcmp(a, b, 32) && memcmp(a, h, 32) && memcmp(b, h, 32);
    printf("== 共通 ==\n");
    report("15) モードの区別", ok, "xof_31 / xof_127 / matrix_hash");
    printf("\n");
}

int main(void){
    srand(12345);
    setup_MDS(&g_MDS);
    affine16_init(&g_AFF);
    affine16_set_A(&g_AFF);
    affine16_set_b(&g_AFF);

    printf("=== MATRIX-XOF テスト ===\n\n");
    run_all("xof_31  (p = 2^31-1)",  xof_31,  xof_31_info());
    run_all("xof_127 (p = 2^127-1)", xof_127, xof_127_info());
    test_modes_distinct();

    printf("=== 結果: %d 項目中 %d OK, %d NG ===\n", g_pass + g_fail, g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}