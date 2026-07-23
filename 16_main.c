#include "16_header.h"

// int is_monomial(const affine16_t *AFF) {
//     fp16_t x1, x2, Lx1, Lx2, f1, f2, l, r;
//     fp16_random(&x1); fp16_random(&x2);
//     state_to_vec16(fp_t v[16], const state_t *S)
//     fp16_mul(&Lx1, AFF->A, &x1);   // A*x1（b は加えない）
//     fp16_mul(&Lx2, AFF->A, &x2);

//     fp16_set(&f1, &x1); fp16_set(&f2, &x2);
//     for (int i = 0; i < 16; i++) {
//         // f1 = x1^{p^i}, f2 = x2^{p^i}
//         fp16_mul(&l, &Lx1, &f2);
//         fp16_mul(&r, &Lx2, &f1);
//         if (fp16_cmp(&l, &r) == 0) return 1;   // 単項式の疑い
//         fp16_frobenius(&f1, &f1);              // f1 ← f1^p
//         fp16_frobenius(&f2, &f2);
//     }
//     return 0;   // 単項式ではない
// }

#define GUARD 16
#define DLEN  32      /* 出力バイト長。実装に合わせて */

static void fill_pattern(uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)(i * 31 + 7);
}

static int hash_guarded(uint8_t *out, size_t dlen,
                        const uint8_t *msg, size_t mlen,
                        int rounds, const state_t *MDS, const affine16_t *AFF)
{
    uint8_t buf[GUARD + 256 + GUARD];
    memset(buf, 0xAA, sizeof(buf));                    /* 番兵 + 未書き込み検出 */
    matrix_hash(buf + GUARD, dlen, msg, mlen, rounds, MDS, AFF);

    for (size_t i = 0; i < GUARD; i++)                  /* 前後を壊していないか */
        if (buf[i] != 0xAA || buf[GUARD + dlen + i] != 0xAA) return 0;

    int all_untouched = 1;                              /* 全バイト書いたか（弱い検査）*/
    for (size_t i = 0; i < dlen; i++)
        if (buf[GUARD + i] != 0xAA) { all_untouched = 0; break; }
    if (all_untouched) return 0;

    memcpy(out, buf + GUARD, dlen);
    return 1;
}

void test_basic(int rounds, const state_t *MDS, const affine16_t *AFF)
{
    /* ブロック長は実装に合わせる。p=2^31-1 なら 496bit = 62byte */
    const size_t lens[] = {0, 1, 2, 31, 61, 62, 63, 64, 123, 124, 125, 200};
    const size_t nlen   = sizeof(lens) / sizeof(lens[0]);
    uint8_t msg[256], h1[DLEN], h2[DLEN], hx[DLEN];

    for (size_t i = 0; i < nlen; i++) {
        fill_pattern(msg, lens[i]);

        if (!hash_guarded(h1, DLEN, msg, lens[i], rounds, MDS, AFF)) {
            printf("FAIL: buffer handling (len=%zu)\n", lens[i]); return;
        }
        if (!hash_guarded(h2, DLEN, msg, lens[i], rounds, MDS, AFF)) {
            printf("FAIL: buffer handling (len=%zu)\n", lens[i]); return;
        }
        if (memcmp(h1, h2, DLEN) != 0) {
            printf("FAIL: not deterministic (len=%zu)\n", lens[i]); return;
        }

        /* 状態リセット漏れの検出：別入力を挟んで再計算 */
        fill_pattern(msg, 17);
        matrix_hash(hx, DLEN, msg, 17, rounds, MDS, AFF);
        fill_pattern(msg, lens[i]);
        matrix_hash(h2, DLEN, msg, lens[i], rounds, MDS, AFF);
        if (memcmp(h1, h2, DLEN) != 0) {
            printf("FAIL: state leakage (len=%zu)\n", lens[i]); return;
        }
    }
    printf("test_basic: OK\n");
}

int main(void){
    uint32_t p_mer = P_MERSENNE;
    uint32_t p_pri = P_PRIME;
    printf("p_mer : %u\n",p_mer);
    // printf("p_pri : %u\n",p_pri);
    printf("α : "); fp4_printf(&alpha);

    state_t MDS; //MDS行列
    state_init(&MDS);

    affine16_t AFF;
    affine16_init(&AFF);
    affine16_set_A(&AFF);
    affine16_set_b(&AFF);
    printf("A:\n");
    affine16_print_A(&AFF);
    printf("b:\n");
    affine16_print_b(&AFF);

    int A = affine16_is_regular(&AFF);

    if (A){
        printf("A is regular\n");
    }else{
        printf("A is not regular\n");
    }

    //MDS行列をセット
    fp_set_ui(&MDS.m[0][0], 1);
    fp_set_ui(&MDS.m[0][1], 1);
    fp_set_ui(&MDS.m[0][2], 2);
    fp_set_ui(&MDS.m[0][3], 8);

    fp_set_ui(&MDS.m[1][0], 8);
    fp_set_ui(&MDS.m[1][1], 1);
    fp_set_ui(&MDS.m[1][2], 1);
    fp_set_ui(&MDS.m[1][3], 2);

    fp_set_ui(&MDS.m[2][0], 2);
    fp_set_ui(&MDS.m[2][1], 8);
    fp_set_ui(&MDS.m[2][2], 1);
    fp_set_ui(&MDS.m[2][3], 1);

    fp_set_ui(&MDS.m[3][0], 1);
    fp_set_ui(&MDS.m[3][1], 2);
    fp_set_ui(&MDS.m[3][2], 8);
    fp_set_ui(&MDS.m[3][3], 1);

    // affine16_set(&AFF);

    uint8_t block[MATRIX_BLOCK_BYTES] = {0};
    uint8_t digest[MATRIX_DIGEST_BYTES];
    uint8_t msg[] = "Hello"; //文字列リテラル "Hello World" の各文字コードが uint8_t 配列に格納される
    size_t msg_len = sizeof(msg) -1; //末尾のヌル文字\0も数えてしまい、1文字多くなってしまうから1を引く

    printf("MDS :\n");
    state_print(&MDS);

    printf("msg :\n");
    printf("%s\n",msg);
    // print_bytes_hex(msg, 11);

    //throughput_matrix_hash(msg, msg_len, 100, &MDS, &AFF);

    matrix_hash(digest, MATRIX_DIGEST_BYTES, msg, msg_len, MATRIX_ROUNDS, &MDS, &AFF);

    printf("digest :\n");
    print_bytes_hex(digest, MATRIX_DIGEST_BYTES);

    // test_basic(MATRIX_ROUNDS, &MDS, &AFF);

    uint8_t a[64], b[64], ha[DLEN], hb[DLEN];
    state_t S1, S2;
    state_init(&S1); state_init(&S2);

    memset(a, 0, 64);
    memcpy(b, a, 64);
    b[0] = 0x7F; b[1] = 0xFF; b[2] = 0xFF; b[3] = 0xFF;   /* 先頭ワードだけ差し替え */

    printf("a:\n");
    print_bytes_hex(a, MATRIX_STATE_BYTES);
    printf("b:\n");
    print_bytes_hex(b, MATRIX_STATE_BYTES);

    matrix_bytes_to_state(&S1, a);
    matrix_bytes_to_state(&S2, b);

    printf("S1:\n");
    state_print(&S1);
    printf("S2:\n");
    state_print(&S2);

    matrix_hash(ha, DLEN, a, 64, MATRIX_ROUNDS, &MDS, &AFF);
    matrix_hash(hb, DLEN, b, 64, MATRIX_ROUNDS, &MDS, &AFF);

    printf("ha :\n");
    print_bytes_hex(ha, MATRIX_DIGEST_BYTES);
    printf("hb :\n");
    print_bytes_hex(hb, MATRIX_DIGEST_BYTES);

    printf("%s\n", memcmp(&ha,&hb,DLEN)==0 ? "COLLISION" : "ok");

    // if(is_monomial(&AFF)){
    //     printf("fuck\n");
    // }else{
    //     printf("ok\n");
    // }

    state_t S;
    state_init(&S);



    state_clear(&MDS);
    affine16_clear(&AFF);



    return 0;
}

// #include "16_header.h"

// static void setup_MDS(state_t *MDS){
//     state_init(MDS);
//     fp_set_ui(&MDS->m[0][0],1); fp_set_ui(&MDS->m[0][1],1); fp_set_ui(&MDS->m[0][2],2); fp_set_ui(&MDS->m[0][3],8);
//     fp_set_ui(&MDS->m[1][0],8); fp_set_ui(&MDS->m[1][1],1); fp_set_ui(&MDS->m[1][2],1); fp_set_ui(&MDS->m[1][3],2);
//     fp_set_ui(&MDS->m[2][0],2); fp_set_ui(&MDS->m[2][1],8); fp_set_ui(&MDS->m[2][2],1); fp_set_ui(&MDS->m[2][3],1);
//     fp_set_ui(&MDS->m[3][0],1); fp_set_ui(&MDS->m[3][1],2); fp_set_ui(&MDS->m[3][2],8); fp_set_ui(&MDS->m[3][3],1);
// }

// static void setup_identity_MDS(state_t *MDS){
//     state_init(MDS);
//     state_set_zero(MDS);
//     for(int i = 0; i < 4; i++){
//         fp_set_ui(&MDS->m[i][i], 1);
//     }
// }

// static void state_set_u32(state_t *S, const uint32_t v[4][4]){
//     for(int i = 0; i < 4; i++){
//         for(int j = 0; j < 4; j++){
//             fp_set_ui(&S->m[i][j], v[i][j]);
//         }
//     }
// }

// static int state_equal_u32(const state_t *S, const uint32_t expected[4][4]){
//     fp_t e;
//     fp_init(&e);

//     for(int i = 0; i < 4; i++){
//         for(int j = 0; j < 4; j++){
//             fp_set_ui(&e, expected[i][j]);
//             if(!fp_is_equal(&S->m[i][j], &e)){
//                 fp_clear(&e);
//                 return 0;
//             }
//         }
//     }

//     fp_clear(&e);
//     return 1;
// }

// static int report_test(const char *name, int ok){
//     printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
//     return ok ? 0 : 1;
// }

// static int test_add_round_constant(void){
//     int fail = 0;
//     state_t in, out_p, out_q;
//     state_init(&in);
//     state_init(&out_p);
//     state_init(&out_q);
//     state_set_zero(&in);

//     matrix_add_round_constant_P(&out_p, &in, 1);
//     matrix_add_round_constant_Q(&out_q, &in, 1);

//     const uint32_t expected_p[4][4] = {
//         {  1,  17,  33,  49 },
//         {  0,   0,   0,   0 },
//         {  0,   0,   0,   0 },
//         {  0,   0,   0,   0 },
//     };
//     const uint32_t expected_q[4][4] = {
//         {255, 255, 255, 255},
//         {255, 255, 255, 255},
//         {255, 255, 255, 255},
//         {254, 238, 222, 206},
//     };

//     fail += report_test("AddRoundConstant P: zero input, r=1", state_equal_u32(&out_p, expected_p));
//     fail += report_test("AddRoundConstant Q: zero input, r=1", state_equal_u32(&out_q, expected_q));

//     state_clear(&in);
//     state_clear(&out_p);
//     state_clear(&out_q);
//     return fail;
// }

// static int test_shiftbytes(void){
//     int fail = 0;
//     state_t in, out_p, out_q;
//     state_init(&in);
//     state_init(&out_p);
//     state_init(&out_q);

//     const uint32_t input[4][4] = {
//         { 0,  1,  2,  3},
//         { 4,  5,  6,  7},
//         { 8,  9, 10, 11},
//         {12, 13, 14, 15},
//     };
//     const uint32_t expected_p[4][4] = {
//         { 0,  1,  2,  3},
//         { 5,  6,  7,  4},
//         {10, 11,  8,  9},
//         {15, 12, 13, 14},
//     };
//     const uint32_t expected_q[4][4] = {
//         { 1,  2,  3,  0},
//         { 7,  4,  5,  6},
//         { 8,  9, 10, 11},
//         {14, 15, 12, 13},
//     };

//     state_set_u32(&in, input);
//     matrix_shiftbytes_P(&out_p, &in);
//     matrix_shiftbytes_Q(&out_q, &in);

//     fail += report_test("ShiftBytes P: row shifts {0,1,2,3}", state_equal_u32(&out_p, expected_p));
//     fail += report_test("ShiftBytes Q: row shifts {1,3,0,2}", state_equal_u32(&out_q, expected_q));

//     state_clear(&in);
//     state_clear(&out_p);
//     state_clear(&out_q);
//     return fail;
// }

// static int test_mixbytes(void){
//     int fail = 0;
//     state_t in, out, MDS, I;
//     state_init(&in);
//     state_init(&out);
//     setup_MDS(&MDS);
//     setup_identity_MDS(&I);

//     const uint32_t input_all[4][4] = {
//         { 0,  1,  2,  3},
//         { 4,  5,  6,  7},
//         { 8,  9, 10, 11},
//         {12, 13, 14, 15},
//     };
//     state_set_u32(&in, input_all);
//     matrix_mixbytes(&out, &in, &I);
//     fail += report_test("MixBytes: identity matrix keeps state unchanged", state_equal(&out, &in));

//     const uint32_t input_col[4][4] = {
//         {1, 0, 0, 0},
//         {2, 0, 0, 0},
//         {3, 0, 0, 0},
//         {4, 0, 0, 0},
//     };
//     const uint32_t expected_col[4][4] = {
//         {41, 0, 0, 0},
//         {21, 0, 0, 0},
//         {25, 0, 0, 0},
//         {33, 0, 0, 0},
//     };
//     state_set_u32(&in, input_col);
//     matrix_mixbytes(&out, &in, &MDS);
//     fail += report_test("MixBytes: MDS * [1,2,3,4]^T", state_equal_u32(&out, expected_col));

//     state_clear(&in);
//     state_clear(&out);
//     state_clear(&MDS);
//     state_clear(&I);
//     return fail;
// }

// static int test_subbytes(void){
//     int fail = 0;
//     state_t in, out;
//     affine16_t AFF;
//     state_init(&in);
//     state_init(&out);
//     affine16_init(&AFF);

//     affine16_set(&AFF);
//     state_set_zero(&in);
//     matrix_subbytes(&out, &in, &AFF);
//     const uint32_t zero[4][4] = {
//         {0, 0, 0, 0},
//         {0, 0, 0, 0},
//         {0, 0, 0, 0},
//         {0, 0, 0, 0},
//     };
//     fail += report_test("SubBytes: identity affine, input 0 -> 0", state_equal_u32(&out, zero));

//     fp16_t one16;
//     fp16_init(&one16);
//     fp16_set_zero(&one16);
//     fp4_set_ui(&one16.x0, 1);
//     state_from_fp16(&in, &one16);
//     matrix_subbytes(&out, &in, &AFF);
//     fail += report_test("SubBytes: identity affine, input 1 -> 1", state_equal(&out, &in));
//     fp16_clear(&one16);

//     affine16_set_A(&AFF);
//     affine16_set_b(&AFF);
//     state_set_zero(&in);
//     matrix_subbytes(&out, &in, &AFF);
//     const uint32_t expected_b[4][4] = {
//         { 1,  2,  3,  4},
//         { 5,  6,  7,  8},
//         { 9, 10, 11, 12},
//         {13, 14, 15, 16},
//     };
//     fail += report_test("SubBytes: configured affine, input 0 -> b", state_equal_u32(&out, expected_b));

//     affine16_clear(&AFF);
//     state_clear(&in);
//     state_clear(&out);
//     return fail;
// }

// int main(void){
//     int fail = 0;

//     printf("Running matrix step tests...\n\n");
//     fail += test_add_round_constant();
//     fail += test_subbytes();
//     fail += test_shiftbytes();
//     fail += test_mixbytes();

//     printf("\nRESULT: %s\n", fail == 0 ? "ALL PASS" : "FAILED");
//     return fail ? 1 : 0;
// }

// #include "16_header.h"

// /* 16_header.h に宣言が無いが 16_matrix.c に定義がある関数の前方宣言 */
// void matrix_bytes_to_state(state_t *S, const uint8_t block[MATRIX_STATE_BYTES]);

// /* ============================================================
//  * MATRIX 主要処理の単体テスト
//  *   逆元 / SubBytes / MDS(MixBytes) / ShiftBytes / 置換P,Q /
//  *   バイト⇔行列変換 / 出力変換(31ビット詰め) / hash全体(決定性)
//  * 各項目を PASS/FAIL + 具体値で標準出力に表示する。
//  * ============================================================ */

// static int g_pass = 0, g_fail = 0;

// static void report(const char *name, int ok, const char *detail){
//     printf("[%s] %-34s %s\n", ok ? "PASS" : "FAIL", name, detail ? detail : "");
//     if(ok) g_pass++; else g_fail++;
// }

// /* MDS 行列 circ(1,1,2,8) をセット */
// static void setup_MDS(state_t *MDS){
//     state_init(MDS);
//     fp_set_ui(&MDS->m[0][0],1); fp_set_ui(&MDS->m[0][1],1); fp_set_ui(&MDS->m[0][2],2); fp_set_ui(&MDS->m[0][3],8);
//     fp_set_ui(&MDS->m[1][0],8); fp_set_ui(&MDS->m[1][1],1); fp_set_ui(&MDS->m[1][2],1); fp_set_ui(&MDS->m[1][3],2);
//     fp_set_ui(&MDS->m[2][0],2); fp_set_ui(&MDS->m[2][1],8); fp_set_ui(&MDS->m[2][2],1); fp_set_ui(&MDS->m[2][3],1);
//     fp_set_ui(&MDS->m[3][0],1); fp_set_ui(&MDS->m[3][1],2); fp_set_ui(&MDS->m[3][2],8); fp_set_ui(&MDS->m[3][3],1);
// }

// /* ---- 1. 逆元: a * inv(a) == 1, かつ別実装と一致 ---- */
// static void test_inverse(void){
//     int ok_id = 1, ok_cross = 1;
//     char buf[128];
//     for(int t = 0; t < 1000; t++){
//         fp16_t a, ia, prod, one;
//         fp16_random(&a);
//         fp16_inv(&ia, &a);
//         fp16_mul(&prod, &a, &ia);
//         /* one = a * inv(a) を 1 と比較するため、1 を作る: a*inv(a) は乗法単位元のはず */
//         /* fp16 の「1」を a*inv(a) 自身が示すので、別の元 b で b*1==b を確認 */
//         fp16_t b, chk;
//         fp16_random(&b);
//         fp16_mul(&chk, &b, &prod);   /* b * (a*inv(a)) == b なら prod==1 */
//         if(!fp16_is_equal(&chk, &b)) ok_id = 0;

//         /* 別実装 fp16_inv_slow と一致するか */
//         fp16_t ia2;
//         fp16_inv_slow(&ia2, &a);
//         if(!fp16_is_equal(&ia, &ia2)) ok_cross = 0;
//     }
//     snprintf(buf,sizeof(buf),"a*inv(a)=1 over 1000 random fp16 elements");
//     report("Inverse: a * inv(a) == 1", ok_id, buf);
//     report("Inverse: fp16_inv == fp16_inv_slow", ok_cross, "two implementations agree (1000 samples)");
// }

// /* ---- 2. SubBytes: 逆元+アフィン。アフィンが可逆 => 単射 ---- */
// static void test_subbytes(void){
//     affine16_t AFF; affine16_init(&AFF); affine16_set_A(&AFF); affine16_set_b(&AFF);
//     int reg = affine16_is_regular(&AFF);
//     report("SubBytes: affine layer is regular", reg,
//            reg ? "affine matrix A is invertible (bijective)" : "A is singular!");

//     /* 異なる入力 -> 異なる出力 (単射) を多数サンプルで確認 */
//     int injective = 1;
//     state_t prev; int have_prev = 0;
//     for(int t = 0; t < 200; t++){
//         state_t s, o1, o2;
//         state_init(&s); state_init(&o1); state_init(&o2);
//         state_random(&s);
//         matrix_subbytes(&o1, &s, &AFF);
//         matrix_subbytes(&o2, &s, &AFF);
//         /* 決定性: 同じ入力で同じ出力 */
//         uint8_t b1[64], b2[64];
//         matrix_state_to_bytes_512(b1,&o1); matrix_state_to_bytes_512(b2,&o2);
//         if(memcmp(b1,b2,64)!=0) injective = 0;
//         (void)prev; (void)have_prev;
//         state_clear(&s); state_clear(&o1); state_clear(&o2);
//     }
//     report("SubBytes: deterministic", injective, "same input -> same output (200 samples)");
//     affine16_clear(&AFF);
// }

// /* ---- 3. ShiftBytes: 全単射(置換)であること ---- */
// static void test_shiftbytes(void){
//     /* P, Q の ShiftBytes は単なるバイト並べ替え。state_random を入れ、
//        出力に含まれる要素集合が入力と同じ(並べ替えのみ)かを確認する簡易チェック。
//        ここでは「2回適用して周期で戻る」ではなく、決定性+非自明な移動を確認 */
//     state_t s, sp, sq;
//     state_init(&s); state_init(&sp); state_init(&sq);
//     state_random(&s);
//     matrix_shiftbytes_P(&sp, &s);
//     matrix_shiftbytes_Q(&sq, &s);
//     /* P と Q で結果が異なる(異なるシフト)ことを確認 */
//     uint8_t bp[64], bq[64], bs[64];
//     matrix_state_to_bytes_512(bp,&sp);
//     matrix_state_to_bytes_512(bq,&sq);
//     matrix_state_to_bytes_512(bs,&s);
//     int p_diff_q = (memcmp(bp,bq,64)!=0);
//     report("ShiftBytes: P and Q differ", p_diff_q, "different shift offsets for P vs Q");
//     state_clear(&s); state_clear(&sp); state_clear(&sq);
// }

// /* ---- 4. MDS(MixBytes): branch number = 5 (70個の小行列式が非零) ---- */
// /* fp 行列の det を mod p で計算するため、小行列を mpz 経由でなく fp 演算で */
// static void fp_mat_det(fp_t *det, fp_t M[4][4], int idx[], int k){
//     /* k<=4 の余因子展開 */
//     if(k==1){ fp_set(det, &M[idx[0]][idx[0]]); return; }
//     /* 実装簡略化のため、ここでは k=2 のみ厳密、k>=3 は再帰... 省略せず一般化 */
// }
// static void test_mds(void){
//     /* MDS判定: 全 k×k 小行列式 (k=1..4) が非零。
//        fp_t の行列式計算は煩雑なので、ここでは circ(1,1,2,8) を整数行列とみなし
//        mod p=2^31-1 で 70個の小行列式を直接計算する。 */
//     long p = (1L<<31)-1;
//     long row[4] = {1,1,2,8};
//     long M[4][4];
//     for(int i=0;i<4;i++) for(int j=0;j<4;j++) M[i][j]=row[(j-i+4)%4];

//     int comb[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
//     int comb3[4][3]={{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
//     int mds = 1; int checked=0;
//     /* k=1 */
//     for(int i=0;i<4;i++) for(int j=0;j<4;j++){ checked++; if(M[i][j]%p==0) mds=0; }
//     /* k=2 */
//     for(int a=0;a<6;a++) for(int b=0;b<6;b++){
//         int r0=comb[a][0],r1=comb[a][1],c0=comb[b][0],c1=comb[b][1];
//         long d=((M[r0][c0]%p)*(M[r1][c1]%p)-(M[r0][c1]%p)*(M[r1][c0]%p))%p;
//         d=((d%p)+p)%p; checked++; if(d==0) mds=0;
//     }
//     /* k=3 */
//     for(int a=0;a<4;a++) for(int b=0;b<4;b++){
//         int r[3]={comb3[a][0],comb3[a][1],comb3[a][2]};
//         int c[3]={comb3[b][0],comb3[b][1],comb3[b][2]};
//         long det=0;
//         int perm[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
//         int sign[6]={1,-1,-1,1,1,-1};
//         for(int pp=0;pp<6;pp++){
//             long term=sign[pp];
//             for(int x=0;x<3;x++) term=(term*(M[r[x]][c[perm[pp][x]]]%p))%p;
//             det=(det+term)%p;
//         }
//         det=((det%p)+p)%p; checked++; if(det==0) mds=0;
//     }
//     /* k=4 (全体) 余因子展開 */
//     {
//         long det=0; int sign[4]={1,-1,1,-1};
//         for(int c0=0;c0<4;c0++){
//             int sc[3],t=0; for(int j=0;j<4;j++) if(j!=c0) sc[t++]=j;
//             int sr[3]={1,2,3};
//             long m3=0; int perm[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
//             int s3[6]={1,-1,-1,1,1,-1};
//             for(int pp=0;pp<6;pp++){
//                 long term=s3[pp];
//                 for(int x=0;x<3;x++) term=(term*(M[sr[x]][sc[perm[pp][x]]]%p))%p;
//                 m3=(m3+term)%p;
//             }
//             det=(det + sign[c0]*(M[0][c0]%p)*(((m3%p)+p)%p))%p;
//         }
//         det=((det%p)+p)%p; checked++; if(det==0) mds=0;
//     }
//     char buf[96];
//     snprintf(buf,sizeof(buf),"all %d minors nonzero mod 2^31-1 (branch number = 5)", checked);
//     report("MDS(MixBytes): circ(1,1,2,8) is MDS", mds, buf);
// }

// /* ---- 5. 置換 P,Q: 決定性 + 単射(異なる入力->異なる出力) ---- */
// static void test_permutation(void){
//     state_t MDS; setup_MDS(&MDS);
//     affine16_t AFF; affine16_init(&AFF); affine16_set_A(&AFF); affine16_set_b(&AFF);

//     int det_ok = 1, inj_ok = 1;
//     uint8_t seen[16][64]; int nseen=0;
//     for(int t=0;t<16;t++){
//         state_t s, o1, o2;
//         state_init(&s); state_init(&o1); state_init(&o2);
//         state_random(&s);
//         matrix_permutation_P(&o1,&s,MATRIX_ROUNDS,&MDS,&AFF);
//         matrix_permutation_P(&o2,&s,MATRIX_ROUNDS,&MDS,&AFF);
//         uint8_t b1[64],b2[64];
//         matrix_state_to_bytes_512(b1,&o1);
//         matrix_state_to_bytes_512(b2,&o2);
//         if(memcmp(b1,b2,64)!=0) det_ok=0;       /* 決定性 */
//         for(int k=0;k<nseen;k++) if(memcmp(seen[k],b1,64)==0) inj_ok=0; /* 重複なし */
//         memcpy(seen[nseen++],b1,64);
//         state_clear(&s); state_clear(&o1); state_clear(&o2);
//     }
//     report("Permutation P: deterministic", det_ok, "same input -> same output (16 samples)");
//     report("Permutation P: no collisions", inj_ok, "16 distinct inputs -> 16 distinct outputs");
//     affine16_clear(&AFF); state_clear(&MDS);
// }

// /* ---- 6. バイト⇔行列変換: ラウンドトリップ ---- */
// static void test_byte_state_roundtrip(void){
//     int ok = 1;
//     char buf[96];
//     for(int t=0;t<100;t++){
//         uint8_t in[64], out[64];
//         for(int i=0;i<64;i++) in[i]=rand()&0xff;
//         /* 注意: 各 Fp 要素は p=2^31-1 未満である必要があるため、
//            入力バイトを 31bit 範囲に収める(各4バイトの最上位ビットを0に) */
//         for(int i=3;i<64;i+=4) in[i] &= 0x7f;
//         state_t s; state_init(&s);
//         matrix_bytes_to_state(&s, in);
//         matrix_state_to_bytes_512(out, &s);
//         /* bytes_to_state と state_to_bytes_512 の表現が異なる(後者は31bit詰め)ため
//            完全一致は仕様次第。ここでは「状態に正しく載るか」=値が保存されるかを確認 */
//         state_clear(&s);
//         /* 簡易: in->state->bytes が決定的に同じ結果を返すか(2回実行で一致) */
//         state_t s2; state_init(&s2); uint8_t out2[64];
//         matrix_bytes_to_state(&s2,in);
//         matrix_state_to_bytes_512(out2,&s2);
//         if(memcmp(out,out2,64)!=0) ok=0;
//         state_clear(&s2);
//     }
//     snprintf(buf,sizeof(buf),"bytes->state->bytes deterministic (100 samples)");
//     report("Byte<->State: deterministic", ok, buf);
// }

// /* ---- 7. 出力変換(31ビット詰め): 定数ビットが無い ---- */
// static void test_output_packing(void){
//     /* 多数のランダム状態を出力し、各ビット位置の OR を取る。
//        全ビットが少なくとも一度は1になる(=定数0ビットが無い)ことを確認。 */
//     uint8_t oracc[64]; memset(oracc,0,64);
//     for(int t=0;t<5000;t++){
//         state_t s; state_init(&s); state_random(&s);
//         uint8_t b[64];
//         matrix_state_to_bytes_512(b,&s);
//         for(int k=0;k<64;k++) oracc[k]|=b[k];
//         state_clear(&s);
//     }
//     int const_bits = 0;
//     for(int bit=0;bit<512;bit++){
//         int set=(oracc[bit>>3]>>(7-(bit&7)))&1;
//         if(!set) const_bits++;
//     }
//     char buf[96];
//     snprintf(buf,sizeof(buf),"%d constant bits out of 512 (digest uses first 256)", const_bits);
//     /* 31ビット詰めなら 496..511 の16ビットだけ0埋め。先頭256ビットに定数があってはいけない */
//     int head_const=0;
//     for(int bit=0;bit<256;bit++){ if(!((oracc[bit>>3]>>(7-(bit&7)))&1)) head_const++; }
//     report("Output packing: no constant bit in digest", head_const==0, buf);
// }

// /* ---- 8. hash全体: 決定性 + 入力1ビット変化で出力が変わる ---- */
// static void test_hash_overall(void){
//     state_t MDS; setup_MDS(&MDS);
//     affine16_t AFF; affine16_init(&AFF); affine16_set_A(&AFF); affine16_set_b(&AFF);

//     uint8_t msg[64]; for(int i=0;i<64;i++) msg[i]=rand()&0xff;
//     uint8_t d1[32], d2[32], d3[32];
//     matrix_hash(d1,32,msg,64,MATRIX_ROUNDS,&MDS,&AFF);
//     matrix_hash(d2,32,msg,64,MATRIX_ROUNDS,&MDS,&AFF);
//     int det_ok = (memcmp(d1,d2,32)==0);
//     report("Hash: deterministic", det_ok, "H(m) == H(m) for same input");

//     uint8_t msg2[64]; memcpy(msg2,msg,64); msg2[0]^=1;  /* 1ビット変える */
//     matrix_hash(d3,32,msg2,64,MATRIX_ROUNDS,&MDS,&AFF);
//     int diff_ok = (memcmp(d1,d3,32)!=0);
//     /* 何ビット変わったか表示 */
//     int flips=0; for(int i=0;i<32;i++) flips+=__builtin_popcount(d1[i]^d3[i]);
//     char buf[96]; snprintf(buf,sizeof(buf),"1-bit input change flips %d/256 output bits", flips);
//     report("Hash: input sensitivity", diff_ok, buf);

//     affine16_clear(&AFF); state_clear(&MDS);
// }

// int main(void){
//     srand(12345);
//     printf("========================================================\n");
//     printf(" MATRIX unit tests (per-operation correctness)\n");
//     printf("========================================================\n");

//     test_inverse();
//     test_subbytes();
//     test_shiftbytes();
//     test_mds();
//     test_permutation();
//     test_byte_state_roundtrip();
//     test_output_packing();
//     test_hash_overall();

//     printf("--------------------------------------------------------\n");
//     printf(" RESULT: %d passed, %d failed\n", g_pass, g_fail);
//     printf("========================================================\n");
//     return g_fail ? 1 : 0;
// }

// #include "16_header.h"
 
// /* ===== ブロック数を変えて雪崩効果を測定 =====
//    使い方: ./avalanche_blocks <num_blocks> <n_samples>
//    例:     ./avalanche_blocks 4 50000
//    出力:   sac_b<num_blocks>.csv  (SAC行列 1024*?? -> 各ブロックで入力ビット数が変わる)
//            hist_b<num_blocks>.csv (反転ビット数の分布)
//    注: SAC行列の入力ビット数 = num_blocks * MATRIX_BLOCK_BYTES * 8 で変動する。
//        ヒートマップは各ブロック数で縦サイズが変わるが描画側で吸収する。
// */
 
// #define OUT_BYTES  MATRIX_DIGEST_BYTES   /* 32 */
// #define OUT_BITS   (OUT_BYTES * 8)       /* 256 */
 
// static void setup_MDS(state_t *MDS){
//     state_init(MDS);
//     fp_set_ui(&MDS->m[0][0],1); fp_set_ui(&MDS->m[0][1],1); fp_set_ui(&MDS->m[0][2],2); fp_set_ui(&MDS->m[0][3],8);
//     fp_set_ui(&MDS->m[1][0],8); fp_set_ui(&MDS->m[1][1],1); fp_set_ui(&MDS->m[1][2],1); fp_set_ui(&MDS->m[1][3],2);
//     fp_set_ui(&MDS->m[2][0],2); fp_set_ui(&MDS->m[2][1],8); fp_set_ui(&MDS->m[2][2],1); fp_set_ui(&MDS->m[2][3],1);
//     fp_set_ui(&MDS->m[3][0],1); fp_set_ui(&MDS->m[3][1],2); fp_set_ui(&MDS->m[3][2],8); fp_set_ui(&MDS->m[3][3],1);
// }
 
// int main(int argc, char **argv){
//     int num_blocks = (argc > 1) ? atoi(argv[1]) : 2;
//     long n_samples = (argc > 2) ? atol(argv[2]) : 50000;
//     int input_bytes = num_blocks * MATRIX_BLOCK_BYTES;  /* 64 * blocks */
//     int in_bits = input_bytes * 8;
 
//     state_t MDS;  setup_MDS(&MDS);
//     affine16_t AFF; affine16_init(&AFF); affine16_set_A(&AFF); affine16_set_b(&AFF);
 
//     long *flip = calloc((size_t)in_bits * OUT_BITS, sizeof(long));
//     long *hist = calloc(OUT_BITS + 1, sizeof(long));
//     long total_flips = 0, total_trials = 0;
 
//     uint8_t *base = malloc(input_bytes);
//     uint8_t *mod  = malloc(input_bytes);
//     uint8_t h0[OUT_BYTES], h1[OUT_BYTES];
 
//     srand(2025 + num_blocks);  /* ブロック数ごとに種を変える */
 
//     for(long s = 0; s < n_samples; s++){
//         for(int b = 0; b < input_bytes; b++) base[b] = rand() & 0xff;
//         matrix_hash(h0, OUT_BYTES, base, input_bytes, MATRIX_ROUNDS, &MDS, &AFF);
 
//         for(int i = 0; i < in_bits; i++){
//             memcpy(mod, base, input_bytes);
//             mod[i >> 3] ^= (uint8_t)(1u << (i & 7));
//             matrix_hash(h1, OUT_BYTES, mod, input_bytes, MATRIX_ROUNDS, &MDS, &AFF);
 
//             int flips_this = 0;
//             for(int j = 0; j < OUT_BITS; j++){
//                 int bit = ((h0[j>>3] ^ h1[j>>3]) >> (j & 7)) & 1;
//                 if(bit){ flip[(size_t)i*OUT_BITS + j]++; flips_this++; }
//             }
//             total_flips += flips_this; total_trials++; hist[flips_this]++;
//         }
//         if((s+1) % 1000 == 0) fprintf(stderr, "  [blocks=%d] sample %ld/%ld\n", num_blocks, s+1, n_samples);
//     }
 
//     double avg = (double)total_flips / total_trials;
//     double max_dev = 0;
//     for(size_t k=0;k<(size_t)in_bits*OUT_BITS;k++){
//         double p=(double)flip[k]/n_samples; double d=p>0.5?p-0.5:0.5-p; if(d>max_dev)max_dev=d;
//     }
//     printf("blocks=%d input_bits=%d samples=%ld avg_ratio=%.5f max|P-0.5|=%.5f\n",
//            num_blocks, in_bits, n_samples, avg/OUT_BITS, max_dev);
 
//     char fn[64];
//     snprintf(fn,sizeof(fn),"sac_b%d.csv",num_blocks);
//     FILE *fs=fopen(fn,"w");
//     for(int i=0;i<in_bits;i++)
//         for(int j=0;j<OUT_BITS;j++)
//             fprintf(fs,"%.5f%s",(double)flip[(size_t)i*OUT_BITS+j]/n_samples,(j==OUT_BITS-1)?"\n":",");
//     fclose(fs);
 
//     snprintf(fn,sizeof(fn),"hist_b%d.csv",num_blocks);
//     FILE *fh=fopen(fn,"w");
//     fprintf(fh,"flipped_bits,count\n");
//     for(int k=0;k<=OUT_BITS;k++) fprintf(fh,"%d,%ld\n",k,hist[k]);
//     fclose(fh);
 
//     /* メタ情報 (描画側で使う) */
//     snprintf(fn,sizeof(fn),"meta_b%d.txt",num_blocks);
//     FILE *fm=fopen(fn,"w");
//     fprintf(fm,"blocks=%d\ninput_bits=%d\nsamples=%ld\navg_ratio=%.5f\nmax_dev=%.5f\n",
//             num_blocks,in_bits,n_samples,avg/OUT_BITS,max_dev);
//     fclose(fm);
 
//     free(flip); free(hist); free(base); free(mod);
//     return 0;
// }











    // state_set_zero(&S);

    // fp_set_ui(&S.m[0][0], 0);
    // fp_set_ui(&S.m[0][1], P_MERSENNE - 16);
    // fp_set_ui(&S.m[0][2], P_MERSENNE - 32);
    // fp_set_ui(&S.m[0][3], P_MERSENNE - 48);

    // fp_set_ui(&S.m[1][0], 0);
    // fp_set_ui(&S.m[1][1], 0);
    // fp_set_ui(&S.m[1][2], 0);
    // fp_set_ui(&S.m[1][3], 0);

    // fp_set_ui(&S.m[2][0], 0);
    // fp_set_ui(&S.m[2][1], 0);
    // fp_set_ui(&S.m[2][2], 0);
    // fp_set_ui(&S.m[2][3], 0);

    // fp_set_ui(&S.m[3][0], 0);
    // fp_set_ui(&S.m[3][1], 0);
    // fp_set_ui(&S.m[3][2], 0);
    // fp_set_ui(&S.m[3][3], 0);

    // printf("S :\n");
    // state_print(&S);

    // test_diffusion_P(&S, 0, 3, 1, &MDS, &AFF, MATRIX_ROUNDS);



    // fp4_t A, B, C, D, E, F, One, Zero;
    // fp16_t A16, B16, C16, D16, E16, One16, Beta16, Alpha16, inv16, res16, A16_saved;
    // fp4_init(&A); fp4_init(&B); fp4_init(&C);
    // fp4_init(&D); fp4_init(&E); fp4_init(&F);
    // fp4_init(&One); fp4_init(&Zero);
    // fp16_init(&A16); fp16_init(&B16); fp16_init(&C16);
    // fp16_init(&D16); fp16_init(&E16); fp16_init(&One16);
    // fp16_init(&Beta16); fp16_init(&Alpha16); fp16_init(&inv16); fp16_init(&res16);
    // fp16_init(&A16_saved);

    // // テストデータをランダムに生成
    // fp4_random(&A);
    // fp4_random(&B);
    // fp4_random(&C);
    // fp16_random(&A16);
    // fp16_random(&B16);
    // fp16_random(&C16);

    // printf("--- Variables ---\n");
    // printf("A = "); fp4_printf(&A);
    // printf("B = "); fp4_printf(&B);

    // // 1. 加減算のテスト: (A + B) - B == A
    // printf("\n[Test 1] Addition & Subtraction: (A + B) - B == A ... ");
    // fp4_add(&D, &A, &B); // D = A + B
    // fp4_sub(&E, &D, &B); // E = D - B
    // if (fp4_is_equal(&A, &E)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    //     printf("Expected: "); fp4_printf(&A);
    //     printf("Result:   "); fp4_printf(&E);
    // }

    // // 2. 乗算の可換性: A * B == B * A
    // printf("[Test 2] Commutativity: A * B == B * A ... ");
    // fp4_mul(&D, &A, &B); // D = A * B
    // fp4_mul(&E, &B, &A); // E = B * A
    // if (fp4_is_equal(&D, &E)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    // }

    // // 3. 分配法則: A * (B + C) == A * B + A * C
    // printf("[Test 3] Distributivity: A * (B + C) == AB + AC ... ");
    // fp4_add(&D, &B, &C); // D = B + C
    // fp4_mul(&E, &A, &D); // E = A * (B + C)

    // fp4_mul(&D, &A, &B); // D = AB
    // fp4_mul(&F, &A, &C); // F = AC
    // fp4_add(&F, &D, &F); // F = AB + AC

    // if (fp4_is_equal(&E, &F)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    // }

    // // 4. 自乗の整合性: sqr(A) == mul(A, A)
    // printf("[Test 4] Squaring: A^2 == A * A ... ");
    // fp4_sqr(&D, &A);     // D = A^2 (sqr)
    // fp4_mul(&E, &A, &A); // E = A * A (mul)
    // if (fp4_is_equal(&D, &E)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    // }

    // // 5. 単位元のテスト: A * 1 == A
    // printf("[Test 5] Identity: A * 1 == A ... ");
    // fp4_set_ui(&One, 1); // One = 1
    // fp4_mul(&D, &A, &One);
    // if (fp4_is_equal(&D, &A)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    //     printf("1 is represented as: "); fp4_printf(&One);
    //     printf("A * 1 = "); fp4_printf(&D);
    // }

    // // 6. Frobenius写像の巡回性: A^(p^4) == A
    // // 正規基底では p乗 は巡回シフトなので、4回やると元に戻るはず
    // printf("[test6] Frobenius Cycle: A^(p^4) == A ... ");
    // fp4_set(&D, &A);
    // for(int i=0; i<4; i++){
    //     fp4_frobenius_map(&D, &D); // D = D^p
    //     // printf("  Apply %d: ", i+1); fp4_printf(&D); // デバッグ用
    // }
    // if (fp4_is_equal(&D, &A)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    //     printf("Original: "); fp4_printf(&A);
    //     printf("Result:   "); fp4_printf(&D);
    // }

    // // 7. 逆元のテスト: A * A^(-1) == 1
    // printf("[Test 7] Inversion: A * A^(-1) == 1 ... ");
    // fp4_inv(&D, &A);     // D = A^(-1)
    // fp4_mul(&E, &A, &D); // E = A * A^(-1)
    // fp4_set_ui(&One, 1); // 1
    
    // if (fp4_is_equal(&E, &One)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    //     printf("Result: "); fp4_printf(&E);
    // }

    // // 8. 位数チェック: ord(A) と A^ord == 1
    // printf("[Test 8] fp4 multiplicative order of A ... ");
    // if (!fp_is_zero(&A.x0) || !fp_is_zero(&A.x1) || !fp_is_zero(&A.x2) || !fp_is_zero(&A.x3)) {
    //     mpz_t ord, group;
    //     mpz_init(ord); mpz_init(group);
    //     fp4_order(ord, &A);
    //     fp4_pow(&D, &A, ord);
    //     fp4_set_ui(&One, 1);
    //     int ok = fp4_is_equal(&D, &One);
    //     mpz_init_set_ui(group, P_MERSENNE);
    //     mpz_pow_ui(group, group, 4);
    //     mpz_sub_ui(group, group, 1);
    //     int divides = mpz_divisible_p(group, ord);
    //     gmp_printf("ord(A) = %Zd , p^4-1 = %Zd ... %s%s\n",
    //         ord, group,
    //         ok ? "A^ord=1 OK" : "A^ord!=1 NG",
    //         divides ? "" : " (ord not dividing p^4-1!)");
    //     mpz_clear(ord); mpz_clear(group);
    // } else {
    //     printf("A is zero, skipped.\n");
    // }

    // // 9. α（拡大用の元）の位数も表示しておく
    // printf("[Test 9] fp4 multiplicative order of alpha ... ");
    // mpz_t ord_alpha, two_pow, r, group_alpha;
    // mpz_inits(ord_alpha, two_pow, r, group_alpha, NULL);
    // fp4_order(ord_alpha, &alpha);
    // mpz_set_ui(group_alpha, P_MERSENNE);
    // mpz_pow_ui(group_alpha, group_alpha, 4);
    // mpz_sub_ui(group_alpha, group_alpha, 1);
    // int divides_alpha = mpz_divisible_p(group_alpha, ord_alpha);
    // gmp_printf("ord(alpha) = %Zd , p^4-1 = %Zd%s\n",
    //     ord_alpha, group_alpha,
    //     divides_alpha ? "" : " (ord not dividing p^4-1!)");
    // // 参考: 2^33で割った商と余り
    // mpz_set_ui(two_pow, 2);
    // mpz_pow_ui(two_pow, two_pow, 33);
    // mpz_tdiv_qr(r, two_pow, ord_alpha, two_pow); // r=ord/2^33, two_pow=余り
    // gmp_printf("ord(alpha)/2^33 = %Zd, rem = %Zd\n", r, two_pow);
    // mpz_clears(ord_alpha, two_pow, r, group_alpha, NULL);

    // // 10. スカラーのべき乗結果がスカラーかチェック
    // printf("[Test 10] Scalar pow keeps scalar (fp4) ... ");
    // mpz_t scalar_exp4;
    // mpz_init_set_ui(scalar_exp4, 12345);
    // fp4_set_ui(&A, 7); // スカラーをセット
    // fp4_pow(&D, &A, scalar_exp4);
    // if (fp4_is_scalar(&D)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    //     printf("Result: "); fp4_printf(&D);
    // }
    // mpz_clear(scalar_exp4);

    // // //8. 4乗非剰余を探索しつつ x^4 - alpha の既約性も表示
    // // printf("[Test 8] Search 4th-nonresidue (100 trials) with irreducibility check...\n");
    // // fp4_quartic_residue_scan(100);

    // // --- fp16 tests (拡大体 Fp^16 = Fp^4[β]/(β^4-α)) ---
    // printf("\n--- fp16 Variables ---\n");
    // printf("A16 = "); fp16_printf(&A16);
    // printf("B16 = "); fp16_printf(&B16);

    // // 1. 加減算: (A+B)-B == A
    // printf("[fp16 Test 1] Addition & Subtraction ... ");
    // fp16_add(&D16, &A16, &B16);
    // fp16_sub(&E16, &D16, &B16);
    // if (fp16_is_equal(&A16, &E16)) printf("OK\n"); else printf("NG\n");

    // // 2. 乗算の可換性
    // printf("[fp16 Test 2] Commutativity ... ");
    // fp16_mul(&D16, &A16, &B16);
    // fp16_mul(&E16, &B16, &A16);
    // if (fp16_is_equal(&D16, &E16)) printf("OK\n"); else printf("NG\n");

    // // 3. 分配法則
    // printf("[fp16 Test 3] Distributivity ... ");
    // fp16_add(&D16, &B16, &C16);         // D16 = B+C
    // fp16_mul(&E16, &A16, &D16);         // E16 = A*(B+C)
    // fp16_mul(&D16, &A16, &B16);         // D16 = AB
    // fp16_mul(&Beta16, &A16, &C16);      // reuse Beta16 as tmp = AC
    // fp16_add(&Beta16, &D16, &Beta16);   // Beta16 = AB + AC
    // if (fp16_is_equal(&E16, &Beta16)) printf("OK\n"); else printf("NG\n");

    // // 4. 自乗
    // printf("[fp16 Test 4] Squaring ... ");
    // fp16_sqr(&D16, &A16);
    // fp16_mul(&E16, &A16, &A16);
    // if (fp16_is_equal(&D16, &E16)) printf("OK\n"); else printf("NG\n");

    // // 5. 単位元: (1,0,0,0)
    // fp4_set_ui(&One, 1);
    // fp4_set_ui(&One16.x1, 0); fp4_set_ui(&One16.x2, 0); fp4_set_ui(&One16.x3, 0);
    // fp4_set(&One16.x0, &One);
    // printf("[fp16 Test 5] Identity ... ");
    // fp16_mul(&D16, &A16, &One16);
    // if (fp16_is_equal(&D16, &A16)) printf("OK\n"); else { printf("NG\n"); printf("1 = "); fp16_printf(&One16); }

    // // 6. β^4 = α のチェック (β = (0,1,0,0))
    // fp4_set_ui(&Beta16.x0, 0); fp4_set_ui(&Beta16.x2, 0); fp4_set_ui(&Beta16.x3, 0);
    // fp4_set(&Beta16.x1, &One); // β の係数は1
    // fp4_set_ui(&Alpha16.x1, 0); fp4_set_ui(&Alpha16.x2, 0); fp4_set_ui(&Alpha16.x3, 0);
    // fp4_set(&Alpha16.x0, &alpha); // α を定数項に
    // fp16_set(&D16, &Beta16);
    // for(int i=0;i<3;i++) fp16_mul(&D16, &D16, &Beta16); // D16 = β^4
    // printf("[fp16 Test 6] beta^4 == alpha ... ");
    // if (fp16_is_equal(&D16, &Alpha16)) printf("OK\n"); else { printf("NG\n"); printf("beta^4 = "); fp16_printf(&D16); }

    // // 7. スカラーのべき乗結果がスカラーかチェック
    // printf("[fp16 Test 7] Scalar pow keeps scalar (fp16) ... ");
    // mpz_t scalar_exp16;
    // mpz_init_set_ui(scalar_exp16, 12345);
    // fp16_set(&A16_saved, &A16); // 元のランダム値を保存
    // fp4_set_ui(&A16.x0, 5); // Fp4 スカラー
    // fp4_set_ui(&A16.x1, 0); fp4_set_ui(&A16.x2, 0); fp4_set_ui(&A16.x3, 0);
    // fp16_pow(&D16, &A16, scalar_exp16);
    // if (fp16_is_scalar(&D16)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    //     printf("Result: "); fp16_printf(&D16);
    // }
    // mpz_clear(scalar_exp16);
    // fp16_set(&A16, &A16_saved); // 以降のテスト用に元に戻す

    // fp16_inv(&inv16,&A16);
    // printf("1/A16 = "); fp16_printf(&inv16);

    // fp16_mul(&res16,&A16,&inv16);

    // printf("A16*1/A16 = "); fp16_printf(&res16);


    // int iters = 100;
    // bench_fp_add_avg(iters, 1000000);
    // bench_fp_sub_avg(iters, 1000000);
    // bench_fp_mul_avg(iters, 1000000);
    // bench_fp_add_plus_avg(iters, 1000000);
    // bench_fp_sub_plus_avg(iters, 1000000);
    // bench_fp_mul_plus_avg(iters, 1000000);

    // //--- Benchmark fp4_mul ---
    // bench_fp4_mul_avg(iters, 1000000);

    // // --- Benchmark fp4_mul_new ---
    // bench_fp4_mul_new_avg(iters, 1000000);

    // // --- Benchmark fp4_mul_karatsuba ---
    // bench_fp4_mul_karatsuba_avg(iters, 1000000);

    // // --- Benchmark fp4_mul_slow ---
    // bench_fp4_mul_slow_avg(iters, 1000000);

    // // --- Benchmark fp16_inv ---
    // int inv_iters = 100;
    // bench_fp16_inv_avg(inv_iters, 1000000);

    // // --- Benchmark fp16_inv_new ---
    // bench_fp16_inv_new_avg(inv_iters, 1000000);

    // // --- Benchmark fp16_inv_karatsuba ---
    // bench_fp16_inv_karatsuba_avg(inv_iters, 1000000);

    // // --- Benchmark fp16_inv_slow ---
    // bench_fp16_inv_slow_avg(inv_iters, 1000000);
