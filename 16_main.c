#include "16_header.h"
int main(void){
    uint32_t p_mer = P_MERSENNE;
    uint32_t p_pri = P_PRIME;
    // printf("p_mer : %u\n",p_mer); //この段階では規定値が入っている
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

    uint8_t block[MATRIX_BLOCK_BYTES_MAX] = {0};
    uint8_t digest[MATRIX_DIGEST_BYTES_MAX];
    uint8_t msg[] = "asahi"; //文字列リテラル "Hello World" の各文字コードが uint8_t 配列に格納される
    size_t msg_len = sizeof(msg) -1; //末尾のヌル文字\0も数えてしまい、1文字多くなってしまうから1を引く
    size_t digest_len =32; //ここで出力長を決定（これはバイト長）8以下を選べばp=2^7-1が使われる

    printf("MDS :\n");
    state_print(&MDS);

    printf("msg :\n");
    printf("%s\n",msg);
    // print_bytes_hex(msg, 11);

    //throughput_matrix_hash(msg, msg_len, 100, &MDS, &AFF);

    matrix_hash(digest, digest_len, msg, msg_len, MATRIX_ROUNDS, &MDS, &AFF);

    printf("digest :\n");
    print_bytes_hex(digest, digest_len);

    state_clear(&MDS);
    affine16_clear(&AFF);

    fp4_t X;
    fp4_init(&X);
    fp_set_ui(&X.x0,1);
    fp_set_ui(&X.x1,0);
    fp_set_ui(&X.x2,0);
    fp_set_ui(&X.x3,6);
    if(!fp4_is_square(&X)){
        printf("OK\n");
    }else{
        printf("f**k\n");
    }
    fp4_clear(&X);
    return 0;
}

/* ============================================================
 * Fp / Fp4 / Fp16 体演算の正当性テスト
 *   ランダムな元を用いて、加減乗除・可換性・分配則・単位元・
 *   逆元・べき乗・複数実装間の一致などを確認する。
 * ============================================================ */

// #define FIELD_TEST_TRIALS 200

// static int g_pass = 0, g_fail = 0;

// static void report(const char *name, int ok){
//     printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
//     if(ok) g_pass++; else g_fail++;
// }

// /* ---------------- Fp ---------------- */
// static void test_fp(void){
//     int ok;

//     printf("\n=== Fp tests (%d random trials each) ===\n", FIELD_TEST_TRIALS);

//     /* 加減算: (a+b)-b == a */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp_t a, b, s, d;
//         fp_random(&a); fp_random(&b);
//         fp_add(&s, &a, &b);
//         fp_sub(&d, &s, &b);
//         if(!fp_is_equal(&d, &a)){ ok = 0; break; }
//     }
//     report("Fp: (a+b)-b == a", ok);

//     /* 加算・乗算の可換性 */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp_t a, b, s1, s2, m1, m2;
//         fp_random(&a); fp_random(&b);
//         fp_add(&s1, &a, &b); fp_add(&s2, &b, &a);
//         fp_mul(&m1, &a, &b); fp_mul(&m2, &b, &a);
//         if(!fp_is_equal(&s1, &s2) || !fp_is_equal(&m1, &m2)){ ok = 0; break; }
//     }
//     report("Fp: a+b==b+a, a*b==b*a", ok);

//     /* 分配法則: a*(b+c) == a*b + a*c */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp_t a, b, c, lhs, tmp1, tmp2, rhs;
//         fp_random(&a); fp_random(&b); fp_random(&c);
//         fp_add(&tmp1, &b, &c);
//         fp_mul(&lhs, &a, &tmp1);
//         fp_mul(&tmp1, &a, &b);
//         fp_mul(&tmp2, &a, &c);
//         fp_add(&rhs, &tmp1, &tmp2);
//         if(!fp_is_equal(&lhs, &rhs)){ ok = 0; break; }
//     }
//     report("Fp: a*(b+c) == a*b + a*c", ok);

//     /* 単位元: a*1 == a, a+0 == a */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp_t a, one, zero, r1, r2;
//         fp_random(&a);
//         fp_set_ui(&one, 1);
//         fp_set_zero(&zero);
//         fp_mul(&r1, &a, &one);
//         fp_add(&r2, &a, &zero);
//         if(!fp_is_equal(&r1, &a) || !fp_is_equal(&r2, &a)){ ok = 0; break; }
//     }
//     report("Fp: a*1==a, a+0==a", ok);

//     /* 逆元: a * inv(a) == 1 (a != 0) */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp_t a, ia, prod, one;
//         do{ fp_random(&a); } while(fp_is_zero(&a));
//         fp_inv(&ia, &a);
//         fp_mul(&prod, &a, &ia);
//         fp_set_ui(&one, 1);
//         if(!fp_is_equal(&prod, &one)){ ok = 0; break; }
//     }
//     report("Fp: a * inv(a) == 1", ok);

//     /* フェルマーの小定理: a^(p-1) == 1 (a != 0) */
//     ok = 1;
//     {
//         mpz_t exp;
//         mpz_init_set_ui(exp, P_MERSENNE);
//         mpz_sub_ui(exp, exp, 1);
//         for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//             fp_t a, r, one;
//             do{ fp_random(&a); } while(fp_is_zero(&a));
//             fp_pow(&r, &a, exp);
//             fp_set_ui(&one, 1);
//             if(!fp_is_equal(&r, &one)){ ok = 0; break; }
//         }
//         mpz_clear(exp);
//     }
//     report("Fp: a^(p-1) == 1 (Fermat)", ok);

//     /* 平方剰余: legendre(a)==1 なら sqrt(a)^2 == a */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp_t a, root, sq;
//         do{ fp_random(&a); } while(fp_is_zero(&a));
//         if(fp_legendre(&a) != 1) continue;
//         if(!fp_sqrt(&root, &a)){ ok = 0; break; }
//         fp_mul(&sq, &root, &root);
//         if(!fp_is_equal(&sq, &a)){ ok = 0; break; }
//     }
//     report("Fp: legendre(a)==1 => sqrt(a)^2 == a", ok);
// }

// /* ---------------- Fp4 ---------------- */
// static void test_fp4(void){
//     int ok;

//     printf("\n=== Fp4 tests (%d random trials each) ===\n", FIELD_TEST_TRIALS);

//     /* 加減算: (A+B)-B == A */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp4_t A, B, S, D;
//         fp4_random(&A); fp4_random(&B);
//         fp4_add(&S, &A, &B);
//         fp4_sub(&D, &S, &B);
//         if(!fp4_is_equal(&D, &A)){ ok = 0; break; }
//     }
//     report("Fp4: (A+B)-B == A", ok);

//     /* 乗算の可換性 */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp4_t A, B, M1, M2;
//         fp4_random(&A); fp4_random(&B);
//         fp4_mul(&M1, &A, &B);
//         fp4_mul(&M2, &B, &A);
//         if(!fp4_is_equal(&M1, &M2)){ ok = 0; break; }
//     }
//     report("Fp4: A*B == B*A", ok);

//     /* 分配法則 */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp4_t A, B, C, lhs, t1, t2, rhs;
//         fp4_random(&A); fp4_random(&B); fp4_random(&C);
//         fp4_add(&t1, &B, &C);
//         fp4_mul(&lhs, &A, &t1);
//         fp4_mul(&t1, &A, &B);
//         fp4_mul(&t2, &A, &C);
//         fp4_add(&rhs, &t1, &t2);
//         if(!fp4_is_equal(&lhs, &rhs)){ ok = 0; break; }
//     }
//     report("Fp4: A*(B+C) == A*B + A*C", ok);

//     /* 自乗: sqr(A) == A*A */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp4_t A, S1, S2;
//         fp4_random(&A);
//         fp4_sqr(&S1, &A);
//         fp4_mul(&S2, &A, &A);
//         if(!fp4_is_equal(&S1, &S2)){ ok = 0; break; }
//     }
//     report("Fp4: sqr(A) == A*A", ok);

//     /* 単位元 */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp4_t A, one, r;
//         fp4_random(&A);
//         fp4_set_ui(&one, 1);
//         fp4_mul(&r, &A, &one);
//         if(!fp4_is_equal(&r, &A)){ ok = 0; break; }
//     }
//     report("Fp4: A*1 == A", ok);

//     /* 複数の乗算実装の一致: mul, mul_new, mul_karatsuba, mul_slow */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp4_t A, B, r1, r2, r3, r4;
//         fp4_random(&A); fp4_random(&B);
//         fp4_mul(&r1, &A, &B);
//         fp4_mul_new(&r2, &A, &B);
//         fp4_mul_karatsuba(&r3, &A, &B);
//         fp4_mul_slow(&r4, &A, &B);
//         if(!fp4_is_equal(&r1, &r2) || !fp4_is_equal(&r1, &r3) || !fp4_is_equal(&r1, &r4)){ ok = 0; break; }
//     }
//     report("Fp4: mul == mul_new == mul_karatsuba == mul_slow", ok);

//     /* Frobenius写像の巡回性: A^(p^4) == A (正規基底では4回のp乗写像で戻る) */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp4_t A, D;
//         fp4_random(&A);
//         fp4_set(&D, &A);
//         for(int i = 0; i < 4; i++) fp4_frobenius_map(&D, &D);
//         if(!fp4_is_equal(&D, &A)){ ok = 0; break; }
//     }
//     report("Fp4: Frobenius cycle A^(p^4) == A", ok);

//     /* 逆元: A * inv(A) == 1、複数実装の一致 */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp4_t A, iA, prod, one;
//         do{ fp4_random(&A); } while(fp4_is_zero(&A));
//         fp4_inv(&iA, &A);
//         fp4_mul(&prod, &A, &iA);
//         fp4_set_ui(&one, 1);
//         if(!fp4_is_equal(&prod, &one)){ ok = 0; break; }

//         fp4_t iA_slow, iA_karatsuba, iA_new;
//         fp4_inv_slow(&iA_slow, &A);
//         fp4_inv_karatsuba(&iA_karatsuba, &A);
//         fp4_inv_new(&iA_new, &A);
//         if(!fp4_is_equal(&iA, &iA_slow) || !fp4_is_equal(&iA, &iA_karatsuba) || !fp4_is_equal(&iA, &iA_new)){ ok = 0; break; }
//     }
//     report("Fp4: A*inv(A)==1, inv==inv_slow==inv_karatsuba==inv_new", ok);

//     /* 平方数の判定: (A*A) は常に平方数のはず */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp4_t A, S;
//         do{ fp4_random(&A); } while(fp4_is_zero(&A));
//         fp4_sqr(&S, &A);
//         if(!fp4_is_square(&S)){ ok = 0; break; }
//     }
//     report("Fp4: A*A is always a square", ok);
// }

// /* ---------------- Fp16 ---------------- */
// static void test_fp16(void){
//     int ok;

//     printf("\n=== Fp16 tests (%d random trials each) ===\n", FIELD_TEST_TRIALS);

//     /* 加減算: (A+B)-B == A */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp16_t A, B, S, D;
//         fp16_random(&A); fp16_random(&B);
//         fp16_add(&S, &A, &B);
//         fp16_sub(&D, &S, &B);
//         if(!fp16_is_equal(&D, &A)){ ok = 0; break; }
//     }
//     report("Fp16: (A+B)-B == A", ok);

//     /* 乗算の可換性 */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp16_t A, B, M1, M2;
//         fp16_random(&A); fp16_random(&B);
//         fp16_mul(&M1, &A, &B);
//         fp16_mul(&M2, &B, &A);
//         if(!fp16_is_equal(&M1, &M2)){ ok = 0; break; }
//     }
//     report("Fp16: A*B == B*A", ok);

//     /* 分配法則 */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp16_t A, B, C, lhs, t1, t2, rhs;
//         fp16_random(&A); fp16_random(&B); fp16_random(&C);
//         fp16_add(&t1, &B, &C);
//         fp16_mul(&lhs, &A, &t1);
//         fp16_mul(&t1, &A, &B);
//         fp16_mul(&t2, &A, &C);
//         fp16_add(&rhs, &t1, &t2);
//         if(!fp16_is_equal(&lhs, &rhs)){ ok = 0; break; }
//     }
//     report("Fp16: A*(B+C) == A*B + A*C", ok);

//     /* 自乗: sqr(A) == A*A */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp16_t A, S1, S2;
//         fp16_random(&A);
//         fp16_sqr(&S1, &A);
//         fp16_mul(&S2, &A, &A);
//         if(!fp16_is_equal(&S1, &S2)){ ok = 0; break; }
//     }
//     report("Fp16: sqr(A) == A*A", ok);

//     /* 単位元: (1,0,0,0) */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp16_t A, one, r;
//         fp16_random(&A);
//         fp16_set_zero(&one);
//         fp4_set_ui(&one.x0, 1);
//         fp16_mul(&r, &A, &one);
//         if(!fp16_is_equal(&r, &A)){ ok = 0; break; }
//     }
//     report("Fp16: A*1 == A", ok);

//     /* 複数の乗算実装の一致: mul, mul_new, mul_karatsuba, mul_slow */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp16_t A, B, r1, r2, r3, r4;
//         fp16_random(&A); fp16_random(&B);
//         fp16_mul(&r1, &A, &B);
//         fp16_mul_new(&r2, &A, &B);
//         fp16_mul_karatsuba(&r3, &A, &B);
//         fp16_mul_slow(&r4, &A, &B);
//         if(!fp16_is_equal(&r1, &r2) || !fp16_is_equal(&r1, &r3) || !fp16_is_equal(&r1, &r4)){ ok = 0; break; }
//     }
//     report("Fp16: mul == mul_new == mul_karatsuba == mul_slow", ok);

//     /* mul_sparse: A * (B_fp4 を x0 に埋め込んだ Fp16) == mul_sparse(A, B_fp4) */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp16_t A, Bfull, r1, r2, r3, r4;
//         fp4_t B4;
//         fp16_random(&A);
//         fp4_random(&B4);
//         fp16_set_zero(&Bfull);
//         fp4_set(&Bfull.x0, &B4);

//         fp16_mul(&r1, &A, &Bfull);
//         fp16_mul_sparse(&r2, &A, &B4);
//         fp16_mul_sparse_slow(&r3, &A, &B4);
//         fp16_mul_sparse_karatsuba(&r4, &A, &B4);
//         if(!fp16_is_equal(&r1, &r2) || !fp16_is_equal(&r1, &r3) || !fp16_is_equal(&r1, &r4)){ ok = 0; break; }
//     }
//     report("Fp16: mul_sparse family agree with full mul by embedded Fp4", ok);

//     /* beta^4 == alpha (beta = (0,1,0,0)) */
//     {
//         fp16_t beta, alpha16, D;
//         fp16_set_zero(&beta);
//         fp4_t one4; fp4_set_ui(&one4, 1);
//         fp4_set(&beta.x1, &one4);
//         fp16_set_zero(&alpha16);
//         fp4_set(&alpha16.x0, &alpha);
//         fp16_set(&D, &beta);
//         for(int i = 0; i < 3; i++) fp16_mul(&D, &D, &beta);
//         report("Fp16: beta^4 == alpha", fp16_is_equal(&D, &alpha16));
//     }

//     /* 逆元: A * inv(A) == 1、複数実装の一致 */
//     ok = 1;
//     for(int t = 0; t < FIELD_TEST_TRIALS; t++){
//         fp16_t A, iA, prod, one;
//         do{ fp16_random(&A); } while(fp16_is_zero(&A));
//         fp16_inv(&iA, &A);
//         fp16_mul(&prod, &A, &iA);
//         fp16_set_zero(&one);
//         fp4_set_ui(&one.x0, 1);
//         if(!fp16_is_equal(&prod, &one)){ ok = 0; break; }

//         fp16_t iA_new, iA_slow, iA_karatsuba;
//         fp16_inv_new(&iA_new, &A);
//         fp16_inv_slow(&iA_slow, &A);
//         fp16_inv_karatsuba(&iA_karatsuba, &A);
//         if(!fp16_is_equal(&iA, &iA_new) || !fp16_is_equal(&iA, &iA_slow) || !fp16_is_equal(&iA, &iA_karatsuba)){ ok = 0; break; }
//     }
//     report("Fp16: A*inv(A)==1, inv==inv_new==inv_slow==inv_karatsuba", ok);
// }

// int main(void){
//     printf("========================================================\n");
//     printf(" Fp / Fp4 / Fp16 field arithmetic tests\n");
//     printf("========================================================\n");

//     test_fp();
//     test_fp4();
//     test_fp16();

//     printf("\n--------------------------------------------------------\n");
//     printf(" RESULT: %d passed, %d failed\n", g_pass, g_fail);
//     printf("========================================================\n");
//     return g_fail ? 1 : 0;
// }