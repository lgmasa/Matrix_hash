/*
 * 16_field_test.c  ---  fp / fp4 / fp16 の体の算術が正しいか確認するテスト
 *
 *  各素数(出力長で選択)について次を確認する:
 *   0) field_select 後の P_MERSENNE が 2^q-1 になっているか(1u<<127 バグ検出)
 *   1) fp   : 非ゼロ a について a * a^{-1} == 1
 *   2) fp   : 平方剰余 a について (sqrt(a))^2 == a
 *   3) fp4  : 非ゼロ A について A * A^{-1} == 1
 *   4) fp16 : 非ゼロ X について X * X^{-1} == 1   (SubBytes の逆元経路)
 *
 *  7/31 は既に動くはずなので正常系の対照、127 が本命。
 *
 * build: gcc -O2 16_field_test.c 16_fp.c 16_fp4.c 16_fp16.c 16_matrix.c -lgmp -o 16_field_test
 *   (matrix.c は field_select_for_output などの依存で必要なら含める。
 *    リンクエラーが出る場合は未定義シンボルのある .c を足す。main 重複を避けるため
 *    16_main.c は含めないこと。)
 * run:   ./16_field_test
 */
#include "16_header.h"
#include <stdio.h>
#include <stdint.h>

typedef unsigned __int128 u128_t;

static void print_u128(u128_t v){
    unsigned long long hi=(unsigned long long)(v>>64), lo=(unsigned long long)v;
    if(hi) printf("0x%llx%016llx", hi, lo);
    else   printf("%llu", lo);
}

/* fp16 の単位元 1 を作る (x0=coeff of 1 に fp4 の 1, 残りは 0) */
static void make_fp16_one(fp16_t *o){
    fp16_set_zero(o);
    fp4_set_ui(&o->x0, 1);
}

static int test_one_prime(size_t n_bits_for_select, int trials){
    if(!field_select_for_output(n_bits_for_select)){
        printf("  field_select_for_output(%zu) 失敗\n", n_bits_for_select);
        return 0;
    }
    printf("== FP_BITS=%u (出力長 %zu bit を要求) ==\n", FP_BITS, n_bits_for_select);

    /* 0) P_MERSENNE == 2^q - 1 か */
    u128_t expect = (((u128_t)1) << FP_BITS) - 1;
    u128_t got    = (u128_t)P_MERSENNE;   /* P_MERSENNE が uint32 のままだと 127 でここが不一致になる */
    printf("   P_MERSENNE = "); print_u128(got);
    printf("   期待 = ");        print_u128(expect);
    printf("  -> %s\n", (got==expect) ? "OK" : "NG(型/生成が127非対応)");
    if(got != expect) return 0;   /* 素数が壊れていたら以降は無意味 */

    int ok = 1;

    /* 1) fp: a * a^-1 == 1 */
    {
        fp_t one; fp_set_ui(&one, 1);
        int fail=0;
        for(int i=0;i<trials;i++){
            fp_t a; fp_random(&a);
            if(fp_is_zero(&a)) continue;
            fp_t inv, prod;
            fp_inv(&inv, &a);
            fp_mul(&prod, &a, &inv);
            if(!fp_is_equal(&prod, &one)){ fail++; }
        }
        printf("   [fp]   a*a^-1==1        : %s\n", fail?"NG":"OK");
        ok &= (fail==0);
    }

    /* 2) fp: sqrt(a)^2 == a (a が平方剰余のとき) */
    {
        int tested=0, fail=0;
        for(int i=0;i<trials;i++){
            fp_t a; fp_random(&a);
            if(fp_is_zero(&a)) continue;
            if(fp_legendre(&a)!=1) continue;   /* 平方剰余だけ */
            fp_t b, bb; fp_sqrt(&b, &a); fp_mul(&bb, &b, &b);
            tested++;
            if(!fp_is_equal(&bb, &a)) fail++;
        }
        printf("   [fp]   sqrt(a)^2==a     : %s (%d件検査)\n", fail?"NG":"OK", tested);
        ok &= (fail==0);
    }

    /* 3) fp4: A * A^-1 == 1 */
    {
        fp4_t one; fp4_set_ui(&one, 1);
        int fail=0;
        for(int i=0;i<trials;i++){
            fp4_t A; fp4_random(&A);
            fp4_t inv, prod;
            fp4_inv(&inv, &A);
            fp4_mul(&prod, &A, &inv);
            if(!fp4_is_equal(&prod, &one)) fail++;
        }
        printf("   [fp4]  A*A^-1==1        : %s\n", fail?"NG":"OK");
        ok &= (fail==0);
    }

    /* 4) fp16: X * X^-1 == 1  (SubBytes の逆元と同じ経路) */
    {
        fp16_t one; make_fp16_one(&one);
        int fail=0;
        for(int i=0;i<trials;i++){
            fp16_t X; fp16_random(&X);
            if(fp16_is_zero(&X)) continue;
            fp16_t inv, prod;
            fp16_inv(&inv, &X);
            fp16_mul(&prod, &X, &inv);
            if(!fp16_is_equal(&prod, &one)) fail++;
        }
        printf("   [fp16] X*X^-1==1        : %s\n", fail?"NG":"OK");
        ok &= (fail==0);
    }

    printf("   => %s\n\n", ok ? "この素数は体の算術OK" : "NG あり");
    return ok;
}

int main(void){
    const int T = 2000;
    int all = 1;
    printf("=== 体の算術テスト (a*a^-1, sqrt, fp4/fp16 逆元) ===\n\n");
    all &= test_one_prime(64,  T);   /* -> q=7   (2^7-1)  対照 */
    all &= test_one_prime(256, T);   /* -> q=31  (2^31-1) 対照 */
    all &= test_one_prime(512, T);   /* -> q=127 (2^127-1) 本命 */
    printf("=== 総合: %s ===\n", all ? "全素数 PASS" : "FAIL あり(上のNGを参照)");
    return all ? 0 : 1;
}