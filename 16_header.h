#ifndef MILLER_HEADER_H
#define MILLER_HEADER_H
#include <stdio.h>
#include <gmp.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>

#define P_MERSENNE 2147483647 //p = 2^31-1

// from fp.c
typedef struct{
    uint32_t x0;
}fp_t;

void fp_init(fp_t *X);
void fp_clear(fp_t *X);
void fp_printf(const fp_t *X);
void fp_set(fp_t *S, const fp_t *X);
int fp_is_equal(const fp_t *X, const fp_t *Y);
int fp_is_zero(const fp_t *X);
void fp_random(fp_t *X);
int fp_cmp(const fp_t *X, const fp_t *Y);
void fp_neg(fp_t *S, const fp_t *X);
void fp_add(fp_t *S, const fp_t *X, const fp_t *Y);
void fp_sub(fp_t *S, const fp_t *X, const fp_t *Y);
void fp_mul(fp_t *S, const fp_t *X, const fp_t *Y);
void fp_inv(fp_t *S, const fp_t *X);
void fp_pow(fp_t *S, const fp_t *X, const mpz_t s);
int fp_legendre(const fp_t *a);
int fp_sqrt(fp_t *b, fp_t *a);

typedef struct{
    fp_t x0; // coeff of a
    fp_t x1; // coeff of a^2
    fp_t x2; // coeff of a^4
    fp_t x3; // coeff of a^3
} fp4_t;

void fp4_init(fp4_t *X);
void fp4_clear(fp4_t *X);
void fp4_printf(const fp4_t *X);
void fp4_set(fp4_t *S, const fp4_t *X);
void fp4_set_ui(fp4_t *S, unsigned long int x); // 整数セット用
void fp4_random(fp4_t *X);
int fp4_is_equal(const fp4_t *A, const fp4_t *B); //一致していれば1、不一致なら0を返す

void fp4_add(fp4_t *S, const fp4_t *X, const fp4_t *Y);
void fp4_sub(fp4_t *S, const fp4_t *X, const fp4_t *Y);
void fp4_mul(fp4_t *S, const fp4_t *X, const fp4_t *Y);
void fp4_sqr(fp4_t *S, const fp4_t *X);
void fp4_frobenius_map(fp4_t *S, const fp4_t *X); // p乗写像 (高速)
void fp4_inv(fp4_t *S, const fp4_t *X);
void fp4_pow(fp4_t *S, const fp4_t *X, const mpz_t exp);
void fp4_neg(fp4_t *S, const fp4_t *X);
int fp4_has_4th_root(const fp4_t *X); // x^{(p^4-1)/4} == 1 をチェック
void fp4_quartic_residue_scan(int trials); // ランダム試行で4乗非剰余を探す
int fp4_is_square(const fp4_t *X); // x^{(p^4-1)/2} == 1 をチェック
int fp4_x4_minus_irreducible(const fp4_t *alpha); // x^4 - alpha が既約なら1
void fp4_order(mpz_t order, const fp4_t *X); // 乗法位数（0 の場合は0を返す）
int fp4_is_zero_vec(const fp4_t *X);
int fp4_is_scalar(const fp4_t *X);
void fp4_mul_slow(fp4_t *S, const fp4_t *X, const fp4_t *Y); // CVMAなしの素朴乗算
void fp4_mul_slow2(fp4_t *S, const fp4_t *X, const fp4_t *Y);

//既約多項式に用いる元α
extern const fp4_t alpha;

typedef struct{
    fp4_t x0; // coeff of 1
    fp4_t x1; // coeff of β
    fp4_t x2; // coeff of β^2
    fp4_t x3; // coeff of β^3
} fp16_t;

void fp16_init(fp16_t *X);
void fp16_clear(fp16_t *X);
void fp16_printf(const fp16_t *X);
void fp16_set(fp16_t *S, const fp16_t *X);
void fp16_random(fp16_t *X);
int fp16_is_equal(const fp16_t *A, const fp16_t *B);
void fp16_add(fp16_t *S, const fp16_t *X, const fp16_t *Y);
void fp16_sub(fp16_t *S, const fp16_t *X, const fp16_t *Y);
void fp16_mul(fp16_t *S, const fp16_t *X, const fp16_t *Y);
void fp16_mul_slow(fp16_t *S, const fp16_t *X, const fp16_t *Y);
void fp16_sqr(fp16_t *S, const fp16_t *X);
void fp16_mul_sparse(fp16_t *S, const fp16_t *X, const fp4_t *Y); // fp4倍
void fp16_pow(fp16_t *S, const fp16_t *X, const mpz_t exp);
void fp16_inv(fp16_t *S, const fp16_t *X);
void fp16_inv_slow(fp16_t *S, const fp16_t *X);
int fp16_is_scalar(const fp16_t *X);

//get time
long bench_fp4_mul(int iters);
long bench_fp4_mul_slow(int iters);
long bench_fp16_inv(int iters);
long bench_fp16_inv_slow(int iters);
extern uint64_t fp_mul_count;

#endif
