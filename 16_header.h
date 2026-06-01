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
#include <stddef.h>

#define P_MERSENNE 2147483647 //p = 2^31-1
#define P_PRIME 2147483629u //p = 2^31-19
#define MATRIX_STATE_BYTES 64
#define MATRIX_DIGEST_BYTES 32
#define MATRIX_ROUNDS 10
#define MATRIX_BLOCK_BYTES 64 //1ブロックあたりのbyte数

// from fp.c
typedef struct{
    uint32_t x0;
}fp_t;

void fp_init(fp_t *X);
void fp_clear(fp_t *X);
void fp_printf(const fp_t *X);
void fp_set(fp_t *S, const fp_t *X);
void fp_set_zero(fp_t *S);
void fp_set_ui(fp_t *S, uint32_t x);
int fp_is_equal(const fp_t *X, const fp_t *Y);
int fp_is_zero(const fp_t *X);
void fp_random(fp_t *X);
int fp_cmp(const fp_t *X, const fp_t *Y);
void fp_neg(fp_t *S, const fp_t *X);
void fp_add(fp_t *S, const fp_t *X, const fp_t *Y);
void fp_sub(fp_t *S, const fp_t *X, const fp_t *Y);
void fp_mul(fp_t *S, const fp_t *X, const fp_t *Y);
void fp_add_plus(fp_t *S, const fp_t *X, const fp_t *Y);
void fp_sub_plus(fp_t *S, const fp_t *X, const fp_t *Y);
void fp_mul_plus(fp_t *S, const fp_t *X, const fp_t *Y);
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
void fp4_set_zero(fp4_t *S);
void fp4_set_ui(fp4_t *S, unsigned long int x); // 整数セット用
void fp4_random(fp4_t *X);
int fp4_is_equal(const fp4_t *A, const fp4_t *B); //一致していれば1、不一致なら0を返す
int fp4_is_zero(const fp4_t *X);
void fp4_add(fp4_t *S, const fp4_t *X, const fp4_t *Y);
void fp4_sub(fp4_t *S, const fp4_t *X, const fp4_t *Y);
void fp4_mul(fp4_t *S, const fp4_t *X, const fp4_t *Y);
void fp4_mul_new(fp4_t *S, const fp4_t *X, const fp4_t *Y);
void fp4_mul_nre(fp4_t *S, const fp4_t *X);
void fp4_mul_slow(fp4_t *S, const fp4_t *X, const fp4_t *Y);
void fp4_add_beta_power(fp4_t *R, const fp_t *c, int k);
void fp4_mul2(fp_t r[3], const fp_t a[2], const fp_t b[2]);
void fp4_mul_karatsuba(fp4_t *R, const fp4_t *A, const fp4_t *B);
void fp4_sqr(fp4_t *S, const fp4_t *X);
void fp4_frobenius_map(fp4_t *S, const fp4_t *X); // p乗写像 (高速)
void fp4_pow(fp4_t *S, const fp4_t *X, const mpz_t exp);
void fp4_inv(fp4_t *S, const fp4_t *X); // p^4-2 乗による単純な逆元
void fp4_inv_slow(fp4_t *S, const fp4_t *X);
void fp4_inv_karatsuba(fp4_t *S, const fp4_t *X);
void fp4_inv_new(fp4_t *S, const fp4_t *X);
void fp4_neg(fp4_t *S, const fp4_t *X);
int fp4_has_4th_root(const fp4_t *X); // x^{(p^4-1)/4} == 1 をチェック
void fp4_quartic_residue_scan(int trials); // ランダム試行で4乗非剰余を探す
int fp4_is_square(const fp4_t *X); // x^{(p^4-1)/2} == 1 をチェック
int fp4_x4_minus_irreducible(const fp4_t *alpha); // x^4 - alpha が既約なら1
void fp4_order(mpz_t order, const fp4_t *X); // 乗法位数（0 の場合は0を返す）
int fp4_is_scalar(const fp4_t *X);

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
void fp16_set_zero(fp16_t *S);
void fp16_random(fp16_t *X);
int fp16_is_equal(const fp16_t *A, const fp16_t *B);
int fp16_is_zero(const fp16_t *X);
void fp16_add(fp16_t *S, const fp16_t *X, const fp16_t *Y);
void fp16_sub(fp16_t *S, const fp16_t *X, const fp16_t *Y);
void fp16_mul(fp16_t *S, const fp16_t *X, const fp16_t *Y);
void fp16_mul_new(fp16_t *S, const fp16_t *X, const fp16_t *Y);
void fp16_mul_slow(fp16_t *S, const fp16_t *X, const fp16_t *Y);
void fp16_mul_karatsuba(fp16_t *S, const fp16_t *X, const fp16_t *Y);
void fp16_sqr(fp16_t *S, const fp16_t *X);
void fp16_mul_sparse(fp16_t *S, const fp16_t *X, const fp4_t *Y); // fp4倍
void fp16_mul_sparse_slow(fp16_t *S, const fp16_t *X, const fp4_t *Y);
void fp16_mul_sparse_karatsuba(fp16_t *S, const fp16_t *X, const fp4_t *Y);
void fp16_pow(fp16_t *S, const fp16_t *X, const mpz_t exp);
void fp16_inv(fp16_t *S, const fp16_t *X);
void fp16_inv_new(fp16_t *S, const fp16_t *X);
void fp16_inv_slow(fp16_t *S, const fp16_t *X);
void fp16_inv_karatsuba(fp16_t *S, const fp16_t *X);
int fp16_is_scalar(const fp16_t *X);

typedef struct{
    fp_t m[4][4];
} state_t;

typedef struct{
    fp_t A[16][16];
    fp_t b[16];
} affine16_t;

void state_init(state_t *S);
void state_set_zero(state_t *S);
void state_random(state_t *S);
int state_is_zero(const state_t *S);
void state_from_fp16(state_t *S, const fp16_t *X);
void state_to_fp16(fp16_t *S, const state_t *X);
void state_clear(state_t *S);
void state_copy(state_t *dst, const state_t *src);
void state_add(state_t *Z, const state_t *X, const state_t *Y);
void state_sub(state_t *Z, const state_t *X, const state_t *Y);
void state_add3(state_t *Z, const state_t *A, const state_t *B, const state_t *C);
int  state_equal(const state_t *A, const state_t *B);
void state_print(const state_t *S);

//mixbytes
void state_mix_column(fp_t y[4], const fp_t x[4], const state_t *M);
void matrix_mixbytes(state_t *S_new, const state_t *S, const state_t *M);

//shiftbytes
void matrix_shiftbytes(state_t *S_new, const state_t *S, const int shift[4]);
void matrix_shiftbytes_P(state_t *S_new, const state_t *S);
void matrix_shiftbytes_Q(state_t *S_new, const state_t *S);

//subbytes
void affine16_init(affine16_t *AFF);
void affine16_clear(affine16_t *AFF);
void state_to_vec16(fp_t v[16], const state_t *S);
void vec16_to_state(state_t *S, const fp_t v[16]);
void affine16_set(affine16_t *AFF);
void affine16_apply_vec(fp_t y[16], const fp_t x[16], const affine16_t *AFF);
void matrix_affine(state_t *S_new, const state_t *S, const affine16_t *AFF);
void matrix_subbytes(state_t *S_new, const state_t *S, const affine16_t *AFF);

//add round constant
void matrix_add_round_constant_P(state_t *S_new, const state_t *S, int r);
void matrix_add_round_constant_Q(state_t *S_new, const state_t *S, int r);

//1ラウンドの処理をまとめる関数
void matrix_round_P(state_t *S_new, const state_t *S, int r, const state_t *MDS, const affine16_t *AFF);
void matrix_round_Q(state_t *S_new, const state_t *S, int r, const state_t *MDS, const affine16_t *AFF);

//ラウンド処理をrounds回行う関数
void matrix_permutation_P(state_t *S_new, const state_t *S, int rounds, const state_t *MDS, const affine16_t *AFF);
void matrix_permutation_Q(state_t *S_new, const state_t *S, int rounds, const state_t *MDS, const affine16_t *AFF);

//圧縮関数
void matrix_compression(state_t *out, const state_t *h, const state_t *m, int rounds, const state_t *MDS, const affine16_t *AFF);

//P(h) + h
void matrix_output_transform(state_t *out, const state_t *h, int rounds, const state_t *MDS, const affine16_t *AFF);

//行列からビット列に変換
static void fp4_to_bytes_128(uint8_t *out, const fp4_t *x);
void fp16_to_bytes_512(uint8_t out[MATRIX_STATE_BYTES], const fp16_t *x);
void matrix_state_to_bytes_512(uint8_t out[MATRIX_STATE_BYTES], const state_t *S);

//最終的な出力関数
int matrix_trunc_tail_bytes(uint8_t *digest, size_t digest_len, const uint8_t full_state[MATRIX_STATE_BYTES]);
int matrix_output_transform_digest(uint8_t *digest, size_t digest_len, const state_t *h, int rounds, const state_t *MDS, const affine16_t *AFF);

int matrix_hash_one_block(uint8_t *digest, size_t digest_len, const uint8_t block[MATRIX_STATE_BYTES], int rounds, const state_t *MDS, const affine16_t *AFF);

//padding関数
size_t matrix_padded_length(size_t msg_len);
int matrix_check_msg_len(size_t msg_len);
void matrix_pad(uint8_t *out, size_t padded_len, const uint8_t *msg, size_t msg_len);

//API関数
int matrix_hash(uint8_t *digest, size_t digest_len, const uint8_t *msg, size_t msg_len, int rounds, const state_t *MDS, const affine16_t *AFF);

//テスト関数
void test_matrix_hash_one_block(void);
void print_bytes_hex(const uint8_t *buf, size_t len);
void test_state_bytes_roundtrip(void);
// void matrix_add_round_constant_p(state_t *S, uint8_t round);
// void matrix_add_round_constant_q(state_t *S, uint8_t round);
// void matrix_shiftbytes(state_t *S);
// void matrix_subbytes_inv(state_t *S);
// void matrix_subbytes_inv_new(state_t *S);
// void matrix_subbytes_inv_karatsuba(state_t *S);
// void matrix_round_p(state_t *S, uint8_t round, int inv_mode);
// void matrix_round_q(state_t *S, uint8_t round, int inv_mode);

//get time
long bench_fp_add_avg(int iters, int count);
long bench_fp_sub_avg(int iters, int count);
long bench_fp_mul_avg(int iters, int count);
long bench_fp_add_plus_avg(int iters, int count);
long bench_fp_sub_plus_avg(int iters, int count);
long bench_fp_mul_plus_avg(int iters, int count);
long bench_fp4_mul_avg(int iters, int count);
long bench_fp4_mul_new_avg(int iters, int count);
long bench_fp4_mul_karatsuba_avg(int iters, int count);
long bench_fp4_mul_slow_avg(int iters, int count);
long bench_fp16_inv_avg(int iters, int count);
long bench_fp16_inv_new_avg(int iters, int count);
long bench_fp16_inv_karatsuba_avg(int iters, int count);
long bench_fp16_inv_slow_avg(int iters, int count);
// long bench_matrix_round_p_inv_avg(int iters, int count);
// long bench_matrix_round_p_inv_new_avg(int iters, int count);
// long bench_matrix_round_p_inv_karatsuba_avg(int iters, int count);
extern uint64_t fp_mul_count;
extern uint64_t fp_add_count;
extern uint64_t fp_sub_count;
extern uint64_t fp4_mul_count;
extern uint64_t fp4_mul_karatsuba_count;
extern uint64_t fp4_mul_slow_count;

#endif
