#include "16_header.h"

const fp4_t alpha = {
    .x0 = { 1441937721 },  // γ
    .x1 = { 46373297 },  // γ^2
    .x2 = { 2124195301 },  // γ^4
    .x3 = { 471336014 }   // γ^3
};

void fp16_init(fp16_t *X){
    fp4_init(&X->x0);
    fp4_init(&X->x1);
    fp4_init(&X->x2);
    fp4_init(&X->x3);
}

void fp16_clear(fp16_t *X){

}

void fp16_printf(const fp16_t *X){
    printf("[\n");
    printf("  x0: "); fp4_printf(&X->x0);
    printf("  x1: "); fp4_printf(&X->x1);
    printf("  x2: "); fp4_printf(&X->x2);
    printf("  x3: "); fp4_printf(&X->x3);
    printf("]\n");
}

void fp16_set(fp16_t *S, const fp16_t *X){
    fp4_set(&S->x0, &X->x0);
    fp4_set(&S->x1, &X->x1);
    fp4_set(&S->x2, &X->x2);
    fp4_set(&S->x3, &X->x3);
}

void fp16_random(fp16_t *X){
    fp4_random(&X->x0);
    fp4_random(&X->x1);
    fp4_random(&X->x2);
    fp4_random(&X->x3);
}

int fp16_is_equal(const fp16_t *A, const fp16_t *B){
    return (fp4_is_equal(&A->x0, &B->x0) &&
            fp4_is_equal(&A->x1, &B->x1) &&
            fp4_is_equal(&A->x2, &B->x2) &&
            fp4_is_equal(&A->x3, &B->x3));
}

void fp16_add(fp16_t *S, const fp16_t *X, const fp16_t *Y){
    fp4_add(&S->x0, &X->x0, &Y->x0);
    fp4_add(&S->x1, &X->x1, &Y->x1);
    fp4_add(&S->x2, &X->x2, &Y->x2);
    fp4_add(&S->x3, &X->x3, &Y->x3);
}

// 減算
void fp16_sub(fp16_t *S, const fp16_t *X, const fp16_t *Y){
    fp4_sub(&S->x0, &X->x0, &Y->x0);
    fp4_sub(&S->x1, &X->x1, &Y->x1);
    fp4_sub(&S->x2, &X->x2, &Y->x2);
    fp4_sub(&S->x3, &X->x3, &Y->x3);
}

// スカラー倍 (Fp4倍) fp16_tの各要素(fp4_t)に同じfp4_tをかけていく
void fp16_mul_sparse(fp16_t *S, const fp16_t *X, const fp4_t *Y){
    fp4_mul(&S->x0, &X->x0, Y);
    fp4_mul(&S->x1, &X->x1, Y);
    fp4_mul(&S->x2, &X->x2, Y);
    fp4_mul(&S->x3, &X->x3, Y);
}

// 乗算
// S = A * B mod (y^4 - ALPHA)
void fp16_mul(fp16_t *S, const fp16_t *X, const fp16_t *Y){
    fp4_t t0, t1, t2, t3, t4, t5, t6, tmp;
    // 愚直な展開 (Karatsubaを使えばもっと速くなるが、まずは確実性重視)
    
    // t0 (定数項): x0*y0
    fp4_mul(&t0, &X->x0, &Y->x0);
    
    // t4 (4次の項): x1*y3 + x2*y2 + x3*y1
    // これに ALPHA を掛けて t0 に足す (y^4 = ALPHA)
    fp4_mul(&tmp, &X->x1, &Y->x3); fp4_mul(&t4, &X->x2, &Y->x2); fp4_add(&t4, &t4, &tmp);
    fp4_mul(&tmp, &X->x3, &Y->x1); fp4_add(&t4, &t4, &tmp);
    fp4_mul(&t4, &t4, &alpha);
    fp4_add(&t0, &t0, &t4);

    // t1 (1次の項): x0*y1 + x1*y0
    fp4_mul(&t1, &X->x0, &Y->x1);
    fp4_mul(&tmp, &X->x1, &Y->x0); fp4_add(&t1, &t1, &tmp);

    // t5 (5次の項): x2*y3 + x3*y2
    // これに ALPHA を掛けて t1 に足す (y^5 = ALPHA * y)
    fp4_mul(&t5, &X->x2, &Y->x3);
    fp4_mul(&tmp, &X->x3, &Y->x2); fp4_add(&t5, &t5, &tmp);
    fp4_mul(&t5, &t5, &alpha);
    fp4_add(&t1, &t1, &t5);

    // t2 (2次の項): x0*y2 + x1*y1 + x2*y0
    fp4_mul(&t2, &X->x0, &Y->x2);
    fp4_mul(&tmp, &X->x1, &Y->x1); fp4_add(&t2, &t2, &tmp);
    fp4_mul(&tmp, &X->x2, &Y->x0); fp4_add(&t2, &t2, &tmp);

    // t6 (6次の項): x3*y3
    // これに ALPHA を掛けて t2 に足す (y^6 = ALPHA * y^2)
    fp4_mul(&t6, &X->x3, &Y->x3);
    fp4_mul(&t6, &t6, &alpha);
    fp4_add(&t2, &t2, &t6);

    // t3 (3次の項): x0*y3 + x1*y2 + x2*y1 + x3*y0
    // 7次以上の項はないので、これはそのまま
    fp4_mul(&t3, &X->x0, &Y->x3);
    fp4_mul(&tmp, &X->x1, &Y->x2); fp4_add(&t3, &t3, &tmp);
    fp4_mul(&tmp, &X->x2, &Y->x1); fp4_add(&t3, &t3, &tmp);
    fp4_mul(&tmp, &X->x3, &Y->x0); fp4_add(&t3, &t3, &tmp);

    // 結果格納
    fp16_set(S, (fp16_t*)&t0); // 構造体のレイアウトが同じならキャストも可だが、
    // ここでは安全に変数からセット
    fp4_set(&S->x0, &t0);
    fp4_set(&S->x1, &t1);
    fp4_set(&S->x2, &t2);
    fp4_set(&S->x3, &t3);
}

// 2乗
void fp16_sqr(fp16_t *S, const fp16_t *X){
    fp16_mul(S, X, X);
}

// 逆元 S = 1/X
// 2段階の共役を利用して計算
void fp16_inv(fp16_t *S, const fp16_t *X){
    fp16_t X_conj, M, M_conj, T;
    fp4_t norm_inv;

    // 1. 第1共役 X' = (x0, -x1, x2, -x3)
    // y の奇数次の符号を反転
    fp4_set(&X_conj.x0, &X->x0);
    fp4_set(&X_conj.x2, &X->x2);
    // x1, x3 は符号反転
    fp4_t tmp_neg;
    // -x1
    fp4_neg(&tmp_neg, &X->x1); // ※もしfp4_negがあればそれを使う
    fp4_set(&X_conj.x1, &tmp_neg);
    
    // -x3
    fp4_neg(&tmp_neg, &X->x3);
    fp4_set(&X_conj.x3, &tmp_neg);

    // 2. 部分ノルム M = X * X'
    // M は x0, x2 成分しか持たないはず (y^2 の多項式になる)
    fp16_mul(&M, X, &X_conj);

    // 3. 第2共役 M' = (m0, m1, -m2, m3) 
    // M は実質 c0 + c1*y^2 なので、y^2 の項 (x2) を反転
    fp16_set(&M_conj, &M);
    
    // -M.x2
    fp4_neg(&tmp_neg, &M.x2);
    fp4_set(&M_conj.x2, &tmp_neg);

    // 4. 全体ノルム N = M * M'
    // 結果はスカラ (x0成分のみ) になる
    fp16_mul(&T, &M, &M_conj);

    //printf("Norm = "); fp16_printf(&T);

    // 5. ノルムの逆数 norm_inv = 1 / T.x0
    fp4_inv(&norm_inv, &T.x0);

    // 6. 結果 S = X' * M' * norm_inv
    fp16_mul(&T, &X_conj, &M_conj);
    fp16_mul_sparse(S, &T, &norm_inv); // スカラ倍
}

// 繰り返し2乗法によるべき乗
void fp16_pow(fp16_t *S, const fp16_t *X, const mpz_t exp){
    fp16_t result, base;
    // result = 1
    fp4_set_ui(&result.x0, 1);
    fp4_set_ui(&result.x1, 0);
    fp4_set_ui(&result.x2, 0);
    fp4_set_ui(&result.x3, 0);

    fp16_set(&base, X);

    size_t bit_len = mpz_sizeinbase(exp, 2);
    for (size_t i = 0; i < bit_len; i++) {
        if (mpz_tstbit(exp, i)) {
            fp16_mul(&result, &result, &base);
        }
        fp16_sqr(&base, &base);
    }
    fp16_set(S, &result);
}

int fp16_is_scalar(const fp16_t *X){
    return fp4_is_scalar(&X->x0) &&
           fp4_is_zero_vec(&X->x1) &&
           fp4_is_zero_vec(&X->x2) &&
           fp4_is_zero_vec(&X->x3);
}