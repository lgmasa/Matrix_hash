#include "16_header.h"

// const fp4_t alpha = {
//     .x0 = { 1441937721 },  // γ
//     .x1 = { 46373297 },  // γ^2
//     .x2 = { 2124195301 },  // γ^4
//     .x3 = { 471336014 }   // γ^3
// };

const fp4_t alpha = {
    .x0 = { 1 },  // γ
    .x1 = { 0 },  // γ^2
    .x2 = { 0 },  // γ^4
    .x3 = { 6 }   // γ^3
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

void fp16_set_zero(fp16_t *S){
    fp4_set_zero(&S->x0);
    fp4_set_zero(&S->x1);
    fp4_set_zero(&S->x2);
    fp4_set_zero(&S->x3);
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

int fp16_is_zero(const fp16_t *X){
    return fp4_is_zero(&X->x0) &&
           fp4_is_zero(&X->x1) &&
           fp4_is_zero(&X->x2) &&
           fp4_is_zero(&X->x3);
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

void fp16_mul_sparse_slow(fp16_t *S, const fp16_t *X, const fp4_t *Y){
    fp4_mul_slow(&S->x0, &X->x0, Y);
    fp4_mul_slow(&S->x1, &X->x1, Y);
    fp4_mul_slow(&S->x2, &X->x2, Y);
    fp4_mul_slow(&S->x3, &X->x3, Y);
}

void fp16_mul_sparse_karatsuba(fp16_t *S, const fp16_t *X, const fp4_t *Y){
    fp4_mul_karatsuba(&S->x0, &X->x0, Y);
    fp4_mul_karatsuba(&S->x1, &X->x1, Y);
    fp4_mul_karatsuba(&S->x2, &X->x2, Y);
    fp4_mul_karatsuba(&S->x3, &X->x3, Y);
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

void fp16_mul_new(fp16_t *S, const fp16_t *X, const fp16_t *Y){
    fp4_t t0, t1, t2, t3, t4, t5, t6, tmp;
    // 愚直な展開 (Karatsubaを使えばもっと速くなるが、まずは確実性重視)
    
    // t0 (定数項): x0*y0
    fp4_mul_new(&t0, &X->x0, &Y->x0);
    
    // t4 (4次の項): x1*y3 + x2*y2 + x3*y1
    // これに ALPHA を掛けて t0 に足す (y^4 = ALPHA)
    fp4_mul_new(&tmp, &X->x1, &Y->x3); fp4_mul_new(&t4, &X->x2, &Y->x2); fp4_add(&t4, &t4, &tmp);
    fp4_mul_new(&tmp, &X->x3, &Y->x1); fp4_add(&t4, &t4, &tmp);
    fp4_mul_nre(&t4, &t4);
    fp4_add(&t0, &t0, &t4);

    // t1 (1次の項): x0*y1 + x1*y0
    fp4_mul_new(&t1, &X->x0, &Y->x1);
    fp4_mul_new(&tmp, &X->x1, &Y->x0); fp4_add(&t1, &t1, &tmp);

    // t5 (5次の項): x2*y3 + x3*y2
    // これに ALPHA を掛けて t1 に足す (y^5 = ALPHA * y)
    fp4_mul_new(&t5, &X->x2, &Y->x3);
    fp4_mul_new(&tmp, &X->x3, &Y->x2); fp4_add(&t5, &t5, &tmp);
    fp4_mul_nre(&t5, &t5);
    fp4_add(&t1, &t1, &t5);

    // t2 (2次の項): x0*y2 + x1*y1 + x2*y0
    fp4_mul_new(&t2, &X->x0, &Y->x2);
    fp4_mul_new(&tmp, &X->x1, &Y->x1); fp4_add(&t2, &t2, &tmp);
    fp4_mul_new(&tmp, &X->x2, &Y->x0); fp4_add(&t2, &t2, &tmp);

    // t6 (6次の項): x3*y3
    // これに ALPHA を掛けて t2 に足す (y^6 = ALPHA * y^2)
    fp4_mul_new(&t6, &X->x3, &Y->x3);
    fp4_mul_nre(&t6, &t6);
    fp4_add(&t2, &t2, &t6);

    // t3 (3次の項): x0*y3 + x1*y2 + x2*y1 + x3*y0
    // 7次以上の項はないので、これはそのまま
    fp4_mul_new(&t3, &X->x0, &Y->x3);
    fp4_mul_new(&tmp, &X->x1, &Y->x2); fp4_add(&t3, &t3, &tmp);
    fp4_mul_new(&tmp, &X->x2, &Y->x1); fp4_add(&t3, &t3, &tmp);
    fp4_mul_new(&tmp, &X->x3, &Y->x0); fp4_add(&t3, &t3, &tmp);

    // 結果格納
    fp16_set(S, (fp16_t*)&t0); // 構造体のレイアウトが同じならキャストも可だが、
    // ここでは安全に変数からセット
    fp4_set(&S->x0, &t0);
    fp4_set(&S->x1, &t1);
    fp4_set(&S->x2, &t2);
    fp4_set(&S->x3, &t3);
}

void fp16_mul_slow(fp16_t *S, const fp16_t *X, const fp16_t *Y){
    fp4_t t0, t1, t2, t3, t4, t5, t6, tmp;
    // 愚直な展開 (Karatsubaを使えばもっと速くなるが、まずは確実性重視)
    
    // t0 (定数項): x0*y0
    fp4_mul_slow(&t0, &X->x0, &Y->x0);
    
    // t4 (4次の項): x1*y3 + x2*y2 + x3*y1
    // これに ALPHA を掛けて t0 に足す (y^4 = ALPHA)
    fp4_mul_slow(&tmp, &X->x1, &Y->x3); fp4_mul_slow(&t4, &X->x2, &Y->x2); fp4_add(&t4, &t4, &tmp);
    fp4_mul_slow(&tmp, &X->x3, &Y->x1); fp4_add(&t4, &t4, &tmp);
    fp4_mul_slow(&t4, &t4, &alpha);
    fp4_add(&t0, &t0, &t4);

    // t1 (1次の項): x0*y1 + x1*y0
    fp4_mul_slow(&t1, &X->x0, &Y->x1);
    fp4_mul_slow(&tmp, &X->x1, &Y->x0); fp4_add(&t1, &t1, &tmp);

    // t5 (5次の項): x2*y3 + x3*y2
    // これに ALPHA を掛けて t1 に足す (y^5 = ALPHA * y)
    fp4_mul_slow(&t5, &X->x2, &Y->x3);
    fp4_mul_slow(&tmp, &X->x3, &Y->x2); fp4_add(&t5, &t5, &tmp);
    fp4_mul_slow(&t5, &t5, &alpha);
    fp4_add(&t1, &t1, &t5);

    // t2 (2次の項): x0*y2 + x1*y1 + x2*y0
    fp4_mul_slow(&t2, &X->x0, &Y->x2);
    fp4_mul_slow(&tmp, &X->x1, &Y->x1); fp4_add(&t2, &t2, &tmp);
    fp4_mul_slow(&tmp, &X->x2, &Y->x0); fp4_add(&t2, &t2, &tmp);

    // t6 (6次の項): x3*y3
    // これに ALPHA を掛けて t2 に足す (y^6 = ALPHA * y^2)
    fp4_mul_slow(&t6, &X->x3, &Y->x3);
    fp4_mul_slow(&t6, &t6, &alpha);
    fp4_add(&t2, &t2, &t6);

    // t3 (3次の項): x0*y3 + x1*y2 + x2*y1 + x3*y0
    // 7次以上の項はないので、これはそのまま
    fp4_mul_slow(&t3, &X->x0, &Y->x3);
    fp4_mul_slow(&tmp, &X->x1, &Y->x2); fp4_add(&t3, &t3, &tmp);
    fp4_mul_slow(&tmp, &X->x2, &Y->x1); fp4_add(&t3, &t3, &tmp);
    fp4_mul_slow(&tmp, &X->x3, &Y->x0); fp4_add(&t3, &t3, &tmp);

    // 結果格納
    fp16_set(S, (fp16_t*)&t0); // 構造体のレイアウトが同じならキャストも可だが、
    // ここでは安全に変数からセット
    fp4_set(&S->x0, &t0);
    fp4_set(&S->x1, &t1);
    fp4_set(&S->x2, &t2);
    fp4_set(&S->x3, &t3);
}

void fp16_mul_karatsuba(fp16_t *S, const fp16_t *X, const fp16_t *Y){
    fp4_t t0, t1, t2, t3, t4, t5, t6, tmp;
    // 愚直な展開 (Karatsubaを使えばもっと速くなるが、まずは確実性重視)
    
    // t0 (定数項): x0*y0
    fp4_mul_karatsuba(&t0, &X->x0, &Y->x0);
    
    // t4 (4次の項): x1*y3 + x2*y2 + x3*y1
    // これに ALPHA を掛けて t0 に足す (y^4 = ALPHA)
    fp4_mul_karatsuba(&tmp, &X->x1, &Y->x3); fp4_mul_karatsuba(&t4, &X->x2, &Y->x2); fp4_add(&t4, &t4, &tmp);
    fp4_mul_karatsuba(&tmp, &X->x3, &Y->x1); fp4_add(&t4, &t4, &tmp);
    fp4_mul_karatsuba(&t4, &t4, &alpha);
    fp4_add(&t0, &t0, &t4);

    // t1 (1次の項): x0*y1 + x1*y0
    fp4_mul_karatsuba(&t1, &X->x0, &Y->x1);
    fp4_mul_karatsuba(&tmp, &X->x1, &Y->x0); fp4_add(&t1, &t1, &tmp);

    // t5 (5次の項): x2*y3 + x3*y2
    // これに ALPHA を掛けて t1 に足す (y^5 = ALPHA * y)
    fp4_mul_karatsuba(&t5, &X->x2, &Y->x3);
    fp4_mul_karatsuba(&tmp, &X->x3, &Y->x2); fp4_add(&t5, &t5, &tmp);
    fp4_mul_karatsuba(&t5, &t5, &alpha);
    fp4_add(&t1, &t1, &t5);

    // t2 (2次の項): x0*y2 + x1*y1 + x2*y0
    fp4_mul_karatsuba(&t2, &X->x0, &Y->x2);
    fp4_mul_karatsuba(&tmp, &X->x1, &Y->x1); fp4_add(&t2, &t2, &tmp);
    fp4_mul_karatsuba(&tmp, &X->x2, &Y->x0); fp4_add(&t2, &t2, &tmp);

    // t6 (6次の項): x3*y3
    // これに ALPHA を掛けて t2 に足す (y^6 = ALPHA * y^2)
    fp4_mul_karatsuba(&t6, &X->x3, &Y->x3);
    fp4_mul_karatsuba(&t6, &t6, &alpha);
    fp4_add(&t2, &t2, &t6);

    // t3 (3次の項): x0*y3 + x1*y2 + x2*y1 + x3*y0
    // 7次以上の項はないので、これはそのまま
    fp4_mul_karatsuba(&t3, &X->x0, &Y->x3);
    fp4_mul_karatsuba(&tmp, &X->x1, &Y->x2); fp4_add(&t3, &t3, &tmp);
    fp4_mul_karatsuba(&tmp, &X->x2, &Y->x1); fp4_add(&t3, &t3, &tmp);
    fp4_mul_karatsuba(&tmp, &X->x3, &Y->x0); fp4_add(&t3, &t3, &tmp);

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

void fp16_inv_new(fp16_t *S, const fp16_t *X){
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
    fp16_mul_new(&M, X, &X_conj);

    // 3. 第2共役 M' = (m0, m1, -m2, m3) 
    // M は実質 c0 + c1*y^2 なので、y^2 の項 (x2) を反転
    fp16_set(&M_conj, &M);
    
    // -M.x2
    fp4_neg(&tmp_neg, &M.x2);
    fp4_set(&M_conj.x2, &tmp_neg);

    // 4. 全体ノルム N = M * M'
    // 結果はスカラ (x0成分のみ) になる
    fp16_mul_new(&T, &M, &M_conj);

    //printf("Norm = "); fp16_printf(&T);

    // 5. ノルムの逆数 norm_inv = 1 / T.x0
    fp4_inv_new(&norm_inv, &T.x0);

    // 6. 結果 S = X' * M' * norm_inv
    fp16_mul_new(&T, &X_conj, &M_conj);
    fp16_mul_sparse(S, &T, &norm_inv); // スカラ倍
}

void fp16_inv_slow(fp16_t *S, const fp16_t *X){
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
    fp16_mul_slow(&M, X, &X_conj);

    // 3. 第2共役 M' = (m0, m1, -m2, m3) 
    // M は実質 c0 + c1*y^2 なので、y^2 の項 (x2) を反転
    fp16_set(&M_conj, &M);
    
    // -M.x2
    fp4_neg(&tmp_neg, &M.x2);
    fp4_set(&M_conj.x2, &tmp_neg);

    // 4. 全体ノルム N = M * M'
    // 結果はスカラ (x0成分のみ) になる
    fp16_mul_slow(&T, &M, &M_conj);

    //printf("Norm = "); fp16_printf(&T);

    // 5. ノルムの逆数 norm_inv = 1 / T.x0
    fp4_inv_slow(&norm_inv, &T.x0);

    // 6. 結果 S = X' * M' * norm_inv
    fp16_mul_slow(&T, &X_conj, &M_conj);
    fp16_mul_sparse_slow(S, &T, &norm_inv); // スカラ倍
}

void fp16_inv_karatsuba(fp16_t *S, const fp16_t *X){
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
    fp16_mul_karatsuba(&M, X, &X_conj);

    // 3. 第2共役 M' = (m0, m1, -m2, m3) 
    // M は実質 c0 + c1*y^2 なので、y^2 の項 (x2) を反転
    fp16_set(&M_conj, &M);
    
    // -M.x2
    fp4_neg(&tmp_neg, &M.x2);
    fp4_set(&M_conj.x2, &tmp_neg);

    // 4. 全体ノルム N = M * M'
    // 結果はスカラ (x0成分のみ) になる
    fp16_mul_karatsuba(&T, &M, &M_conj);

    //printf("Norm = "); fp16_printf(&T);

    // 5. ノルムの逆数 norm_inv = 1 / T.x0
    fp4_inv_karatsuba(&norm_inv, &T.x0);

    // 6. 結果 S = X' * M' * norm_inv
    fp16_mul_karatsuba(&T, &X_conj, &M_conj);
    fp16_mul_sparse_karatsuba(S, &T, &norm_inv); // スカラ倍
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
           fp4_is_zero(&X->x1) &&
           fp4_is_zero(&X->x2) &&
           fp4_is_zero(&X->x3);
}


/* ===================================================================
 * naive 法(CVMA を使わない乗算)での逆元計算と、素体乗算の回数の測定
 * =================================================================== */

/* S = X^exp。fp16_pow と同じ右から左へのバイナリ法だが、乗算に fp16_mul_slow を使う */
void fp16_pow_slow(fp16_t *S, const fp16_t *X, const mpz_t exp){
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
            fp16_mul_slow(&result, &result, &base);
        }
        fp16_mul_slow(&base, &base, &base);   // 2乗
    }
    fp16_set(S, &result);
}

/* ランダムな元 X に対して、naive 法で X^(p^16-2) を計算し、
 *   1) 結果が逆元になっているか(X * X^(p^16-2) == 1、および fp16_inv の結果と一致するか)
 *   2) 計算に使った素体乗算(fp_mul)の回数
 * を表示する。
 * n_bits は素数を選ぶための出力ビット長(64 -> 2^7-1, 256 -> 2^31-1, 1016 -> 2^127-1) */
void fp16_inv_naive_check(size_t n_bits){
    if (!field_select_for_output(n_bits)) {
        printf("素数の選択に失敗しました (n_bits = %zu)\n", n_bits);
        return;
    }

    fp16_t X, Y, Z, one;

    // ランダムな非ゼロの元 X
    do { fp16_random(&X); } while (fp16_is_zero(&X));

    // 指数 e = p^16 - 2
    mpz_t p, e;
    mpz_init(p);
    mpz_init(e);
    mpz_ui_pow_ui(p, 2, FP_BITS);
    mpz_sub_ui(p, p, 1);
    mpz_pow_ui(e, p, 16);
    mpz_sub_ui(e, e, 2);
    size_t L = mpz_sizeinbase(e, 2);   // 指数のビット長
    size_t w = mpz_popcount(e);        // 指数の 1 のビット数

    // fp16_mul_slow 1回あたりの fp_mul 回数
    uint64_t c0 = fp_mul_count;
    fp16_mul_slow(&Z, &X, &X);
    uint64_t per_mul = fp_mul_count - c0;

    // 逆元計算(ここだけを数える)
    c0 = fp_mul_count;
    uint64_t s0 = fp4_mul_slow_count;
    fp16_pow_slow(&Y, &X, e);
    uint64_t n_fp_mul  = fp_mul_count       - c0;
    uint64_t n_fp4_mul = fp4_mul_slow_count - s0;

    // 確認 1: X * Y == 1(確認には CVMA 版の fp16_mul を使う = 別の実装でのクロスチェック)
    fp4_set_ui(&one.x0, 1);
    fp4_set_ui(&one.x1, 0);
    fp4_set_ui(&one.x2, 0);
    fp4_set_ui(&one.x3, 0);
    fp16_mul(&Z, &X, &Y);
    int ok_one = fp16_is_equal(&Z, &one);

    // 確認 2: 既存の fp16_inv と同じ結果か
    fp16_inv(&Z, &X);
    int ok_inv = fp16_is_equal(&Z, &Y);

    uint64_t theory = (uint64_t)(L + w) * per_mul;

    printf("== naive 法での逆元 x^(p^16-2), p = 2^%u - 1 ==\n", FP_BITS);
    printf("  指数 p^16-2                 : ビット長 L = %zu, 1のビット数 w = %zu\n", L, w);
    printf("  fp16_mul_slow 1回あたり     : fp_mul %llu 回\n", (unsigned long long)per_mul);
    printf("  fp16_mul_slow の回数        : L + w = %zu 回\n", L + w);
    printf("  素体乗算 fp_mul の回数      : %llu 回 (fp4_mul_slow %llu 回)\n",
           (unsigned long long)n_fp_mul, (unsigned long long)n_fp4_mul);
    printf("  理論値 (L + w) x %llu        : %llu 回 -> %s\n",
           (unsigned long long)per_mul, (unsigned long long)theory,
           n_fp_mul == theory ? "一致" : "不一致");
    printf("  X * X^(p^16-2) == 1         : %s\n", ok_one ? "OK" : "NG");
    printf("  fp16_inv の結果と一致       : %s\n\n", ok_inv ? "OK" : "NG");

    mpz_clear(p);
    mpz_clear(e);
}

// ---- xgcd 用: F_{p^4}[y] の多項式 (次数 4 以下, c[i] が y^i の係数) ----
typedef struct{
    fp4_t c[5];
    int deg; // 零多項式は -1
} fp4_poly_t;

static void fp4_poly_set_zero(fp4_poly_t *P){
    for (int i = 0; i < 5; i++) fp4_set_zero(&P->c[i]);
    P->deg = -1;
}

// 最高次の係数が 0 なら次数を下げる
static void fp4_poly_normalize(fp4_poly_t *P){
    while (P->deg >= 0 && fp4_is_zero(&P->c[P->deg])) P->deg--;
}

// 逆元 S = 1/X (拡張ユークリッド互除法)
// F_{p^16} = F_{p^4}[y]/(y^4 - α) なので、X(y) と M(y) = y^4 - α に互除法を適用し
//   T(y) * X(y) + U(y) * M(y) = g   (g は F_{p^4} の非零定数)
// を満たす T を求めて S = T / g とする。
// 各段で t_i * X ≡ r_i (mod M) が保たれるので、U は追跡しない。
// 係数の逆元は愚直法 fp4_inv_naive (X^(p^4-2) を fp4_pow で計算)、
// 係数の積は naive 法の fp4_mul_slow を使う。
void fp16_inv_xgcd(fp16_t *S, const fp16_t *X){
    if (fp16_is_zero(X)) {
        // 0の逆元はないので 0 を返す
        fp16_set_zero(S);
        return;
    }

    fp4_poly_t r0, r1, t0, t1, swap;
    fp4_t lc_inv, q, tmp;

    // r0 = M = y^4 - α, t0 = 0
    fp4_poly_set_zero(&r0);
    fp4_neg(&r0.c[0], &alpha);
    fp4_set_ui(&r0.c[4], 1);
    r0.deg = 4;
    fp4_poly_set_zero(&t0);

    // r1 = X, t1 = 1
    fp4_poly_set_zero(&r1);
    fp4_set(&r1.c[0], &X->x0);
    fp4_set(&r1.c[1], &X->x1);
    fp4_set(&r1.c[2], &X->x2);
    fp4_set(&r1.c[3], &X->x3);
    r1.deg = 3;
    fp4_poly_normalize(&r1);
    fp4_poly_set_zero(&t1);
    fp4_set_ui(&t1.c[0], 1);
    t1.deg = 0;

    // r1 が定数になるまで繰り返す (M は既約なので gcd は非零定数)
    while (r1.deg > 0) {
        // r0 = q * r1 + (余り) の割り算を 1 項ずつ行い、同時に t0 = t0 - q * t1 を計算
        fp4_inv_xgcd(&lc_inv, &r1.c[r1.deg]);
        while (r0.deg >= r1.deg) {
            int k = r0.deg - r1.deg;
            // 商の y^k の係数 q = (r0 の最高次係数) / (r1 の最高次係数)
            fp4_mul_slow(&q, &r0.c[r0.deg], &lc_inv);
            // r0 = r0 - q * y^k * r1
            for (int i = 0; i <= r1.deg; i++) {
                fp4_mul_slow(&tmp, &q, &r1.c[i]);
                fp4_sub(&r0.c[i + k], &r0.c[i + k], &tmp);
            }
            // t0 = t0 - q * y^k * t1
            for (int i = 0; i <= t1.deg; i++) {
                fp4_mul_slow(&tmp, &q, &t1.c[i]);
                fp4_sub(&t0.c[i + k], &t0.c[i + k], &tmp);
            }
            if (t1.deg + k > t0.deg) t0.deg = t1.deg + k;
            fp4_poly_normalize(&r0);
            fp4_poly_normalize(&t0);
        }
        // (r0, r1) <- (r1, 余り), (t0, t1) <- (t1, t0 - q * t1)
        swap = r0; r0 = r1; r1 = swap;
        swap = t0; t0 = t1; t1 = swap;
    }

    // t1 * X ≡ g (定数 r1.c[0]) なので S = t1 / g
    // t1 の次数は 3 以下 (deg t_i = 4 - deg r_{i-1})
    fp4_inv_xgcd(&lc_inv, &r1.c[0]);
    fp4_mul_slow(&S->x0, &t1.c[0], &lc_inv);
    fp4_mul_slow(&S->x1, &t1.c[1], &lc_inv);
    fp4_mul_slow(&S->x2, &t1.c[2], &lc_inv);
    fp4_mul_slow(&S->x3, &t1.c[3], &lc_inv);
}

/* xgcd 法の逆元 fp16_inv_xgcd (係数の逆元は愚直法 fp4_inv_naive) を検証し、素体乗算の回数を数える。
 *   1) fp4_inv_naive, fp16_inv_xgcd で X * X^(-1) == 1 となり、既存の逆元 (fp4_inv, fp16_inv) と一致するか
 *   2) 次数の低い元 (互除法が途中で終わる場合) や 0 でも正しいか
 *   3) fp16_inv_xgcd 1 回あたりの fp_mul, fp4_mul_slow, fp4_inv_naive (内部の fp4_mul) の回数 (平均・最小・最大)
 * を表示する。
 * n_bits は素数を選ぶための出力ビット長(64 -> 2^7-1, 256 -> 2^31-1, 1016 -> 2^127-1) */
void fp16_inv_xgcd_check(size_t n_bits, int trials){
    if (!field_select_for_output(n_bits)) {
        printf("素数の選択に失敗しました (n_bits = %zu)\n", n_bits);
        return;
    }

    // ---- F_{p^4}: 愚直法 X^(p^4-2) ----
    fp4_t one4;
    fp4_set_ui(&one4, 1);
    int ok_fp4 = 1;
    uint64_t mul_fp4inv = 0, slow_fp4inv = 0;
    for (int t = 0; t < trials; t++) {
        fp4_t a, b, c, prod;
        do { fp4_random(&a); } while (fp4_is_zero(&a));
        uint64_t m0 = fp_mul_count, s0 = fp4_mul_slow_count;
        fp4_inv_naive(&b, &a);
        mul_fp4inv  += fp_mul_count - m0;
        slow_fp4inv += fp4_mul_slow_count - s0;
        fp4_inv(&c, &a);
        fp4_mul_slow(&prod, &a, &b);
        if (!fp4_is_equal(&b, &c) || !fp4_is_equal(&prod, &one4)) ok_fp4 = 0;
    }
    fp4_t z4, iz4;
    fp4_set_zero(&z4);
    fp4_inv_naive(&iz4, &z4);
    int ok_fp4_zero = fp4_is_zero(&iz4);

    // ---- F_{p^16} ----
    fp16_t X, Y, Z, one;
    fp16_set_zero(&one);
    fp4_set_ui(&one.x0, 1);

    int ok_one = 1, ok_inv = 1;
    uint64_t sum_mul = 0, sum_fp4mul_slow = 0, sum_fp4inv = 0, sum_fp4mul = 0;
    uint64_t min_mul = UINT64_MAX, max_mul = 0;
    uint64_t mul_ita = 0;

    for (int t = 0; t < trials; t++) {
        do { fp16_random(&X); } while (fp16_is_zero(&X));

        uint64_t m0 = fp_mul_count, s0 = fp4_mul_slow_count;
        uint64_t i0 = fp4_inv_naive_count, f0 = fp4_mul_count;
        fp16_inv_xgcd(&Y, &X);
        uint64_t m = fp_mul_count - m0;
        sum_mul         += m;
        sum_fp4mul_slow += fp4_mul_slow_count - s0;
        sum_fp4inv      += fp4_inv_naive_count - i0;
        sum_fp4mul      += fp4_mul_count - f0;
        if (m < min_mul) min_mul = m;
        if (m > max_mul) max_mul = m;

        m0 = fp_mul_count;
        fp16_inv(&Z, &X);
        mul_ita += fp_mul_count - m0;

        if (!fp16_is_equal(&Y, &Z)) ok_inv = 0;
        fp16_mul(&Z, &X, &Y);
        if (!fp16_is_equal(&Z, &one)) ok_one = 0;
    }

    // 次数の低い元: 互除法が途中で終わる分岐の確認
    int ok_low = 1;
    for (int t = 0; t < trials; t++) {
        fp16_set_zero(&X);
        int shape = t % 3;
        do { fp4_random(&X.x0); } while (fp4_is_zero(&X.x0));
        if (shape == 1) fp4_random(&X.x1);        // x0 + x1*y
        if (shape == 2) fp4_random(&X.x2);        // x0 + x2*y^2
        fp16_inv_xgcd(&Y, &X);
        fp16_inv(&Z, &X);
        if (!fp16_is_equal(&Y, &Z)) ok_low = 0;
        fp16_mul(&Z, &X, &Y);
        if (!fp16_is_equal(&Z, &one)) ok_low = 0;
    }

    // 0 の場合は 0 を返す
    fp16_set_zero(&X);
    fp16_inv_xgcd(&Y, &X);
    int ok_zero = fp16_is_zero(&Y);

    printf("== xgcd 法での逆元, p = 2^%u - 1 (%d 回) ==\n", FP_BITS, trials);
    printf("  [F_p^4]  fp4_inv_naive が fp4_inv と一致, X * X^(-1) == 1 : %s\n", ok_fp4 ? "OK" : "NG");
    printf("  [F_p^4]  fp4_inv_naive は 0 に対して 0 を返す            : %s\n", ok_fp4_zero ? "OK" : "NG");
    printf("  [F_p^16] fp16_inv と一致                   : %s\n", ok_inv ? "OK" : "NG");
    printf("  [F_p^16] X * X^(-1) == 1                   : %s\n", ok_one ? "OK" : "NG");
    printf("  [F_p^16] 次数の低い元でも正しい            : %s\n", ok_low ? "OK" : "NG");
    printf("  [F_p^16] 0 に対して 0 を返す               : %s\n", ok_zero ? "OK" : "NG");
    printf("  fp16_inv_xgcd 1 回あたり (ランダムな元, 平均)\n");
    printf("    素体乗算 fp_mul         : %.2f 回 (最小 %llu, 最大 %llu)\n",
           (double)sum_mul / trials, (unsigned long long)min_mul, (unsigned long long)max_mul);
    printf("    fp4_mul_slow (naive)    : %.2f 回 (互除法本体 + fp4_inv_naive 内部)\n", (double)sum_fp4mul_slow / trials);
    printf("    fp4_inv_naive           : %.2f 回 (1 回あたり fp4_mul_slow %.2f 回, fp_mul %.2f 回)\n",
           (double)sum_fp4inv / trials, (double)slow_fp4inv / trials, (double)mul_fp4inv / trials);
    printf("    fp4_mul (CVMA)          : %.2f 回\n", (double)sum_fp4mul / trials);
    printf("    (参考) fp16_inv の fp_mul : %.2f 回\n\n", (double)mul_ita / trials);
}

// ---- 演算回数の計測用 ----
typedef struct{
    double mul, add, sub; // 素体の乗算・加算・減算
    double fpinv, div;    // fp_inv_xgcd の呼び出し回数, その中の整数除算の回数
    double inv;           // fp_inv (p-2 乗) の呼び出し回数 (中の乗算は mul に含まれる)
} opcount_t;

static void opcount_snap(opcount_t *c){
    c->mul = fp_mul_count;  c->add = fp_add_count;  c->sub = fp_sub_count;
    c->fpinv = fp_inv_xgcd_count;  c->div = fp_xgcd_div_count;
    c->inv = fp_inv_count;
}

// acc += (after - before)
static void opcount_acc(opcount_t *acc, const opcount_t *before, const opcount_t *after){
    acc->mul   += after->mul   - before->mul;
    acc->add   += after->add   - before->add;
    acc->sub   += after->sub   - before->sub;
    acc->fpinv += after->fpinv - before->fpinv;
    acc->div   += after->div   - before->div;
    acc->inv   += after->inv   - before->inv;
}

static void opcount_print(const char *name, const opcount_t *c, int n){
    printf("  %-30s mul %8.2f  add %8.2f  sub %8.2f  (add+sub %8.2f, 合計 %8.2f)",
           name, c->mul / n, c->add / n, c->sub / n,
           (c->add + c->sub) / n, (c->mul + c->add + c->sub) / n);
    if (c->inv > 0)   printf("  fp_inv %.2f 回", c->inv / n);
    if (c->fpinv > 0) printf("  fp_inv_xgcd %.2f 回 (整数除算 %.2f 回)", c->fpinv / n, c->div / n);
    printf("\n");
}

/* 逆元計算 1 回あたりの素体演算 (fp_mul, fp_add, fp_sub) の回数を、ランダムな元 trials 個の平均で表示する。
 *   - F_{p^16}: fp16_inv_xgcd (xgcd) と fp16_inv (ITA)
 *   - 部品: F_{p^4}, F_p の各逆元と fp4 の乗算
 *   - fp16_inv_xgcd の互除法本体 (全体から、中で呼ばれた F_{p^4} の逆元の分を引いたもの)
 * fp_inv (p-2 乗) の中の乗算は mul に含め、呼び出し回数も別に表示する。
 * fp_neg と、fp_inv_xgcd 内の整数演算 (除算・乗算・加減算) はカウントしない。
 * n_bits は素数を選ぶための出力ビット長(64 -> 2^7-1, 256 -> 2^31-1, 1016 -> 2^127-1) */
void fp16_inv_opcount_check(size_t n_bits, int trials){
    if (!field_select_for_output(n_bits)) {
        printf("素数の選択に失敗しました (n_bits = %zu)\n", n_bits);
        return;
    }

    opcount_t b, a;
    opcount_t c_fp16_xgcd = {0}, c_fp16_ita = {0};
    opcount_t c_fp4_ita = {0}, c_fp4_xgcd = {0}, c_fp4_naive = {0};
    opcount_t c_fp4_mul_slow = {0}, c_fp4_mul = {0};
    opcount_t c_fp_pow = {0}, c_fp_xgcd = {0};
    // fp16_inv_xgcd の中で呼ばれた F_{p^4} の逆元の回数 (種類別)
    uint64_t n_ita = 0, n_xgcd = 0, n_naive = 0;

    for (int t = 0; t < trials; t++) {
        // F_{p^16}
        fp16_t X, Y;
        do { fp16_random(&X); } while (fp16_is_zero(&X));
        uint64_t i0 = fp4_inv_count, x0 = fp4_inv_xgcd_count, v0 = fp4_inv_naive_count;
        opcount_snap(&b); fp16_inv_xgcd(&Y, &X); opcount_snap(&a); opcount_acc(&c_fp16_xgcd, &b, &a);
        n_ita   += fp4_inv_count       - i0;
        n_xgcd  += fp4_inv_xgcd_count  - x0;
        n_naive += fp4_inv_naive_count - v0;
        opcount_snap(&b); fp16_inv(&Y, &X);      opcount_snap(&a); opcount_acc(&c_fp16_ita, &b, &a);

        // F_{p^4}
        fp4_t x4, y4, z4;
        do { fp4_random(&x4); } while (fp4_is_zero(&x4));
        fp4_random(&z4);
        opcount_snap(&b); fp4_inv(&y4, &x4);          opcount_snap(&a); opcount_acc(&c_fp4_ita, &b, &a);
        opcount_snap(&b); fp4_inv_xgcd(&y4, &x4);     opcount_snap(&a); opcount_acc(&c_fp4_xgcd, &b, &a);
        opcount_snap(&b); fp4_inv_naive(&y4, &x4);    opcount_snap(&a); opcount_acc(&c_fp4_naive, &b, &a);
        opcount_snap(&b); fp4_mul_slow(&y4, &x4, &z4); opcount_snap(&a); opcount_acc(&c_fp4_mul_slow, &b, &a);
        opcount_snap(&b); fp4_mul(&y4, &x4, &z4);     opcount_snap(&a); opcount_acc(&c_fp4_mul, &b, &a);

        // F_p
        fp_t x1, y1;
        do { fp_random(&x1); } while (fp_is_zero(&x1));
        opcount_snap(&b); fp_inv(&y1, &x1);      opcount_snap(&a); opcount_acc(&c_fp_pow, &b, &a);
        opcount_snap(&b); fp_inv_xgcd(&y1, &x1); opcount_snap(&a); opcount_acc(&c_fp_xgcd, &b, &a);
    }

    // 互除法本体 = fp16_inv_xgcd 全体 - (中で呼ばれた F_{p^4} の逆元の回数 x それぞれの 1 回あたりの演算回数)
    // (1 回あたりの演算回数は上で単独に測った平均を使う)
    double r_ita = (double)n_ita / trials, r_xgcd = (double)n_xgcd / trials, r_naive = (double)n_naive / trials;
    opcount_t body = {
        c_fp16_xgcd.mul - (r_ita * c_fp4_ita.mul + r_xgcd * c_fp4_xgcd.mul + r_naive * c_fp4_naive.mul),
        c_fp16_xgcd.add - (r_ita * c_fp4_ita.add + r_xgcd * c_fp4_xgcd.add + r_naive * c_fp4_naive.add),
        c_fp16_xgcd.sub - (r_ita * c_fp4_ita.sub + r_xgcd * c_fp4_xgcd.sub + r_naive * c_fp4_naive.sub),
        0, 0, 0
    };

    printf("== 逆元計算 1 回あたりの素体演算回数, p = 2^%u - 1 (%d 回平均) ==\n", FP_BITS, trials);
    printf(" [F_p^16]\n");
    opcount_print("fp16_inv_xgcd (xgcd)", &c_fp16_xgcd, trials);
    opcount_print("fp16_inv (ITA)", &c_fp16_ita, trials);
    opcount_print("  うち xgcd の互除法本体", &body, trials);
    printf("    (fp16_inv_xgcd 内の F_p^4 逆元: fp4_inv %.2f 回, fp4_inv_xgcd %.2f 回, fp4_inv_naive %.2f 回)\n",
           r_ita, r_xgcd, r_naive);
    printf(" [F_p^4]\n");
    opcount_print("fp4_inv (ITA)", &c_fp4_ita, trials);
    opcount_print("fp4_inv_xgcd", &c_fp4_xgcd, trials);
    opcount_print("fp4_inv_naive (X^(p^4-2))", &c_fp4_naive, trials);
    opcount_print("fp4_mul_slow (naive)", &c_fp4_mul_slow, trials);
    opcount_print("fp4_mul (CVMA)", &c_fp4_mul, trials);
    printf(" [F_p]\n");
    opcount_print("fp_inv (X^(p-2))", &c_fp_pow, trials);
    opcount_print("fp_inv_xgcd", &c_fp_xgcd, trials);
    printf("  ※ fp_inv (p-2 乗) の中の乗算は mul に含む。fp_neg と fp_inv_xgcd 内の整数演算はカウントしない\n\n");
}