#include "16_header.h"

uint64_t fp4_mul_count = 0; // fp4 乗算回数カウンタ
uint64_t fp4_mul_karatsuba_count = 0; // Karatsuba 版乗算の呼び出し回数
uint64_t fp4_mul_slow_count = 0; // fp4 素朴乗算（slow）の呼び出し回数

void fp4_init(fp4_t *X){
    fp_init(&X->x0);
    fp_init(&X->x1);
    fp_init(&X->x2);
    fp_init(&X->x3);
}

void fp4_clear(fp4_t *X){
    fp_clear(&X->x0);
    fp_clear(&X->x1);
    fp_clear(&X->x2);
    fp_clear(&X->x3);
}

void fp4_printf(const fp4_t *X){
    printf("%u*a + %u*a^2 + %u*a^4 + %u*a^3\n",
        X->x0.x0,X->x1.x0,X->x2.x0,X->x3.x0);
}

void fp4_set(fp4_t *S, const fp4_t *X){
    fp_set(&S->x0,&X->x0);
    fp_set(&S->x1,&X->x1);
    fp_set(&S->x2,&X->x2);
    fp_set(&S->x3,&X->x3);
}

void fp4_set_ui(fp4_t *S, unsigned long int x){
    // 1 = -(γ + ... + γ^3)
    uint32_t val = (uint32_t)x;
    // mod p
    while (val >= P_MERSENNE) val -= P_MERSENNE;
    
    // -val mod p
    if (val != 0) val = P_MERSENNE - val;

    S->x0.x0 = val; S->x1.x0 = val;
    S->x2.x0 = val; S->x3.x0 = val;
}

// ランダム生成
void fp4_random(fp4_t *X){
    fp_random(&X->x0);
    fp_random(&X->x1);
    fp_random(&X->x2);
    fp_random(&X->x3);
}

int fp4_is_equal(const fp4_t *A, const fp4_t *B) {
    if (fp_cmp(&A->x0, &B->x0) == 0 &&
        fp_cmp(&A->x1, &B->x1) == 0 &&
        fp_cmp(&A->x2, &B->x2) == 0 &&
        fp_cmp(&A->x3, &B->x3) == 0) {
        return 1; // Equal
    }
    return 0; // Not equal
}

void fp4_add(fp4_t *S, const fp4_t *X, const fp4_t *Y){
    fp_add(&S->x0, &X->x0, &Y->x0);
    fp_add(&S->x1, &X->x1, &Y->x1);
    fp_add(&S->x2, &X->x2, &Y->x2);
    fp_add(&S->x3, &X->x3, &Y->x3);
}

void fp4_sub(fp4_t *S, const fp4_t *X, const fp4_t *Y){
    fp_sub(&S->x0, &X->x0, &Y->x0);
    fp_sub(&S->x1, &X->x1, &Y->x1);
    fp_sub(&S->x2, &X->x2, &Y->x2);
    fp_sub(&S->x3, &X->x3, &Y->x3);
}

void fp4_neg(fp4_t *S, const fp4_t *X){
    fp_neg(&S->x0, &X->x0);
    fp_neg(&S->x1, &X->x1);
    fp_neg(&S->x2, &X->x2);
    fp_neg(&S->x3, &X->x3);
}

void fp4_frobenius_map(fp4_t *S, const fp4_t *X){
    fp_t tmp;
    fp_init(&tmp);
    
    // SとXが同じ場合を考慮して退避
    fp_set(&tmp, &X->x3);
    
    fp_set(&S->x3, &X->x2);
    fp_set(&S->x2, &X->x1);
    fp_set(&S->x1, &X->x0);
    fp_set(&S->x0, &tmp);
    
    fp_clear(&tmp);
}

// 乗算 (Type I ONB, CVMA: 10乗算で全係数を生成)
// 基底: γ, γ^2, γ^4, γ^3（Frobeniusで巡回）
void fp4_mul(fp4_t *S, const fp4_t *X, const fp4_t *Y){
    fp4_mul_count++;
    // Algorithm 2 (Type-I CVMA, h=1) を m=4, 基底 {γ,γ^2,γ^4,γ^3} に展開した形。
    // σ 写像は (i,j)→{ (0,1)->3, (0,2)->4, (0,3)->2, (1,2)->0, (1,3)->4, (2,3)->1 }。
    fp_t v[5], dx, dy, t;

    // 対角項 v0..v3
    fp_mul(&v[0], &X->x0, &Y->x0);
    fp_mul(&v[1], &X->x1, &Y->x1);
    fp_mul(&v[2], &X->x2, &Y->x2);
    fp_mul(&v[3], &X->x3, &Y->x3);
    v[4].x0 = 0; // v_m

    // (0,1) -> σ=3
    fp_sub(&dx, &X->x0, &X->x1); fp_sub(&dy, &Y->x0, &Y->x1);
    fp_mul(&t, &dx, &dy); fp_add(&v[3], &v[3], &t);

    // (0,2) -> σ=4
    fp_sub(&dx, &X->x0, &X->x2); fp_sub(&dy, &Y->x0, &Y->x2);
    fp_mul(&t, &dx, &dy); fp_add(&v[4], &v[4], &t);

    // (0,3) -> σ=2
    fp_sub(&dx, &X->x0, &X->x3); fp_sub(&dy, &Y->x0, &Y->x3);
    fp_mul(&t, &dx, &dy); fp_add(&v[2], &v[2], &t);

    // (1,2) -> σ=0
    fp_sub(&dx, &X->x1, &X->x2); fp_sub(&dy, &Y->x1, &Y->x2);
    fp_mul(&t, &dx, &dy); fp_add(&v[0], &v[0], &t);

    // (1,3) -> σ=4
    fp_sub(&dx, &X->x1, &X->x3); fp_sub(&dy, &Y->x1, &Y->x3);
    fp_mul(&t, &dx, &dy); fp_add(&v[4], &v[4], &t);

    // (2,3) -> σ=1
    fp_sub(&dx, &X->x2, &X->x3); fp_sub(&dy, &Y->x2, &Y->x3);
    fp_mul(&t, &dx, &dy); fp_add(&v[1], &v[1], &t);

    // z_l = v_m - v_l
    fp_sub(&S->x0, &v[4], &v[0]);
    fp_sub(&S->x1, &v[4], &v[1]);
    fp_sub(&S->x2, &v[4], &v[2]);
    fp_sub(&S->x3, &v[4], &v[3]);
}

void fp4_mul_slow(fp4_t *S, const fp4_t *X, const fp4_t *Y){
    fp4_mul_slow_count++;
    fp_t v[16];
    for(int i=0;i<16;i++){
        fp_init(&v[i]);
    }
    fp_mul(&v[0],&X->x0,&Y->x0);
    fp_mul(&v[1],&X->x0,&Y->x1);
    fp_mul(&v[2],&X->x0,&Y->x2);
    fp_mul(&v[3],&X->x0,&Y->x3);
    fp_mul(&v[4],&X->x1,&Y->x0);
    fp_mul(&v[5],&X->x1,&Y->x1);
    fp_mul(&v[6],&X->x1,&Y->x2);
    fp_mul(&v[7],&X->x1,&Y->x3);
    fp_mul(&v[8],&X->x2,&Y->x0);
    fp_mul(&v[9],&X->x2,&Y->x1);
    fp_mul(&v[10],&X->x2,&Y->x2);
    fp_mul(&v[11],&X->x2,&Y->x3);
    fp_mul(&v[12],&X->x3,&Y->x0);
    fp_mul(&v[13],&X->x3,&Y->x1);
    fp_mul(&v[14],&X->x3,&Y->x2);
    fp_mul(&v[15],&X->x3,&Y->x3);

    fp_t con;
    fp_init(&con);
    fp_add(&con,&v[2],&v[8]);
    fp_add(&con,&con,&v[7]);
    fp_add(&con,&con,&v[13]);

    //c0
    fp_add(&S->x0,&v[6],&v[10]);
    fp_add(&S->x0,&S->x0,&v[15]);
    fp_sub(&S->x0,&S->x0,&con);
    //c1
    fp_add(&S->x1,&v[0],&v[11]);
    fp_add(&S->x1,&S->x0,&v[14]);
    fp_sub(&S->x1,&S->x0,&con);
    //c2
    fp_add(&S->x2,&v[3],&v[5]);
    fp_add(&S->x2,&S->x0,&v[12]);
    fp_sub(&S->x2,&S->x0,&con);
    //c3
    fp_add(&S->x3,&v[1],&v[4]);
    fp_add(&S->x3,&S->x0,&v[10]);
    fp_sub(&S->x3,&S->x0,&con);
}

void fp4_add_beta_power(fp4_t *R, const fp_t *c, int k){
    k %= 5;
    if(k < 0) k += 5;

    switch(k){
        case 1: fp_add(&R->x0, &R->x0, c); break;
        case 2: fp_add(&R->x1, &R->x1, c); break;
        case 3: fp_add(&R->x3, &R->x3, c); break;
        case 4: fp_add(&R->x2, &R->x2, c); break;
        case 0:
            fp_sub(&R->x0, &R->x0, c);
            fp_sub(&R->x1, &R->x1, c);
            fp_sub(&R->x2, &R->x2, c);
            fp_sub(&R->x3, &R->x3, c);
            break;
    }
}

// (a0 + a1*x) * (b0 + b1*x)
// → r[0] + r[1]*x + r[2]*x^2
void fp4_mul2(fp_t r[3], const fp_t a[2], const fp_t b[2])
{
    // Karatsuba: 3 mul (m0, m2, m1 cross term)
    fp_t m0, m2, s1, s2, m1;

    // m0 = a0*b0, m2 = a1*b1
    fp_mul(&m0, &a[0], &b[0]);
    fp_mul(&m2, &a[1], &b[1]);

    // m1 = (a0+a1)*(b0+b1) - m0 - m2
    fp_add(&s1, &a[0], &a[1]);
    fp_add(&s2, &b[0], &b[1]);
    fp_mul(&m1, &s1, &s2);
    fp_sub(&m1, &m1, &m0);
    fp_sub(&m1, &m1, &m2);

    r[0] = m0;
    r[1] = m1;
    r[2] = m2;
}

void fp4_mul_karatsuba(fp4_t *R, const fp4_t *A, const fp4_t *B)
{
    fp4_mul_karatsuba_count++;
    fp_t X0[2] = {A->x0, A->x1}; // β, β^2
    fp_t X1[2] = {A->x2, A->x3}; // β^4, β^3
    fp_t Y0[2] = {B->x0, B->x1};
    fp_t Y1[2] = {B->x2, B->x3};

    fp_t T0[3], T1[3], T2[3];
    fp_t Sx[2], Sy[2];

    // T0 = X0 * Y0
    fp4_mul2(T0, X0, Y0);

    // T1 = X1 * Y1
    fp4_mul2(T1, X1, Y1);

    // Sx = X0 + X1, Sy = Y0 + Y1
    fp_add(&Sx[0], &X0[0], &X1[0]);
    fp_add(&Sx[1], &X0[1], &X1[1]);
    fp_add(&Sy[0], &Y0[0], &Y1[0]);
    fp_add(&Sy[1], &Y0[1], &Y1[1]);

    // T2 = (X0+X1)(Y0+Y1)
    fp4_mul2(T2, Sx, Sy);

    // M = T2 - T0 - T1
    for (int i = 0; i < 3; i++) {
        fp_sub(&T2[i], &T2[i], &T0[i]);
        fp_sub(&T2[i], &T2[i], &T1[i]);
    }

    // 係数は k mod 5 の加算規則に基づき展開して直接合成する
    // x0 = T2[1] + T1[2] - T2[0]
    fp_add(&R->x0, &T2[1], &T1[2]);
    fp_sub(&R->x0, &R->x0, &T2[0]);

    // x1 = T0[0] + T2[2] + T1[1] - T2[0]
    fp_add(&R->x1, &T0[0], &T2[2]);
    fp_add(&R->x1, &R->x1, &T1[1]);
    fp_sub(&R->x1, &R->x1, &T2[0]);

    // x2 = T0[2] - T2[0]
    fp_sub(&R->x2, &T0[2], &T2[0]);

    // x3 = T0[1] + T1[0] - T2[0]
    fp_add(&R->x3, &T0[1], &T1[0]);
    fp_sub(&R->x3, &R->x3, &T2[0]);
}


// 2乗 (実装簡略化のため mul を呼び出す)
// ※ 正規基底では S = X^(2^k) は高速だが、S = X^2 は乗算と同じコストがかかる
void fp4_sqr(fp4_t *S, const fp4_t *X){
    fp4_mul(S, X, X);
}

// 繰り返し2乗法によるべき乗
void fp4_pow(fp4_t *S, const fp4_t *X, const mpz_t exp){
    fp4_t result, base;
    fp4_set_ui(&result, 1);
    fp4_set(&base, X);

    size_t bit_len = mpz_sizeinbase(exp, 2);
    for (size_t i = 0; i < bit_len; i++) {
        if (mpz_tstbit(exp, i)) {
            fp4_mul(&result, &result, &base);
        }
        fp4_sqr(&base, &base);
    }
    fp4_set(S, &result);
}

// 逆元 S = 1/X
// X^(-1) = (X^p * X^(p^2) * X^(p^3)) / Norm(X)
void fp4_inv(fp4_t *S, const fp4_t *X){
    fp4_t t1, t2, numerator, norm;
    // 変数の初期化は構造体定義によるが、今回は代入で上書きされるので不要
    
    // 1. t1 = X^p
    fp4_frobenius_map(&t1, X);
    
    // 2. t2 = X * X^p  (= X^(1+p))
    fp4_mul(&t2, X, &t1);
    
    // 3. t1 = X^(p^2)
    fp4_frobenius_map(&t1, &t1); // (X^p)^p
    
    // 4. numerator = (X * X^p) * X^(p^2)  (= X^(1+p+p^2))
    // これが逆元の分子の元（X^p * X^p^2 * X^p^3）に近い形
    fp4_mul(&numerator, &t2, &t1);
    
    // 5. t1 = X^(p^3)
    fp4_frobenius_map(&t1, &t1); // (X^p^2)^p
    
    // 6. norm = numerator * X^(p^3)  (= X^(1+p+p^2+p^3))
    // これがノルム。正規基底ではスカラ値となる。
    fp4_mul(&norm, &numerator, &t1);
    
    // 7. 分子の仕上げ: numerator を p乗する
    // 現在 numerator = X^(1+p+p^2) なので、
    // p乗すると X^(p+p^2+p^3) となり、求めたかった「自分以外の積」になる
    fp4_frobenius_map(&numerator, &numerator);

    // 8. ノルムの逆数計算 (Fp上での計算)
    // ノルムはスカラなので、正規基底表現では全係数が同じ値になっているはず。
    // x0 成分だけ取り出して Fp 上で逆元を取ればよい。
    
    if (fp_is_zero(&norm.x0)) {
        // 0の逆元はないので 0 を返す（またはエラー処理）
        fp4_set_ui(S, 0);
        return;
    }
    
    // d = norm.x0 とすると、このベクトルは整数 -d を表している。
    // 欲しいのは (-d)^(-1) を表すベクトル。
    // (-d)^(-1) = -(d^(-1)) なので、係数は d^(-1) になる。
    // つまり、単に係数の逆数を取って、全成分にセットすればよい。
    fp_inv(&norm.x0, &norm.x0); // x0 = x0^(-1)
    
    // 求めた逆数を全成分にセットして、スカラ倍用のベクトルを作る
    fp_set(&norm.x1, &norm.x0);
    fp_set(&norm.x2, &norm.x0);
    fp_set(&norm.x3, &norm.x0);
    
    // 9. 最後に分子とノルムの逆数を掛ける
    fp4_mul(S, &numerator, &norm);
}

void fp4_inv_slow(fp4_t *S, const fp4_t *X){
    fp4_t t1, t2, numerator, norm;

    fp4_frobenius_map(&t1, X);
    
    fp4_mul_slow(&t2, X, &t1);
    
    fp4_frobenius_map(&t1, &t1); // (X^p)^p
    
    fp4_mul_slow(&numerator, &t2, &t1);
    
    fp4_frobenius_map(&t1, &t1); // (X^p^2)^p
    
    fp4_mul_slow(&norm, &numerator, &t1);
    
    fp4_frobenius_map(&numerator, &numerator);

    if (fp_is_zero(&norm.x0)) {
        // 0の逆元はないので 0 を返す（またはエラー処理）
        fp4_set_ui(S, 0);
        return;
    }

    fp_inv(&norm.x0, &norm.x0); // x0 = x0^(-1)
    
    fp_set(&norm.x1, &norm.x0);
    fp_set(&norm.x2, &norm.x0);
    fp_set(&norm.x3, &norm.x0);
    
    fp4_mul_slow(S, &numerator, &norm);
}

void fp4_inv_karatsuba(fp4_t *S, const fp4_t *X){
    fp4_t t1, t2, numerator, norm;

    fp4_frobenius_map(&t1, X);
    
    fp4_mul_karatsuba(&t2, X, &t1);
    
    fp4_frobenius_map(&t1, &t1); // (X^p)^p
    
    fp4_mul_karatsuba(&numerator, &t2, &t1);
    
    fp4_frobenius_map(&t1, &t1); // (X^p^2)^p
    
    fp4_mul_karatsuba(&norm, &numerator, &t1);
    
    fp4_frobenius_map(&numerator, &numerator);

    if (fp_is_zero(&norm.x0)) {
        // 0の逆元はないので 0 を返す（またはエラー処理）
        fp4_set_ui(S, 0);
        return;
    }

    fp_inv(&norm.x0, &norm.x0); // x0 = x0^(-1)
    
    fp_set(&norm.x1, &norm.x0);
    fp_set(&norm.x2, &norm.x0);
    fp_set(&norm.x3, &norm.x0);
    
    fp4_mul_karatsuba(S, &numerator, &norm);
}

// quartic residue 判定: x^{(p^4-1)/4} = 1 なら4乗根が存在
int fp4_has_4th_root(const fp4_t *X){
    // 0 は自明に4乗根 (0) を持つ
    if (fp_is_zero(&X->x0) && fp_is_zero(&X->x1) && fp_is_zero(&X->x2) && fp_is_zero(&X->x3)) {
        return 1;
    }

    mpz_t p, exp;
    mpz_init_set_ui(p, P_MERSENNE);
    mpz_init(exp);

    // exp = (p^4 - 1) / 4
    mpz_pow_ui(exp, p, 4);
    mpz_sub_ui(exp, exp, 1);
    mpz_tdiv_q_ui(exp, exp, 4);

    fp4_t res, one;
    fp4_pow(&res, X, exp);
    fp4_set_ui(&one, 1);

    int is_residue = fp4_is_equal(&res, &one);

    mpz_clear(p);
    mpz_clear(exp);
    return is_residue;
}

// ランダム試行で4乗非剰余を探す。見つけた全てを表示し、既約性もチェック。
void fp4_quartic_residue_scan(int trials){
    fp4_t tmp;
    fp4_init(&tmp);
    int found = 0;
    for(int i=0;i<trials;i++){
        fp4_random(&tmp);
        if(!fp4_has_4th_root(&tmp)){
            printf("Found 4th-nonresidue at trial %d: ", i); fp4_printf(&tmp);
            int irr = fp4_x4_minus_irreducible(&tmp);
            printf("  x^4 - alpha is %s\n", irr ? "irreducible" : "reducible");
            found = 1;
        }
    }
    if(!found){
        printf("No 4th-nonresidue found in %d random trials.\n", trials);
    }
    fp4_clear(&tmp);
}

// 平方剰余判定: x^{(p^4-1)/2} = 1 ?
int fp4_is_square(const fp4_t *X){
    // 0 は平方とみなす
    if (fp_is_zero(&X->x0) && fp_is_zero(&X->x1) && fp_is_zero(&X->x2) && fp_is_zero(&X->x3)) {
        return 1;
    }
    mpz_t p, exp;
    mpz_init_set_ui(p, P_MERSENNE);
    mpz_init(exp);

    // exp = (p^4 - 1) / 2
    mpz_pow_ui(exp, p, 4);
    mpz_sub_ui(exp, exp, 1);
    mpz_tdiv_q_ui(exp, exp, 2);

    fp4_t res, one;
    fp4_pow(&res, X, exp);
    fp4_set_ui(&one, 1);

    int is_sq = fp4_is_equal(&res, &one);

    mpz_clear(p);
    mpz_clear(exp);
    return is_sq;
}

// x^4 - alpha が既約なら1、そうでなければ0
// 判定: alpha が 4乗・平方でなく、かつ -4*alpha が 4乗でないこと
int fp4_x4_minus_irreducible(const fp4_t *alpha){
    // 0 の場合は x^4 なので既約ではない
    if (fp_is_zero(&alpha->x0) && fp_is_zero(&alpha->x1) && fp_is_zero(&alpha->x2) && fp_is_zero(&alpha->x3)) {
        return 0;
    }

    // alpha が 4乗なら 1次因子を持つ
    if (fp4_has_4th_root(alpha)) return 0;
    // alpha が平方なら 2次×2次に分解
    if (fp4_is_square(alpha)) return 0;

    // -4 * alpha が 4乗なら 2次×2次に分解する形が存在
    fp4_t tmp, minus4alpha;
    fp4_set(&tmp, alpha);

    fp_t four, m4;
    four.x0 = 4 % P_MERSENNE;
    fp_neg(&m4, &four); // m4 = -4 mod p

    // minus4alpha = m4 * alpha (スカラ倍)
    fp_mul(&minus4alpha.x0, &alpha->x0, &m4);
    fp_mul(&minus4alpha.x1, &alpha->x1, &m4);
    fp_mul(&minus4alpha.x2, &alpha->x2, &m4);
    fp_mul(&minus4alpha.x3, &alpha->x3, &m4);

    if (fp4_has_4th_root(&minus4alpha)) return 0;

    return 1; // どの条件にも引っかからなければ既約
}

// 乗法位数を計算（X=0 の場合は order=0）
void fp4_order(mpz_t order, const fp4_t *X){
    // 0 の場合は位数なし
    if (fp_is_zero(&X->x0) && fp_is_zero(&X->x1) && fp_is_zero(&X->x2) && fp_is_zero(&X->x3)) {
        mpz_set_ui(order, 0);
        return;
    }

    // group order = p^4 - 1
    mpz_t ord, p, exp;
    mpz_init_set_ui(p, P_MERSENNE);
    mpz_init(ord);
    mpz_pow_ui(ord, p, 4);
    mpz_sub_ui(ord, ord, 1);

    fp4_t one, tmp;
    fp4_set_ui(&one, 1);

    // 素因子（重複は不要）: 2,3,5,7,11,31,151,331,733,1709,368140581013
    const uint64_t primes[] = {2ULL,3ULL,5ULL,7ULL,11ULL,31ULL,151ULL,331ULL,733ULL,1709ULL,368140581013ULL};
    size_t plen = sizeof(primes)/sizeof(primes[0]);

    mpz_init(exp);
    for(size_t i=0;i<plen;i++){
        uint64_t q = primes[i];
        while(mpz_divisible_ui_p(ord, q)){
            mpz_divexact_ui(exp, ord, q);
            fp4_pow(&tmp, X, exp);
            if (fp4_is_equal(&tmp, &one)){
                mpz_set(ord, exp); // 位数を更新
            }else{
                break;
            }
        }
    }

    mpz_set(order, ord);
    mpz_clear(p);
    mpz_clear(ord);
    mpz_clear(exp);
}

int fp4_is_scalar(const fp4_t *X){
    return fp_is_equal(&X->x0, &X->x1) &&
           fp_is_equal(&X->x0, &X->x2) &&
           fp_is_equal(&X->x0, &X->x3);
}

int fp4_is_zero_vec(const fp4_t *X){
    return fp_is_zero(&X->x0) && fp_is_zero(&X->x1) &&
           fp_is_zero(&X->x2) && fp_is_zero(&X->x3);
}
