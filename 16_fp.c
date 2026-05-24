#include "16_header.h"

uint64_t fp_mul_count = 0; // 乗算回数カウンタ
uint64_t fp_add_count = 0; // 加算回数カウンタ
uint64_t fp_sub_count = 0; // 減算回数カウンタ

// 64bitの計算結果を 31bit (mod p) に落とし込む関数
static inline uint32_t reduce_mersenne(uint64_t x) {
    // 上位31bitと下位31bitを足す
    // (x mod 2^31) + (x / 2^31) と等価
    uint32_t result = (uint32_t)(x & P_MERSENNE) + (uint32_t)(x >> 31);
    
    // 足した結果、もう一度 P を超える可能性があるので調整
    if (result >= P_MERSENNE) {
        result -= P_MERSENNE;
    }
    return result;
}

static inline uint32_t reduce_mod(uint64_t x) {
    volatile uint32_t p = P_PRIME;
    uint64_t mod = x % p;
    uint32_t result = (uint32_t)mod;
    return result;
}

// 初期化
void fp_init(fp_t *X){
    X->x0 = 0;
}

// 解放
void fp_clear(fp_t *X){

}

// 表示
void fp_printf(const fp_t *X){
    gmp_printf("%u",X->x0);
}

// 代入
void fp_set(fp_t *S, const fp_t *X){
    S->x0 = X->x0;
}

void fp_set_zero(fp_t *S){
    S->x0 = 0;
}

void fp_set_ui(fp_t *S, uint32_t x){
    S->x0 = x;
}

int fp_is_equal(const fp_t *X, const fp_t *Y){
    return(X->x0 == Y->x0);
}


int fp_is_zero(const fp_t *X){
    return (X->x0 == 0);
}

void fp_random(fp_t *X) {
    // rand() のシードを1回だけ初期化
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }

    uint32_t r = (uint32_t)rand();
    r ^= ((uint32_t)rand() << 16); // 上位ビットも埋める
    
    X->x0 = r & P_MERSENNE;
    if (X->x0 == P_MERSENNE) X->x0 = 0; // Pと等しい場合は0にする
}

// 比較
int fp_cmp(const fp_t *X, const fp_t *Y){
    if (X->x0 == Y->x0) 
        return 0;
    return 1; // GMPの挙動に合わせて「異なれば1」を返すならこれ
}

// 負数
void fp_neg(fp_t *S, const fp_t *X){
    if (X->x0 == 0) {
            S->x0 = 0;
        } else {
            S->x0 = P_MERSENNE - X->x0;
        }
}

// 和 S = X + Y
void fp_add(fp_t *S, const fp_t *X, const fp_t *Y){
    fp_add_count++;
    uint32_t sum = X->x0 + Y->x0;
        if (sum >= P_MERSENNE) {
            sum -= P_MERSENNE;
        }
        S->x0 = sum;
}

void fp_add_plus(fp_t *S, const fp_t *X, const fp_t *Y){
    fp_add_count++;
    uint32_t sum = X->x0 + Y->x0;
        if(sum >= P_PRIME){
            sum -= P_PRIME;
        }
        S->x0 = sum;
}

// 差 S = X - Y
void fp_sub(fp_t *S, const fp_t *X, const fp_t *Y){
    fp_sub_count++;
    uint32_t x = X->x0;
    uint32_t y = Y->x0;
    
    if (x < y) {
        // 負になる場合は P を足してから引く (mod計算のテクニック)
        S->x0 = x + P_MERSENNE - y;
    } else {
        S->x0 = x - y;
    }
}

void fp_sub_plus(fp_t *S, const fp_t *X, const fp_t *Y){
    fp_sub_count++;
    uint32_t x = X->x0;
    uint32_t y = Y->x0;
    
    if (x < y) {
        // 負になる場合は P を足してから引く (mod計算のテクニック)
        S->x0 = x + P_PRIME - y;
    } else {
        S->x0 = x - y;
    }
}

// 積 S = X * Y
void fp_mul(fp_t *S, const fp_t *X, const fp_t *Y){
    fp_mul_count++;
    // 64bitで掛け算してから、31bitに落とす
    uint64_t prod = (uint64_t)X->x0 * (uint64_t)Y->x0;
    S->x0 = reduce_mersenne(prod);
}

// 積 S = X * Y
void fp_mul_plus(fp_t *S, const fp_t *X, const fp_t *Y){
    fp_mul_count++;
    // 64bitで掛け算してから、31bitに落とす
    uint64_t prod = (uint64_t)X->x0 * (uint64_t)Y->x0;
    S->x0 = reduce_mod(prod);
}

// 逆数 S = 1 / X
void fp_inv(fp_t *S, const fp_t *X){
    if (X->x0 == 0) {
        printf("/0 is not defined\n");
        return;
    }
    
    // 指数 p - 2 を作る
    mpz_t exp;
    mpz_init_set_str(exp, "7FFFFFFD", 16); // 2^31 - 3 (つまり P - 2)
    
    fp_pow(S, X, exp);
    
    mpz_clear(exp);
}

// 冪乗 S = X ** s
void fp_pow(fp_t *S, const fp_t *X, const mpz_t s){
    fp_t base, res;
    fp_set(&base, X);
    res.x0 = 1;

    // mpz_t のビット数を取得してループ
    size_t bit_len = mpz_sizeinbase(s, 2);
    
    for (size_t i = 0; i < bit_len; i++) {
        // s の i ビット目が 1 なら掛ける
        if (mpz_tstbit(s, i)) {
            fp_mul(&res, &res, &base);
        }
        // base を2乗
        fp_mul(&base, &base, &base);
    }
    fp_set(S, &res);
}

// 平方剰余判定
// a^((p-1)/2) mod p == 1 なら平方剰余
int fp_legendre(const fp_t *X){
    if (X->x0 == 0) return 0;

    fp_t res;
    mpz_t exp;
    // (p-1)/2 = (2^31 - 2) / 2 = 2^30 - 1 = 0x3FFFFFFF
    mpz_init_set_str(exp, "3FFFFFFF", 16);

    fp_pow(&res, X, exp);
    mpz_clear(exp);

    if (res.x0 == 1) return 1;
    else return -1;
}

// 1.2.7 平方根計算 b = √a
int fp_sqrt(fp_t *b, fp_t *a){
// 平方剰余でなければ失敗
    if (fp_legendre(a) != 1) {
        if (fp_is_zero(a)) {
            b->x0 = 0;
            return 1;
        }
        return 0; 
    }

    // 指数 (p+1)/4 を計算
    // p = 2^31 - 1
    // p+1 = 2^31
    // (p+1)/4 = 2^31 / 2^2 = 2^29
    mpz_t exp;
    mpz_init(exp);
    mpz_set_ui(exp, 1);
    mpz_mul_2exp(exp, exp, 29); // 2^29

    // べき乗計算だけで平方根が求まる！ (Tonelli-Shanks不要)
    fp_pow(b, a, exp);

    mpz_clear(exp);
    return 1;
}
