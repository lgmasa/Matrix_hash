#include "16_header.h"

u128 P_MERSENNE = (1u << 31) - 1;   // 既定は 2^31-1
unsigned FP_BITS    = 31;
size_t state_bytes = 62; //16*31=496bit=62byte
size_t block_bytes = 48; //16要素 * load_bits(24bit) / 8
unsigned load_bits  = 24; //1要素あたりの読み込みbit数(q未満に抑える)
static const u128 P127 = (((u128)1) << 127) - 1;

// 127bit境界の fold 還元 (s < 2^128 を [0,p) に)
static inline u128 reduce127(u128 s){
    s = (s & P127) + (s >> 127);
    if (s >= P127) s -= P127;
    return s;
}

// r = a*b mod (2^127-1).  a,b < p
u128 mulmod127(u128 a, u128 b){
    uint64_t a0=(uint64_t)(a&MASK64), a1=(uint64_t)(a>>64);
    uint64_t b0=(uint64_t)(b&MASK64), b1=(uint64_t)(b>>64);
    u128 t00=(u128)a0*b0, t01=(u128)a0*b1, t10=(u128)a1*b0, t11=(u128)a1*b1;
    u128 mid = t01 + t10;                       // < 2^128
    u128 Plo = t00, Phi = t11;
    uint64_t midlo=(uint64_t)(mid&MASK64), midhi=(uint64_t)(mid>>64);
    u128 add1=((u128)midlo)<<64, old=Plo;
    Plo += add1; if (Plo < old) Phi += 1;       // 桁上がり
    Phi += midhi;
    u128 lo127 = Plo & P127;
    u128 hi     = (Phi << 1) | (Plo >> 127);    // P >> 127
    return reduce127(lo127 + hi);
}

//1要素あたりの読み込みbit数を決める。
//fp_set_uiはmod還元するが、還元前のバイアスを避けるため常にqを下回る幅にする。
//バイト境界に揃えられるならその方が実装が単純なので、q>8ではq未満最大のバイト境界を、
//バイト境界が取れない小さいqではq-1bitを使う。
static unsigned load_bits_for_q(unsigned q){
    if (q > 8) {
        return ((q - 1) / 8) * 8;
    }
    return q - 1;
}

static unsigned q_for_output(size_t n_bits){
    if (n_bits == 0)   return 0;
    if (n_bits <= 64)  return 7;
    if (n_bits <= 256) return 31;
    if (n_bits <= 1016) return 127;
    return 0;
}
int field_select_for_output(size_t n_bits){
    unsigned q = q_for_output(n_bits);
    if (q == 0) return 0;                    // 未対応 -> 失敗
    FP_BITS    = q;
    P_MERSENNE = (((u128)1 << q) - 1);
    state_bytes = 2 * q;
    load_bits   = load_bits_for_q(q); //ブロックの1要素当たり何bitか 2^7-1→6bit 2^31-1→24bit 2^127-1→120bit
    block_bytes = (16 * load_bits) / 8; //1ブロック何byteか 2^7-1→12byte 2^31-1→48byte 2^127-1→240byte
    return 1;
}

uint64_t fp_mul_count = 0; // 乗算回数カウンタ
uint64_t fp_add_count = 0; // 加算回数カウンタ
uint64_t fp_sub_count = 0; // 減算回数カウンタ

// 64bitの計算結果を 31bit (mod p) に落とし込む関数
static inline uint32_t reduce_mersenne(uint64_t x) {
    // 上位31bitと下位31bitを足す
    // (x mod 2^31) + (x / 2^31) と等価
    uint32_t result = (uint32_t)(x & P_MERSENNE) + (uint32_t)(x >> FP_BITS);
    
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
    unsigned long long hi=(unsigned long long)(X->x0 >> 64);
    unsigned long long lo=(unsigned long long)(X->x0 & 0xFFFFFFFFFFFFFFFFULL);
    if (hi) gmp_printf("%llx%016llx", hi, lo);   // 上位があれば hex 連結
    else    gmp_printf("%llu", lo);              // 収まるなら10進
}

// 代入
void fp_set(fp_t *S, const fp_t *X){
    S->x0 = X->x0;
}

void fp_set_zero(fp_t *S){
    S->x0 = 0;
}

void fp_set_ui(fp_t *S, u128 x){
    S->x0 = x % P_MERSENNE;
}

int fp_is_equal(const fp_t *X, const fp_t *Y){
    return(X->x0 == Y->x0);
}


int fp_is_zero(const fp_t *X){
    return (X->x0 == 0);
}

void fp_random(fp_t *X){
    static int seeded = 0;
    if (!seeded){ srand((unsigned int)time(NULL)); seeded = 1; }
    u128 r = 0;
    for (int i = 0; i < 4; i++)                 // 32bit×4=128bit 埋める
        r = (r << 32) | (u128)((uint32_t)rand());
    if (FP_BITS <= 31) {
        uint32_t r = (uint32_t)rand();
        r ^= ((uint32_t)rand() << 16);      // ← 元と完全に同じ
        X->x0 = (u128)(r & P_MERSENNE);
    } else {
        u128 r = 0;
        for (int i=0;i<4;i++) r = (r<<32) | (u128)((uint32_t)rand());
        X->x0 = mulmod127(r, 1);
    }
    if (X->x0 == P_MERSENNE) X->x0 = 0;
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
    u128 sum = X->x0 + Y->x0;
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
    u128 x = X->x0;
    u128 y = Y->x0;
    
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
    if(FP_BITS <= 31){
        uint64_t prod = (uint64_t)X->x0 * (uint64_t)Y->x0;
        S->x0 = reduce_mersenne(prod);
    } else{
        S->x0 = mulmod127(X->x0, Y->x0);
    }
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
    mpz_init(exp);
    mpz_ui_pow_ui(exp, 2, FP_BITS); //2^7 or 2^31 or 2^127
    mpz_sub_ui(exp, exp, 1); // = p
    mpz_sub_ui(exp, exp, 2); // p - 2
    
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
    // (p-1)/2
    mpz_init(exp); mpz_ui_pow_ui(exp, 2, FP_BITS); mpz_sub_ui(exp, exp, 1); // = p
    mpz_sub_ui(exp, exp, 1);
    mpz_tdiv_q_ui(exp, exp, 2);

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
    mpz_mul_2exp(exp, exp, FP_BITS - 2); // 2^(FP_BITS - 2)

    // べき乗計算だけで平方根が求まる！ (Tonelli-Shanks不要)
    fp_pow(b, a, exp);

    mpz_clear(exp);
    return 1;
}
