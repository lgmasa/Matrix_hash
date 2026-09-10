/*
 * fp4_is_square_multi.c
 * -----------------------------------------------------------------------------
 * 素数 p を引数に取る, 独立した (production の fp/fp4 に依存しない) fp4 平方剰余
 * 判定。GMP 上で Fp4 演算を行うので, 2^127-1 も含めて任意の Mersenne 素数を
 * そのまま扱える。3素数 (2^7-1, 2^31-1, 2^127-1) を同時にテストして, すべてで
 * 平方非剰余になる α (= x^4 - α を既約にする θ) を探すための検証ツール。
 *
 *   体構造は本体の 16_fp4.c と厳密に一致させている:
 *     - Type-I CVMA (h=1), 基底 {γ, γ^p, γ^{p^2}, γ^{p^3}}
 *     - σ 写像: (0,1)->3,(0,2)->4,(0,3)->2,(1,2)->0,(1,3)->4,(2,3)->1
 *     - 単位元 1 = -(γ+γ^p+γ^{p^2}+γ^{p^3}) すなわち座標 (-1,-1,-1,-1)
 *
 *   判定は Euler の規準: α != 0 かつ α^((p^4-1)/2) != 1  <=>  平方非剰余。
 *   (Fp4 では q=p^4≡1 mod4 なので「非剰余」だけで x^4-α 既約が保証される。)
 *
 * build: gcc -O2 fp4_is_square_multi.c -lgmp -o fp4_is_square_multi
 * -----------------------------------------------------------------------------
 */
#include <stdio.h>
#include <gmp.h>

/* GMP 版 fp4 元: 正規基底の4座標 */
typedef struct { mpz_t c[4]; } fp4g_t;

static void fp4g_init (fp4g_t *X){ for(int i=0;i<4;i++) mpz_init(X->c[i]); }
static void fp4g_clear(fp4g_t *X){ for(int i=0;i<4;i++) mpz_clear(X->c[i]); }

/* 小さい整数座標 (負も可) から設定し, mod p に正規化 */
static void fp4g_set_si(fp4g_t *X, const long a[4], const mpz_t p){
    for(int i=0;i<4;i++){ mpz_set_si(X->c[i], a[i]); mpz_mod(X->c[i], X->c[i], p); }
}
/* 単位元 1 = (-1,-1,-1,-1) mod p */
static void fp4g_set_one(fp4g_t *X, const mpz_t p){
    for(int i=0;i<4;i++){ mpz_sub_ui(X->c[i], p, 1); }
}
static int fp4g_is_one(const fp4g_t *X, const mpz_t p){
    mpz_t m1; mpz_init(m1); mpz_sub_ui(m1, p, 1);
    int r = 1; for(int i=0;i<4;i++) if(mpz_cmp(X->c[i], m1)!=0){ r=0; break; }
    mpz_clear(m1); return r;
}
static int fp4g_is_zero(const fp4g_t *X){
    for(int i=0;i<4;i++) if(mpz_sgn(X->c[i])!=0) return 0; return 1;
}

/* S = X * Y over Fp4 (本体 fp4_mul と同一の CVMA) */
static void fp4g_mul(fp4g_t *S, const fp4g_t *X, const fp4g_t *Y, const mpz_t p){
    mpz_t v[5], dx, dy, t;
    for(int i=0;i<5;i++) mpz_init(v[i]);
    mpz_inits(dx, dy, t, NULL);

    /* 対角 v0..v3 */
    for(int i=0;i<4;i++){ mpz_mul(v[i], X->c[i], Y->c[i]); mpz_mod(v[i], v[i], p); }

    /* 各 (i,j) -> σ の交差項. マクロで簡潔に */
    #define CROSS(i,j,sig) \
        mpz_sub(dx, X->c[i], X->c[j]); mpz_sub(dy, Y->c[i], Y->c[j]); \
        mpz_mul(t, dx, dy); mpz_add(v[sig], v[sig], t); mpz_mod(v[sig], v[sig], p);

    mpz_set_ui(v[4], 0);
    CROSS(0,1,3)
    CROSS(0,2,4)
    CROSS(0,3,2)
    CROSS(1,2,0)
    CROSS(1,3,4)
    CROSS(2,3,1)
    #undef CROSS

    /* z_l = v4 - v_l */
    for(int i=0;i<4;i++){ mpz_sub(S->c[i], v[4], v[i]); mpz_mod(S->c[i], S->c[i], p); }

    for(int i=0;i<5;i++) mpz_clear(v[i]);
    mpz_clears(dx, dy, t, NULL);
}

/* R = X^e over Fp4 (square-and-multiply) */
static void fp4g_pow(fp4g_t *R, const fp4g_t *X, const mpz_t e, const mpz_t p){
    fp4g_t base, acc; fp4g_init(&base); fp4g_init(&acc);
    for(int i=0;i<4;i++) mpz_set(base.c[i], X->c[i]);
    fp4g_set_one(&acc, p);
    for(long b = mpz_sizeinbase(e,2)-1, k=0; k<=b; k++){
        if(mpz_tstbit(e, k)){ fp4g_t tmp; fp4g_init(&tmp);
            fp4g_mul(&tmp, &acc, &base, p);
            for(int i=0;i<4;i++) mpz_swap(acc.c[i], tmp.c[i]); fp4g_clear(&tmp); }
        fp4g_t sq; fp4g_init(&sq);
        fp4g_mul(&sq, &base, &base, p);
        for(int i=0;i<4;i++) mpz_swap(base.c[i], sq.c[i]); fp4g_clear(&sq);
    }
    for(int i=0;i<4;i++) mpz_set(R->c[i], acc.c[i]);
    fp4g_clear(&base); fp4g_clear(&acc);
}

/* 平方剰余判定: 0 は平方とみなし 1 を返す。非剰余なら 0 を返す。
 * すなわち返り値 1 = 平方 (剰余), 0 = 非剰余。 */
static int fp4g_is_square(const fp4g_t *X, const mpz_t p){
    if(fp4g_is_zero(X)) return 1;               /* 0 は平方扱い */
    mpz_t e; mpz_init(e);
    mpz_pow_ui(e, p, 4); mpz_sub_ui(e, e, 1); mpz_tdiv_q_ui(e, e, 2); /* (p^4-1)/2 */
    fp4g_t res; fp4g_init(&res);
    fp4g_pow(&res, X, e, p);
    int is_sq = fp4g_is_one(&res, p);
    fp4g_clear(&res); mpz_clear(e);
    return is_sq;
}

/* おまけ: Fp4 では「非ゼロかつ非剰余」<=> x^4-α 既約 */
static int fp4g_x4_minus_irreducible(const fp4g_t *X, const mpz_t p){
    return (!fp4g_is_zero(X)) && (!fp4g_is_square(X, p));
}

/* ---- ドライバ ---- */
static void mersenne(mpz_t p, unsigned q){          /* p = 2^q - 1 */
    mpz_ui_pow_ui(p, 2, q); mpz_sub_ui(p, p, 1);
}

int main(void){
    /* 3素数 */
    unsigned qs[3] = {7, 31, 127};
    const char *names[3] = {"2^7-1", "2^31-1", "2^127-1"};
    mpz_t P[3];
    for(int k=0;k<3;k++){ mpz_init(P[k]); mersenne(P[k], qs[k]); }

    /* --- 各素数ごとの判定を数例で提示 --- */
    long demo[][4] = { {1,0,0,6}, {1,2,0,0}, {1,0,0,0}, {2,0,0,0}, {3,1,4,1} };
    printf("=== 各素数ごとの平方非剰余判定 (NR=非剰余, QR=剰余) ===\n");
    printf("%-14s %8s %8s %8s   3素数共通NR?\n", "alpha", names[0], names[1], names[2]);
    for(int d=0; d<5; d++){
        fp4g_t A; fp4g_init(&A);
        int all_nr = 1; char buf[3][4];
        for(int k=0;k<3;k++){
            fp4g_set_si(&A, demo[d], P[k]);
            int sq = fp4g_is_square(&A, P[k]);      /* 1=QR, 0=NR */
            snprintf(buf[k], 4, "%s", sq ? "QR" : "NR");
            if(sq) all_nr = 0;
        }
        printf("(%ld,%ld,%ld,%ld)%*s %8s %8s %8s      %s\n",
               demo[d][0],demo[d][1],demo[d][2],demo[d][3],
               (int)(14-11), "", buf[0], buf[1], buf[2], all_nr?"YES":"no");
        fp4g_clear(&A);
    }

    /* --- 小さい非負座標を全探索し, 3素数共通の非剰余を疎さ順に --- */
    const int R = 6;
    printf("\n=== 座標 0..%d で3素数すべて非剰余 (=x^4-α 既約) になる α ===\n", R);
    int count = 0, shown = 0;
    /* 非ゼロ座標数の少ない順に見せたいので, まず nz=1..4 の順で走査 */
    for(int want_nz=1; want_nz<=4; want_nz++){
        long a[4];
        for(a[0]=0;a[0]<=R;a[0]++)for(a[1]=0;a[1]<=R;a[1]++)
        for(a[2]=0;a[2]<=R;a[2]++)for(a[3]=0;a[3]<=R;a[3]++){
            int nz=0; for(int i=0;i<4;i++) nz += (a[i]!=0);
            if(nz!=want_nz) continue;
            fp4g_t A; fp4g_init(&A);
            int all_nr = 1;
            for(int k=0;k<3;k++){
                fp4g_set_si(&A, a, P[k]);
                if(fp4g_is_square(&A, P[k])){ all_nr=0; break; }
            }
            fp4g_clear(&A);
            if(all_nr){
                count++;
                if(shown < 8){
                    printf("  α=(%ld,%ld,%ld,%ld)  非ゼロ数=%d 係数和=%ld\n",
                           a[0],a[1],a[2],a[3], nz, a[0]+a[1]+a[2]+a[3]);
                    shown++;
                }
            }
        }
    }
    printf("  ... 共通 α の総数 = %d 個\n", count);

    for(int k=0;k<3;k++) mpz_clear(P[k]);
    return 0;
}