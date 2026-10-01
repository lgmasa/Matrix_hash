/*
 * xof.c --- MATRIX-XOF(スポンジ構造)
 *
 *  既存のファイル・関数には手を加えず、次をそのまま使う:
 *    field_select_for_output / load_bits(素数の選択と1要素あたりのビット数)
 *    load_bits_be(メッセージブロックからの1要素の読み込み)
 *    fp_init / fp_set_ui / fp_add(レート部への足し込み)
 *    state_init / state_set_zero / state_clear
 *    matrix_permutation_P(置換 P。ハッシュモードと共有)
 *
 *  中身(吸収・絞り出し)は matrix_xof() の1つだけ。
 *  xof_31 / xof_127 はパラメータを固定した入口で、比較のために用意している。
 *
 *  状態 16 要素のうち、行優先で先頭 r_e 個をレート部、残り c_e 個をキャパシティ部とする。
 *     k = 0 .. 15  ->  S.m[k/4][k%4]
 *
 *  128bit 強度を狙ったパラメータ(c = c_e * q >= 256):
 *     xof_127 : q=127, c_e=3 (381bit), r_e=13, 1要素15byte, 1ブロック195byte
 *     xof_31  : q=31,  c_e=9 (279bit), r_e=7,  1要素 3byte, 1ブロック 21byte
 *
 *  メッセージの読み込みには 16_matrix.c の load_bits_be をそのまま使う
 *  (static を外したもの。プロトタイプは xof.h で宣言)。
 */
#include "16_header.h"

#define XOF_DOMAIN          0x1F   /* ドメイン分離 + pad10*1 の先頭の1(SHAKE と同じ流儀) */
#define XOF_MAX_RATE_BYTES  256

/* ---- 比較対象のパラメータ(ここを変えれば比較条件が変わる) ---- */
#define XOF31_Q         31
#define XOF31_CAP       9
#define XOF127_Q        127
#define XOF127_CAP      3

static inline fp_t *xof_elem(state_t *S, unsigned k){
    return &S->m[k / 4][k % 4]; 
}

/* 素数 q を選ぶ。既存の field_select_for_output に 8*q を渡すと
 * ちょうど q が選ばれる(<=64->7, <=256->31, <=1016->127)ので、それを使う */
static int xof_select_prime(unsigned q){
    if(!field_select_for_output((size_t)8 * q)) return 0;
    return FP_BITS == q;
}

/* 吸収: レート部の各要素に load_bits ビットずつ Fp 加算で足し込む。
 * 読み込みはハッシュ側と同じ load_bits_be(MSB ファースト)を使う。
 * load_bits < q なので値は必ず p 未満(ハッシュ側の 24bit 制限と同じ理由で単射) */
static void xof_absorb_block(state_t *S, const uint8_t *blk, unsigned r_e){
    size_t bitpos = 0;
    for(unsigned k = 0; k < r_e; k++){
        u128 v = load_bits_be(blk, bitpos, load_bits);
        fp_t t;
        fp_init(&t);
        fp_set_ui(&t, v);
        fp_add(xof_elem(S, k), xof_elem(S, k), &t);
        bitpos += load_bits;
    }
}

/* 絞り出し: レート部の各要素の下位 lb バイトを big-endian で取り出す。
 * キャパシティ部(k >= r_e)は絶対に出力しない */
static void xof_squeeze_block(uint8_t *blk, state_t *S, unsigned r_e, size_t lb){
    for(unsigned k = 0; k < r_e; k++){
        u128 v = xof_elem(S, k)->x0;
        for(size_t b = 0; b < lb; b++)
            blk[k * lb + (lb - 1 - b)] = (uint8_t)(v >> (8 * b));
    }
}

/* ===================== 共通の中身 ===================== */
int matrix_xof(uint8_t *out, size_t out_len,
               const uint8_t *msg, size_t msg_len,
               unsigned q, unsigned cap_elems,
               int rounds, const state_t *MDS, const affine16_t *AFF){
    if(out == NULL || (msg == NULL && msg_len > 0) || MDS == NULL || AFF == NULL) return 0;
    if(!xof_select_prime(q))              return 0;
    if(cap_elems == 0 || cap_elems >= 16) return 0;
    if(load_bits % 8 != 0)                return 0;   /* 2^7-1(6bit/要素)は対象外:
                                                         パディングと出力をバイト単位で行うため */

    const unsigned r_e        = 16 - cap_elems;
    const size_t   lb         = load_bits / 8;        /* 31 -> 3, 127 -> 15 */
    const size_t   rate_bytes = r_e * lb;
    if(rate_bytes == 0 || rate_bytes > XOF_MAX_RATE_BYTES) return 0;

    /* 準備: 状態を全0に */
    state_t S;
    state_init(&S);
    state_set_zero(&S);

    /* 吸収: 完全なブロック */
    size_t off = 0;
    while(msg_len - off >= rate_bytes){
        xof_absorb_block(&S, msg + off, r_e);
        matrix_permutation_P(&S, &S, rounds, MDS, AFF);
        off += rate_bytes;
    }

    /* 最後のブロック: パディング + ドメイン分離。
     * 長さがレートの倍数でも、パディングだけのブロックを必ず1つ吸収する */
    uint8_t last[XOF_MAX_RATE_BYTES];
    size_t rem = msg_len - off;              /* rem < rate_bytes */
    memset(last, 0, rate_bytes);
    if(rem > 0) memcpy(last, msg + off, rem);
    last[rem]            ^= XOF_DOMAIN;
    last[rate_bytes - 1] ^= 0x80;
    xof_absorb_block(&S, last, r_e);
    matrix_permutation_P(&S, &S, rounds, MDS, AFF);

    /* 絞り出し: レート部を出す -> 足りなければ P -> また出す */
    size_t done = 0;
    while(done < out_len){
        uint8_t blk[XOF_MAX_RATE_BYTES];
        xof_squeeze_block(blk, &S, r_e, lb);
        size_t n = out_len - done;
        if(n > rate_bytes) n = rate_bytes;
        memcpy(out + done, blk, n);
        done += n;
        if(done < out_len)
            matrix_permutation_P(&S, &S, rounds, MDS, AFF);
    }

    state_clear(&S);
    return 1;
}

/* ===================== 比較用の入口 ===================== */
int xof_31(uint8_t *out, size_t out_len, const uint8_t *msg, size_t msg_len,
           const state_t *MDS, const affine16_t *AFF){
    return matrix_xof(out, out_len, msg, msg_len,
                      XOF31_Q, XOF31_CAP, MATRIX_ROUNDS, MDS, AFF);
}

int xof_127(uint8_t *out, size_t out_len, const uint8_t *msg, size_t msg_len,
            const state_t *MDS, const affine16_t *AFF){
    return matrix_xof(out, out_len, msg, msg_len,
                      XOF127_Q, XOF127_CAP, MATRIX_ROUNDS, MDS, AFF);
}

/* ===================== パラメータ情報 ===================== */
/* 素数の選択(グローバル変数の書き換え)を起こさずに計算するため、
 * 1要素あたりのバイト数は load_bits_for_q と同じ規則(q-1 以下で最大の8の倍数)で求める */
xof_info xof_get_info(unsigned q, unsigned cap_elems){
    xof_info in;
    size_t lb = (q > 8) ? (size_t)((q - 1) / 8) : 0;
    in.q             = q;
    in.cap_elems     = cap_elems;
    in.rate_elems    = 16 - cap_elems;
    in.rate_bytes    = in.rate_elems * lb;
    in.capacity_bits = cap_elems * q;
    in.security_bits = in.capacity_bits / 2;
    in.state_bits    = 16 * q;
    return in;
}

xof_info xof_31_info(void)  { return xof_get_info(XOF31_Q,  XOF31_CAP);  }
xof_info xof_127_info(void) { return xof_get_info(XOF127_Q, XOF127_CAP); }