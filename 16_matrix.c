#include "16_header.h"


typedef void (*fp16_inv_func)(fp16_t *S, const fp16_t *X);

void state_init(state_t *S){
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            fp_init(&S->m[i][j]);
        }
    }
}

void state_set_zero(state_t *S){
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            S->m[i][j].x0 = 0;
        }
    }
}

void state_random(state_t *S){
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            fp_random(&S->m[i][j]);
        }
    }
}

int state_is_zero(const state_t *S){
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (!fp_is_zero(&S->m[i][j])) return 0;
        }
    }
    return 1;
}

void state_from_fp16(state_t *S, const fp16_t *X){
    const fp4_t *rows[4] = {&X->x0, &X->x1, &X->x2, &X->x3};
    for (int i = 0; i < 4; i++) {
        S->m[i][0] = rows[i]->x0;
        S->m[i][1] = rows[i]->x1;
        S->m[i][2] = rows[i]->x2;
        S->m[i][3] = rows[i]->x3;
    }
}

void state_to_fp16(fp16_t *S, const state_t *X){
    fp4_t *rows[4] = {&S->x0, &S->x1, &S->x2, &S->x3};
    for (int i = 0; i < 4; i++) {
        rows[i]->x0 = X->m[i][0];
        rows[i]->x1 = X->m[i][1];
        rows[i]->x2 = X->m[i][2];
        rows[i]->x3 = X->m[i][3];
    }
}


void state_clear(state_t *S){
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            fp_clear(&S->m[i][j]);
        }
    }
}

void state_copy(state_t *dst, const state_t *src){
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            fp_set(&dst->m[i][j], &src->m[i][j]);
        }
    }
}

void state_add(state_t *Z, const state_t *X, const state_t *Y){
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            fp_add(&Z->m[i][j], &X->m[i][j], &Y->m[i][j]);
        }
    }
}

void state_sub(state_t *Z, const state_t *X, const state_t *Y){
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            fp_sub(&Z->m[i][j], &X->m[i][j], &Y->m[i][j]);
        }
    }
}

void state_add3(state_t *Z, const state_t *A, const state_t *B, const state_t *C){
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            fp_t tmp;
            fp_add(&tmp, &A->m[i][j], &B->m[i][j]);
            fp_add(&Z->m[i][j], &tmp, &C->m[i][j]);
        }
    }
}

int  state_equal(const state_t *A, const state_t *B){
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (!fp_is_equal(&A->m[i][j], &B->m[i][j])) return 0;
        }
    }
    return 1;
}

void state_print(const state_t *S){

    for (int i = 0; i < 4; i++) {
        printf("  [");
        for (int j = 0; j < 4; j++) {
            fp_printf(&S->m[i][j]);
            if (j < 3) printf(", ");
        }
        printf("]\n");
    }
}

//MDS行列×状態行列の各列
void state_mix_column(fp_t y[4], const fp_t x[4], const state_t *M){
    fp_t acc;
    fp_t term;

    fp_init(&acc);
    fp_init(&term);

    for (int i = 0; i < 4; i++) {
        fp_set_zero(&acc);

        for (int j = 0; j < 4; j++) {
            // term = M[i][j] * x[j]
            fp_mul(&term, &M->m[i][j], &x[j]);

            // acc += term
            fp_add(&acc, &acc, &term);
        }

        fp_set(&y[i], &acc);
    }

    fp_clear(&acc);
    fp_clear(&term);
}

void matrix_mixbytes(state_t *S_new, const state_t *S, const state_t *M){
    fp_t x[4];
    fp_t y[4];

    for (int i = 0; i < 4; i++) {
        fp_init(&x[i]);
        fp_init(&y[i]);
    }

    for (int col = 0; col < 4; col++) {
        // col列を取り出す
        for (int row = 0; row < 4; row++) {
            fp_set(&x[row], &S->m[row][col]);
        }

        // y = M x
        state_mix_column(y, x, M);

        // 結果を書き戻す
        for (int row = 0; row < 4; row++) {
            fp_set(&S_new->m[row][col], &y[row]);
        }
    }

    for (int i = 0; i < 4; i++) {
        fp_clear(&x[i]);
        fp_clear(&y[i]);
    }
}

//shift[4]=[0,1,2,3]なら、0行目はシフトなし、1行目は左に1ビットシフト、2行目は左に2ビットシフト、3行目は左に3ビットシフトさせる関数
void matrix_shiftbytes(state_t *S_new, const state_t *S, const int shift[4]){
    for (int row = 0; row < 4; row++){
        int s = shift[row] % 4;

        for (int col = 0; col < 4; col++){
            fp_set(&S_new->m[row][col], &S->m[row][(col + s) % 4]);
        }
    }
}

void matrix_shiftbytes_P(state_t *S_new, const state_t *S){
    const int shift_P[4] = {0,1,2,3};
    matrix_shiftbytes(S_new,S,shift_P);
}

void matrix_shiftbytes_Q(state_t *S_new, const state_t *S){
    const int shift_Q[4] = {1,3,0,2};
    matrix_shiftbytes(S_new,S,shift_Q);
}

void affine16_init(affine16_t *AFF){
    for (int i = 0; i < 16; i++){
        fp_init(&AFF->b[i]);
        for (int j = 0; j < 16; j++){
            fp_init(&AFF->A[i][j]);
        }
    }
}

void affine16_clear(affine16_t *AFF){
    for (int i = 0; i < 16; i++){
        fp_clear(&AFF->b[i]);
        for (int j = 0; j < 16; j++){
            fp_clear(&AFF->A[i][j]);
        }
    }
}

void affine16_print_A(const affine16_t *F){
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            printf("%u ", F->A[i][j].x0);
        }
        printf("\n");
    }
}

void affine16_print_b(const affine16_t *F)
{
    for (int i = 0; i < 16; i++) {
        printf("%u ", F->b[i].x0);
    }

    printf("\n");
}

//4*4を16*1に変換する関数
void state_to_vec16(fp_t v[16], const state_t *S){
    int idx = 0;

    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            fp_set(&v[idx], &S->m[i][j]);
            idx++;
        }
    }
}

//16*1を4*4に変換する関数
void vec16_to_state(state_t *S, const fp_t v[16]){
    int idx = 0;

    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            fp_set(&S->m[i][j], &v[idx]);
            idx++;
        }
    }
}

//Aに値をセットする関数
void affine16_set_A(affine16_t *AFF)
{
    for (int i = 0; i < 16; i++) {
        fp_set_zero(&AFF->b[i]);

        for (int j = 0; j < 16; j++) {
            fp_set_zero(&AFF->A[i][j]);
        }
    }

    for (int i = 0; i < 16; i++) {
        fp_set_ui(&AFF->A[i][i], 1);
        fp_set_ui(&AFF->A[i][(i + 4) % 16], 1);
        fp_set_ui(&AFF->A[i][(i + 5) % 16], 1);
        fp_set_ui(&AFF->A[i][(i + 6) % 16], 1);
        fp_set_ui(&AFF->A[i][(i + 7) % 16], 1);
    }
}

void affine16_set_b(affine16_t *AFF)
{
    for (int i = 0; i < 16; i++) {
        fp_set_ui(&AFF->b[i], i + 1);
    }
}

//A,bをセットする関数(一旦Aは単位ベクトル、bは零ベクトルでセット)
void affine16_set(affine16_t *AFF){
    for (int i = 0; i < 16; i++){
        fp_set_zero(&AFF->b[i]);
        for (int j = 0; j < 16; j++){
        fp_set_zero(&AFF->A[i][j]);
        }
    }

    for (int i = 0; i < 16; i++){
        fp_set_ui(&AFF->A[i][i], 1);
    }
}

//16*16行列Aが正則行列(逆行列を持つ)かどうかを判定する関数
//素体F_p上でガウスの消去法による前進消去を行い、全ての列でピボットが立てば正則
//正則なら1、特異(正則でない)なら0を返す
int affine16_is_regular(const affine16_t *AFF){
    fp_t M[16][16];

    //Aをコピー(消去法で破壊的に変更するため)
    for (int i = 0; i < 16; i++){
        for (int j = 0; j < 16; j++){
            fp_set(&M[i][j], &AFF->A[i][j]);
        }
    }

    fp_t inv, factor, term, tmp;
    fp_init(&inv);
    fp_init(&factor);
    fp_init(&term);
    fp_init(&tmp);

    int regular = 1;

    for (int col = 0; col < 16; col++){
        //col列で非零成分(ピボット)を持つ行をcol行以降から探す
        int pivot = -1;
        for (int row = col; row < 16; row++){
            if (!fp_is_zero(&M[row][col])){
                pivot = row;
                break;
            }
        }

        //ピボットが見つからない→この列は消去後すべて0→正則ではない
        if (pivot == -1){
            regular = 0;
            break;
        }

        //ピボット行を対角位置(col行)に移動
        if (pivot != col){
            for (int j = 0; j < 16; j++){
                fp_set(&tmp, &M[col][j]);
                fp_set(&M[col][j], &M[pivot][j]);
                fp_set(&M[pivot][j], &tmp);
            }
        }

        //ピボットの逆元
        fp_inv(&inv, &M[col][col]);

        //col行より下の行のcol列成分を消去
        for (int row = col + 1; row < 16; row++){
            if (fp_is_zero(&M[row][col])) continue;

            //factor = M[row][col] / M[col][col]
            fp_mul(&factor, &M[row][col], &inv);

            //row行 = row行 - factor * col行
            for (int j = col; j < 16; j++){
                fp_mul(&term, &factor, &M[col][j]);
                fp_sub(&M[row][j], &M[row][j], &term);
            }
        }
    }

    fp_clear(&inv);
    fp_clear(&factor);
    fp_clear(&term);
    fp_clear(&tmp);

    return regular;
}

//xを16*1に変換した後に使う
void affine16_apply_vec(fp_t y[16], const fp_t x[16], const affine16_t *AFF){
    fp_t acc;
    fp_t term;

    fp_init(&acc);
    fp_init(&term);

    for (int i = 0; i <16; i++){
        fp_set_zero(&acc);
        for (int j = 0; j < 16; j++){
            //term = A[i][j] * x[j]
            fp_mul(&term, &AFF->A[i][j], &x[j]);

            //acc = acc + term
            fp_add(&acc, &acc, &term);
        }
        //y[i] = acc + b[i]
        fp_add(&y[i], &acc, &AFF->b[i]);
    }
    fp_clear(&acc);
    fp_clear(&term);
}

void matrix_affine(state_t *S_new, const state_t *S, const affine16_t *AFF){
    fp_t x[16];
    fp_t y[16];

    for (int i = 0; i < 16; i++){
        fp_init(&x[i]);
        fp_init(&y[i]);
    }

    //逆元計算を終えたS(4*4行列)をx(16*1行列)に変換する
    state_to_vec16(x, S);

    //y=A*x+bを行う(yは16*1行列、Aは16*16行列、bは16*1行列)
    affine16_apply_vec(y, x, AFF);

    //yを16*1から4*4行列に戻す
    vec16_to_state(S_new, y);

    for (int i = 0; i < 16; i++){
        fp_clear(&x[i]);
        fp_clear(&y[i]);
    }
}



//xにSをセットし、y=1/x→yをS_newにセット
void matrix_subbytes(state_t *S_new, const state_t *S, const affine16_t *AFF){
    fp16_t x,y;
    state_t S_inv;
    fp16_init(&x);
    fp16_init(&y);
    state_init(&S_inv);

    state_to_fp16(&x,S);

    if(fp16_is_zero(&x)){
        fp16_set_zero(&y);
    } else{
        //逆元計算
        fp16_inv(&y,&x);
    }

    state_from_fp16(&S_inv,&y);

    //affine変換
    matrix_affine(S_new, &S_inv, AFF);

    state_clear(&S_inv);
    fp16_clear(&x);
    fp16_clear(&y);
}

//Pでは1行目にラウンド固定の値を入れる
void matrix_add_round_constant_P(state_t *S_new, const state_t *S, int r){
    state_copy(S_new,S);
    fp_t c;
    fp_init(&c);

    fp_set_ui(&c, (0x00 ^ r));
    fp_add(&S_new->m[0][0], &S_new->m[0][0], &c);

    fp_set_ui(&c, (0x10 ^ r));
    fp_add(&S_new->m[0][1], &S_new->m[0][1], &c);

    fp_set_ui(&c, (0x20 ^ r));
    fp_add(&S_new->m[0][2], &S_new->m[0][2], &c);

    fp_set_ui(&c, (0x30 ^ r));
    fp_add(&S_new->m[0][3], &S_new->m[0][3], &c);

    fp_clear(&c);
}

void matrix_add_round_constant_Q(state_t *S_new, const state_t *S, int r){
    state_copy(S_new,S);
    fp_t c;
    fp_init(&c);

    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            unsigned int val = 0xff;

            //4行目にラウンド固定の値を入れる
            if(i == 3){
                if(j == 0) val = 0xff ^ r;
                if(j == 1) val = 0xef ^ r;
                if(j == 2) val = 0xdf ^ r;
                if(j == 3) val = 0xcf ^ r;
            }

            //4行目以外はここでffが加算され、4行目はさっき計算した値が加算される
            fp_set_ui(&c, val);
            fp_add(&S_new->m[i][j], &S_new->m[i][j], &c);
        }
    }
    fp_clear(&c);
}

//1ラウンド分の処理
void matrix_round_P(state_t *S_new, const state_t *S, int r, const state_t *MDS, const affine16_t *AFF){
    state_t T1, T2, T3;

    state_init(&T1);
    state_init(&T2);
    state_init(&T3);

    //add round constant
    matrix_add_round_constant_P(&T1,S,r);


    //subbytes
    matrix_subbytes(&T2,&T1, AFF);



    //shiftbytes
    matrix_shiftbytes_P(&T3,&T2);



    //mixbytes
    matrix_mixbytes(S_new,&T3,MDS);



    state_clear(&T1);
    state_clear(&T2);
    state_clear(&T3);
}

void matrix_round_Q(state_t *S_new, const state_t *S, int r, const state_t *MDS, const affine16_t *AFF){
    state_t T1, T2, T3;
    state_init(&T1);
    state_init(&T2);
    state_init(&T3);

    //add round constant
    matrix_add_round_constant_Q(&T1,S,r);

    //subbytes
    matrix_subbytes(&T2,&T1, AFF);

    //shiftbytes
    matrix_shiftbytes_Q(&T3,&T2);

    //mixbytes
    matrix_mixbytes(S_new, &T3, MDS);

    state_clear(&T1);
    state_clear(&T2);
    state_clear(&T3);
}

//matrix_round_Pをround回分行う関数
void matrix_permutation_P(state_t *S_new, const state_t *S, int rounds, const state_t *MDS, const affine16_t *AFF){
    state_t cur, next;

    state_init(&cur);
    state_init(&next);

    state_copy(&cur,S);

    for(int r = 0; r < rounds; r++){
        matrix_round_P(&next, &cur, r, MDS, AFF);
        state_copy(&cur, &next);
    }
    state_copy(S_new, &cur);
    state_clear(&cur);
    state_clear(&next);
}

//matrix_round_Qをround回分行う関数
void matrix_permutation_Q(state_t *S_new, const state_t *S, int rounds, const state_t *MDS, const affine16_t *AFF){
    state_t cur, next;

    state_init(&cur);
    state_init(&next);

    state_copy(&cur,S);

    for(int r = 0; r < rounds; r++){
        matrix_round_Q(&next, &cur, r, MDS, AFF);
        state_copy(&cur, &next);
    }
    state_copy(S_new, &cur);
    state_clear(&cur);
    state_clear(&next);
}

//圧縮関数(out:次の状態行列、h:現在の状態行列、m:圧縮したいメッセージブロック)
void matrix_compression(state_t *out, const state_t *h, const state_t *m, int rounds, const state_t *MDS, const affine16_t *AFF){
    state_t hm;
    state_t p_out;
    state_t q_out;

    state_init(&hm);
    state_init(&p_out);
    state_init(&q_out);

    //hm = h + m
    state_add(&hm, h, m);

    //p_out = P(h + m)
    matrix_permutation_P(&p_out, &hm, rounds, MDS, AFF);

    //q_out = Q(m)
    matrix_permutation_Q(&q_out, m, rounds, MDS, AFF);

    // out = p_out + q_out + h
    state_add3(out, &p_out, &q_out, h);

    state_clear(&hm);
    state_clear(&p_out);
    state_clear(&q_out);
}

//P(h)+hを計算←まだここでは出力は行列のまま
void matrix_output_transform(state_t *out, const state_t *h, int rounds, const state_t *MDS, const affine16_t *AFF){
    state_t p_out;
    state_init(&p_out);

    //p_out = P(h)
    matrix_permutation_P(&p_out, h, rounds, MDS, AFF);

    //out = P(h) + h
    state_add(out, &p_out, h);

    state_clear(&p_out);
}

//fp4の値をout[16]に格納→out[i]はそれぞれ1byteの大きさになる8bit(1byte)*16=128bit(16byte)
static void fp4_to_bytes_128(uint8_t *out, const fp4_t *x){
    uint32_t v;

    v = x->x0.x0;
    out[0] = (uint8_t)((v >> 24) & 0xff);
    out[1] = (uint8_t)((v >> 16) & 0xff);
    out[2] = (uint8_t)((v >> 8) & 0xff);  //v >> 8では、32-8=24で上位24ビット取り出してしまうので、その中の下位8ビットだけが欲しい→&0xffする(0xffは0~7bitすべて1)
    out[3] = (uint8_t)(v & 0xff);

    v = x->x1.x0;
    out[4] = (uint8_t)((v >> 24) & 0xff);
    out[5] = (uint8_t)((v >> 16) & 0xff);
    out[6] = (uint8_t)((v >> 8) & 0xff);
    out[7] = (uint8_t)(v & 0xff);

    v = x->x2.x0;
    out[8] = (uint8_t)((v >> 24) & 0xff);
    out[9] = (uint8_t)((v >> 16) & 0xff);
    out[10] = (uint8_t)((v >> 8) & 0xff);
    out[11] = (uint8_t)(v & 0xff);

    v = x->x3.x0;
    out[12] = (uint8_t)((v >> 24) & 0xff);
    out[13] = (uint8_t)((v >> 16) & 0xff);
    out[14] = (uint8_t)((v >> 8) & 0xff);
    out[15] = (uint8_t)(v & 0xff);
}

//fp4_to_bytes_128 は31ビット詰めでは使わない(残しても無害だが未使用警告が出る場合あり)

//fp16の16要素(各FP_BITSbit)を隙間なくMSBファーストで詰める。16*FP_BITS bit = state_bytes*8 (常に8の倍数)。
void fp16_to_bytes_512(uint8_t out[MATRIX_STATE_BYTES_MAX], const fp16_t *x){
    //16個のFP_BITSbit係数を所定の順序で集める
    const fp4_t *blk[4] = { &x->x0, &x->x1, &x->x2, &x->x3 };
    uint32_t vals[16];
    int idx = 0;
    uint32_t mask = (FP_BITS >= 32) ? 0xffffffffu : ((1u << FP_BITS) - 1u);
    for(int i = 0; i < 4; i++){
        vals[idx++] = blk[i]->x0.x0 & mask;
        vals[idx++] = blk[i]->x1.x0 & mask;
        vals[idx++] = blk[i]->x2.x0 & mask;
        vals[idx++] = blk[i]->x3.x0 & mask;
    }

    //MSBファーストでFP_BITSbitずつ詰める
    memset(out, 0, state_bytes);
    size_t bitpos = 0;
    for(int e = 0; e < 16; e++){
        uint32_t v = vals[e];
        for(int b = (int)FP_BITS - 1; b >= 0; b--){   //FP_BITSbit、上位ビットから
            if((v >> b) & 1u)
                out[bitpos >> 3] |= (uint8_t)(0x80u >> (bitpos & 7));
            bitpos++;
        }
    }
    //bitpos == 16*FP_BITS == state_bytes*8 (16は8の倍数なので端数は出ない)
}

//行列からビット列(512bit)に変換
void matrix_state_to_bytes_512(uint8_t out[MATRIX_STATE_BYTES_MAX], const state_t *S){
    fp16_t x;
    fp16_init(&x);

    state_to_fp16(&x,S);
    fp16_to_bytes_512(out, &x);

    fp16_clear(&x);
}

//ビット列から任意のbyte数だけ取り出す関数
int matrix_trunc_tail_bytes(uint8_t *digest, size_t digest_len, const uint8_t full_state[MATRIX_STATE_BYTES_MAX]){
    if(digest == NULL || full_state == NULL){  //なぜdigest == NULLを確認？→NULLアドレスに書き込むとクラッシュするから
        return 0;
    }

    if(digest_len > state_bytes){
        return 0;
    }

    // size_t start = MATRIX_STATE_BYTES - digest_len; //末尾のbit列を取り出す際には必要だが、現在は先頭256bitを取り出しているため不要
    memcpy(digest, full_state, digest_len); //digest_len =16なら、start = 48なので、full_state[16]~[63]を取り出す[0]が先頭、[MATRIX_STATE_BYTES]が末尾

    return 1;
}

//Ω(h) = trunc_n(P(h) + h)
int matrix_output_transform_digest(uint8_t *digest, size_t digest_len, const state_t *h, int rounds, const state_t *MDS, const affine16_t *AFF){
    state_t state_final;
    uint8_t full_state[MATRIX_STATE_BYTES_MAX];

    if(digest == NULL || h == NULL || MDS == NULL){
        return 0;
    }

    if(digest_len > state_bytes){
        return 0;
    }

    state_init(&state_final);

    //state_final = P(h) + h
    matrix_output_transform(&state_final, h, rounds, MDS, AFF);

    //tを512bit列に変換
    matrix_state_to_bytes_512(full_state, &state_final);

    int ok = matrix_trunc_tail_bytes(digest, digest_len, full_state);

    state_clear(&state_final);

    return ok;
}

//blockからbitpos位置よりnbits分をMSBファーストで読み出す(fp16_to_bytes_512のビットパックの逆演算)
static uint32_t load_bits_be(const uint8_t *block, size_t bitpos, unsigned nbits){
    uint32_t v = 0;
    for(unsigned i = 0; i < nbits; i++){
        size_t bp = bitpos + i;
        int bit = (block[bp >> 3] >> (7 - (bp & 7))) & 1;
        v = (v << 1) | (uint32_t)bit;
    }
    return v;
}

//メッセージブロック(block_bytes byte)を4*4行列に変換
void matrix_bytes_to_state(state_t *S, const uint8_t block[MATRIX_BLOCK_BYTES_MAX]){
    size_t bitpos = 0;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            uint32_t v = load_bits_be(block, bitpos, load_bits); //load_bits < FP_BITS なので mod は実質不要だが安全のため通す
            fp_set_ui(&S->m[i][j], v);

            bitpos += load_bits;
        }
    }
}

//16進数を出力
void print_bytes_hex(const uint8_t *buf, size_t len){
    for (size_t i = 0; i < len; i++) {
        printf("%02x", buf[i]);

        // 4 byteごとに空白を入れる
        if ((i + 1) % 4 == 0) {
            printf(" ");
        }

        // 16 byteごとに改行する
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }

    if (len % 16 != 0) {
        printf("\n");
    }
}

//1ブロック分のハッシュ処理をまとめる関数
int matrix_hash_one_block(uint8_t *digest, size_t digest_len, const uint8_t block[MATRIX_BLOCK_BYTES_MAX], int rounds, const state_t *MDS, const affine16_t *AFF){
    state_t h;
    state_t m;
    state_t h_new;

    if(digest == NULL || block == NULL || MDS == NULL){
        return 0;
    }

    state_init(&h);
    state_init(&m);
    state_init(&h_new);

    //初期値 h = IV
    state_set_zero(&h);

    // printf("h0 :\n");
    // state_print(&h);

    //64byteのブロック→状態行列 m
    matrix_bytes_to_state(&m, block);

    // printf("m :\n");
    // state_print(&m);

    //h_new = 圧縮関数(h,m)
    matrix_compression(&h_new, &h, &m, rounds, MDS, AFF);

    int ok = matrix_output_transform_digest(digest, digest_len, &h_new, rounds, MDS, AFF);

    state_clear(&h);
    state_clear(&m);
    state_clear(&h_new);

    return ok;
}

size_t matrix_padded_length(size_t msg_len){
    size_t len = msg_len + 1 + 8;  //1:0x80、 8:メッセージ長を入れる領域
    size_t rem = len % block_bytes;

    if(rem == 0){
        return len;
    }
    return len + (block_bytes - rem);
}

//メッセージ長の最大値を超えないかの確認
int matrix_check_msg_len(size_t msg_len){
    if (msg_len > UINT64_MAX / 8) {
        return 0; // 長すぎる
    }
    return 1;
}

void matrix_pad(uint8_t *out, size_t padded_len, const uint8_t *msg, size_t msg_len){
    //まず全体を0で埋める
    memset(out, 0, padded_len);

    //元のメッセージを先頭にコピー
    memcpy(out, msg, msg_len);

    //メッセージの末尾の隣に0x80を追加
    out[msg_len] = 0x80;

    //最後の8byteにメッセージ長をbit単位で入れる
    uint64_t bit_len = (uint64_t)msg_len * 8;

    out[padded_len -8] = (uint8_t)(bit_len >> 56);
    out[padded_len -7] = (uint8_t)(bit_len >> 48);
    out[padded_len -6] = (uint8_t)(bit_len >> 40);
    out[padded_len -5] = (uint8_t)(bit_len >> 32);
    out[padded_len -4] = (uint8_t)(bit_len >> 24);
    out[padded_len -3] = (uint8_t)(bit_len >> 16);
    out[padded_len -2] = (uint8_t)(bit_len >> 8);
    out[padded_len -1] = (uint8_t)(bit_len);
}

int matrix_hash(uint8_t *digest, size_t digest_len, const uint8_t *msg, size_t msg_len, int rounds, const state_t *MDS, const affine16_t *AFF){
    
    size_t n_bits = digest_len * 8;
    if(!field_select_for_output(n_bits)) return 0;
    //ここでエラーの原因になりそうなことをチェックしておく
    if(digest == NULL || msg == NULL || MDS == NULL) return 0;
    if(digest_len > state_bytes) return 0;
    if(!matrix_check_msg_len(msg_len)) return 0;
    printf("p_mer : %u\n",P_MERSENNE);


    //パディングを入れた後のバイト長を求める（実際には長さを確保しただけでまだ入れてない）
    size_t padded_len = matrix_padded_length(msg_len);

    uint8_t *padded = malloc(padded_len); //メモリ確保
    if(padded == NULL) return 0;

    //パディングを入れる
    matrix_pad(padded, padded_len, msg, msg_len);

    state_t h;
    state_t m;
    state_t h_new;

    state_init(&h);
    state_init(&m);
    state_init(&h_new);

    //状態行列の初期値を設定 h = 0x00000100
    state_set_zero(&h);
    fp_set_ui(&h.m[3][3], 256);

    // printf("h0 :\n");
    // state_print(&h);

    size_t num_blocks = padded_len / block_bytes;  //もしpadded_len = 96なら、48byteのブロックが96/48=2つある=num_blocks

    for(size_t b = 0; b < num_blocks; b++){
        const uint8_t *block = padded + b * block_bytes; //&padded[b * MATRIX_BLOCK_BYTES]と同義

        //1ブロック分を行列に変換（blockの開始位置から64byteだけ読み取って使う）
        matrix_bytes_to_state(&m, block);

        //1ブロック分をラウンド処理にかける
        matrix_compression(&h_new, &h, &m, rounds, MDS, AFF);

        //状態行列を更新
        state_copy(&h, &h_new);
    }

    //全ブロックのラウンド処理が終わって出てきた状態行列を使って、最終的な出力を求める
    int ok = matrix_output_transform_digest(digest, digest_len, &h, rounds, MDS, AFF);

    state_clear(&h);
    state_clear(&m);
    state_clear(&h_new);

    free(padded);

    return ok;
}

/* ============================================================
 *  matrix_hash 内検算用ヘルパー群
 *  matrix_hash の処理中に、実データを使って各ステップの正しさを
 *  検証する。各項目の PASS 回数をグローバル配列に集計し、
 *  複数回呼び出した後に main 側でまとめて表示できる。
 * ============================================================ */
 
/* 前方宣言(16_header.h に無いもの) */
void matrix_bytes_to_state(state_t *S, const uint8_t block[MATRIX_BLOCK_BYTES_MAX]);
 
/* 検算項目の番号 */
enum {
    DBG_PADDING_LEN = 0,
    DBG_PADDING_PREFIX,
    DBG_BYTES_TO_STATE,
    DBG_COMPRESSION,
    DBG_PERMUTATION_P,
    DBG_INVERSE,
    DBG_MDS,
    DBG_OUTPUT_PACK,
    DBG_DIGEST_NONZERO,
    DBG_NUM_ITEMS
};
 
/* 各項目の PASS 回数・総試行回数。main から参照・リセットできるよう非 static */
long g_dbg_pass_count[DBG_NUM_ITEMS] = {0};
long g_dbg_total_count[DBG_NUM_ITEMS] = {0};
 
/* 項目名(表示用) */
const char *g_dbg_item_name[DBG_NUM_ITEMS] = {
    "padding length block-aligned",
    "message preserved in prefix",
    "bytes->state deterministic",
    "compression f=P(h+m)+Q(m)+h",
    "permutation P sensitive",
    "inverse a*inv(a)=1",
    "MDS circ(1,1,2,8) minors",
    "output packing no const bit",
    "digest generated non-zero",
};
 
/* 1項目の検算結果を記録(表示せず集計のみ) */
static void dbg_record(int item, int ok){
    g_dbg_total_count[item]++;
    if(ok) g_dbg_pass_count[item]++;
}
 
/* 集計のリセット(main から呼ぶ) */
void matrix_selftest_reset(void){
    for(int i=0;i<DBG_NUM_ITEMS;i++){ g_dbg_pass_count[i]=0; g_dbg_total_count[i]=0; }
}
 
/* 集計結果の表示(main から呼ぶ) */
void matrix_selftest_report(void){
    printf("\n==== matrix_hash self-test summary ====\n");
    printf("  %-32s %s\n", "item", "pass / total");
    printf("  ----------------------------------------------\n");
    int all_ok = 1;
    for(int i=0;i<DBG_NUM_ITEMS;i++){
        printf("  %-32s %ld / %ld%s\n", g_dbg_item_name[i],
               g_dbg_pass_count[i], g_dbg_total_count[i],
               (g_dbg_pass_count[i]==g_dbg_total_count[i]) ? "" : "  <-- FAIL");
        if(g_dbg_pass_count[i]!=g_dbg_total_count[i]) all_ok = 0;
    }
    printf("  ----------------------------------------------\n");
    printf("  RESULT: %s\n", all_ok ? "ALL PASS" : "SOME FAILED");
    printf("=======================================\n\n");
}
 
/* (1) パディング検算 */
static void dbg_check_padding(const uint8_t *padded, size_t padded_len,
                              const uint8_t *msg, size_t msg_len){
    int len_ok = (padded_len % block_bytes== 0) && (padded_len >= msg_len);
    dbg_record(DBG_PADDING_LEN, len_ok);
    int prefix_ok = (memcmp(padded, msg, msg_len) == 0);
    dbg_record(DBG_PADDING_PREFIX, prefix_ok);
}
 
/* (2) bytes->state 変換検算 */
static void dbg_check_bytes_to_state(const uint8_t *block){
    state_t a, b; state_init(&a); state_init(&b);
    matrix_bytes_to_state(&a, block);
    matrix_bytes_to_state(&b, block);
    uint8_t ba[64], bb[64];
    matrix_state_to_bytes_512(ba, &a);
    matrix_state_to_bytes_512(bb, &b);
    dbg_record(DBG_BYTES_TO_STATE, memcmp(ba,bb,64)==0);
    state_clear(&a); state_clear(&b);
}
 
/* (3) 圧縮関数検算 + P の単射性 */
static void dbg_check_compression(const state_t *h_new, const state_t *h,
                                  const state_t *m, int rounds,
                                  const state_t *MDS, const affine16_t *AFF){
    state_t hm, p_out, q_out, recomputed;
    state_init(&hm); state_init(&p_out); state_init(&q_out); state_init(&recomputed);
    state_add(&hm, h, m);
    matrix_permutation_P(&p_out, &hm, rounds, MDS, AFF);
    matrix_permutation_Q(&q_out, m, rounds, MDS, AFF);
    state_add3(&recomputed, &p_out, &q_out, h);
    uint8_t b1[64], b2[64];
    matrix_state_to_bytes_512(b1, h_new);
    matrix_state_to_bytes_512(b2, &recomputed);
    dbg_record(DBG_COMPRESSION, memcmp(b1,b2,64)==0);
 
    state_t hm2, p2;
    state_init(&hm2); state_init(&p2);
    state_copy(&hm2, &hm);
    fp_t one; fp_set_ui(&one, 1);
    fp_add(&hm2.m[0][0], &hm2.m[0][0], &one);
    matrix_permutation_P(&p2, &hm2, rounds, MDS, AFF);
    uint8_t bp1[64], bp2[64];
    matrix_state_to_bytes_512(bp1, &p_out);
    matrix_state_to_bytes_512(bp2, &p2);
    dbg_record(DBG_PERMUTATION_P, memcmp(bp1,bp2,64)!=0);
 
    state_clear(&hm); state_clear(&p_out); state_clear(&q_out); state_clear(&recomputed);
    state_clear(&hm2); state_clear(&p2);
}
 
/* (4) 逆元検算 */
static void dbg_check_inverse(void){
    int ok = 1;
    for(int t=0;t<100;t++){
        fp16_t a, ia, prod, b, chk;
        fp16_random(&a);
        fp16_inv(&ia, &a);
        fp16_mul(&prod, &a, &ia);
        fp16_random(&b);
        fp16_mul(&chk, &b, &prod);
        if(!fp16_is_equal(&chk, &b)) ok = 0;
    }
    dbg_record(DBG_INVERSE, ok);
}
 
/* (5) MDS 検算 */
static void dbg_check_mds(void){
    long p=(1L<<31)-1, row[4]={1,1,2,8}, M[4][4];
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) M[i][j]=row[(j-i+4)%4];
    int mds=1;
    for(int i=0;i<4;i++) for(int j=0;j<4;j++){ if(M[i][j]%p==0) mds=0; }
    int c2[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    for(int a=0;a<6;a++) for(int b=0;b<6;b++){
        long d=((M[c2[a][0]][c2[b][0]]*M[c2[a][1]][c2[b][1]]
                -M[c2[a][0]][c2[b][1]]*M[c2[a][1]][c2[b][0]])%p+p)%p;
        if(d==0) mds=0;
    }
    int c3[4][3]={{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
    int pm[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}}, sg[6]={1,-1,-1,1,1,-1};
    for(int a=0;a<4;a++) for(int b=0;b<4;b++){
        long det=0;
        for(int q=0;q<6;q++){ long tm=sg[q];
            for(int x=0;x<3;x++) tm=(tm*(M[c3[a][x]][c3[b][pm[q][x]]]%p))%p;
            det=(det+tm)%p; }
        det=((det%p)+p)%p; if(det==0) mds=0;
    }
    { long det=0; int s4[4]={1,-1,1,-1};
      for(int c0=0;c0<4;c0++){ int sc[3],t=0; for(int j=0;j<4;j++) if(j!=c0) sc[t++]=j;
        int sr[3]={1,2,3}; long m3=0;
        for(int q=0;q<6;q++){ long tm=sg[q];
            for(int x=0;x<3;x++) tm=(tm*(M[sr[x]][sc[pm[q][x]]]%p))%p; m3=(m3+tm)%p; }
        det=(det+s4[c0]*(M[0][c0]%p)*(((m3%p)+p)%p))%p; }
      det=((det%p)+p)%p; if(det==0) mds=0; }
    dbg_record(DBG_MDS, mds);
}
 
/* (6) 出力詰め検算 */
static void dbg_check_output_packing(void){
    uint8_t oracc[64]; memset(oracc,0,64);
    for(int t=0;t<2000;t++){
        state_t s; state_init(&s); state_random(&s);
        uint8_t bb[64]; matrix_state_to_bytes_512(bb,&s);
        for(int k=0;k<64;k++) oracc[k]|=bb[k];
        state_clear(&s);
    }
    int head_const=0;
    for(int bit=0;bit<256;bit++) if(!((oracc[bit>>3]>>(7-(bit&7)))&1)) head_const++;
    dbg_record(DBG_OUTPUT_PACK, head_const==0);
}

int matrix_hash_test(uint8_t *digest, size_t digest_len, const uint8_t *msg, size_t msg_len, int rounds, const state_t *MDS, const affine16_t *AFF){

    //ここでエラーの原因になりそうなことをチェックしておく
    if(digest == NULL || msg == NULL || MDS == NULL){
        return 0;
    }
    if(digest_len > state_bytes){
        return 0;
    }
    if(!matrix_check_msg_len(msg_len)){
        return 0;
    }

    /* ===================== 埋め込みテスト用カウンタ ===================== */
    int dbg_pass = 0, dbg_fail = 0;
    #define DBG_CHECK(name, cond, ...) do {                          \
        int _ok = (cond);                                           \
        printf("  [%s] %-32s ", _ok ? "PASS" : "FAIL", (name));     \
        printf(__VA_ARGS__);                                        \
        printf("\n");                                               \
        if(_ok) dbg_pass++; else dbg_fail++;                        \
    } while(0)

    printf("================ matrix_hash internal self-test ================\n");

    //パディングを入れた後のバイト長を求める（実際には長さを確保しただけでまだ入れてない）
    size_t padded_len = matrix_padded_length(msg_len);

    uint8_t *padded = malloc(padded_len); //メモリ確保
    if(padded == NULL){
        return 0;
    }

    //パディングを入れる
    matrix_pad(padded, padded_len, msg, msg_len);

    /* --- TEST 1: パディング --- */
    DBG_CHECK("padding length multiple of block",
              (padded_len % block_bytes) == 0,
              "msg_len=%zu -> padded_len=%zu (%zu blocks)",
              msg_len, padded_len, padded_len / block_bytes);
    {
        /* 先頭が元メッセージと一致しているか */
        int head_ok = (memcmp(padded, msg, msg_len) == 0);
        DBG_CHECK("padding preserves message", head_ok,
                  "first %zu bytes equal original msg", msg_len);
    }

    state_t h;
    state_t m;
    state_t h_new;

    state_init(&h);
    state_init(&m);
    state_init(&h_new);

    //状態行列の初期値を設定 h = 0x00000100
    state_set_zero(&h);
    fp_set_ui(&h.m[3][3], 256);

    /* --- TEST 2: 逆元 (SubBytes の核) --- */
    {
        int inv_ok = 1, cross_ok = 1;
        fp16_t a, ia, ia2, b, prod, chk;
        for(int t = 0; t < 200; t++){
            fp16_random(&a);
            fp16_inv(&ia, &a);
            fp16_mul(&prod, &a, &ia);          /* prod は乗法単位元のはず */
            fp16_random(&b);
            fp16_mul(&chk, &b, &prod);         /* b * 1 == b を確認 */
            if(!fp16_is_equal(&chk, &b)) inv_ok = 0;
            fp16_inv_slow(&ia2, &a);           /* 別実装と一致 */
            if(!fp16_is_equal(&ia, &ia2)) cross_ok = 0;
        }
        DBG_CHECK("inverse a*inv(a)==1", inv_ok, "200 random fp16 elements");
        DBG_CHECK("inverse impls agree", cross_ok, "fp16_inv == fp16_inv_slow");
    }

    /* --- TEST 3: MDS (MixBytes) circ(1,1,2,8) が branch number 5 --- */
    {
        long p = (1L<<31)-1;
        long row[4] = {1,1,2,8}, M[4][4];
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) M[i][j]=row[(j-i+4)%4];
        int mds = 1, checked = 0;
        int c2[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
        for(int i=0;i<4;i++) for(int j=0;j<4;j++){ checked++; if(M[i][j]%p==0) mds=0; }
        for(int a=0;a<6;a++) for(int b=0;b<6;b++){
            long d=((M[c2[a][0]][c2[b][0]]%p)*(M[c2[a][1]][c2[b][1]]%p)
                  -(M[c2[a][0]][c2[b][1]]%p)*(M[c2[a][1]][c2[b][0]]%p))%p;
            d=((d%p)+p)%p; checked++; if(d==0) mds=0;
        }
        int c3[4][3]={{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
        int pm[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
        int sg[6]={1,-1,-1,1,1,-1};
        for(int a=0;a<4;a++) for(int b=0;b<4;b++){
            long det=0;
            for(int pp=0;pp<6;pp++){ long term=sg[pp];
                for(int x=0;x<3;x++) term=(term*(M[c3[a][x]][c3[b][pm[pp][x]]]%p))%p;
                det=(det+term)%p; }
            det=((det%p)+p)%p; checked++; if(det==0) mds=0;
        }
        { long det=0; int sign4[4]={1,-1,1,-1};
          for(int c0=0;c0<4;c0++){ int sc[3],t=0; for(int j=0;j<4;j++) if(j!=c0) sc[t++]=j;
            int sr[3]={1,2,3}; long m3=0;
            for(int pp=0;pp<6;pp++){ long term=sg[pp];
                for(int x=0;x<3;x++) term=(term*(M[sr[x]][sc[pm[pp][x]]]%p))%p;
                m3=(m3+term)%p; }
            det=(det+sign4[c0]*(M[0][c0]%p)*(((m3%p)+p)%p))%p; }
          det=((det%p)+p)%p; checked++; if(det==0) mds=0; }
        DBG_CHECK("MDS circ(1,1,2,8)", mds, "all %d minors nonzero (branch number 5)", checked);
    }

    /* --- TEST 4: 置換 P,Q の決定性と P!=Q --- */
    {
        state_t s, p1, p2, q1;
        state_init(&s); state_init(&p1); state_init(&p2); state_init(&q1);
        state_random(&s);
        matrix_permutation_P(&p1, &s, rounds, MDS, AFF);
        matrix_permutation_P(&p2, &s, rounds, MDS, AFF);
        matrix_permutation_Q(&q1, &s, rounds, MDS, AFF);
        uint8_t bp1[64], bp2[64], bq1[64];
        matrix_state_to_bytes_512(bp1,&p1);
        matrix_state_to_bytes_512(bp2,&p2);
        matrix_state_to_bytes_512(bq1,&q1);
        DBG_CHECK("permutation P deterministic", memcmp(bp1,bp2,64)==0, "P(s)==P(s)");
        DBG_CHECK("permutations P != Q", memcmp(bp1,bq1,64)!=0, "P(s) differs from Q(s)");
        state_clear(&s); state_clear(&p1); state_clear(&p2); state_clear(&q1);
    }

    size_t num_blocks = padded_len / block_bytes;

    /* --- TEST 5: 圧縮関数 f(h,m)=P(h+m)+Q(m)+h を1ブロック目で検算 --- */
    {
        state_t m0, p, q, hm, expect, got;
        state_init(&m0); state_init(&p); state_init(&q);
        state_init(&hm); state_init(&expect); state_init(&got);
        matrix_bytes_to_state(&m0, padded);                 /* 1ブロック目 */
        state_add(&hm, &h, &m0);                            /* h+m */
        matrix_permutation_P(&p, &hm, rounds, MDS, AFF);    /* P(h+m) */
        matrix_permutation_Q(&q, &m0, rounds, MDS, AFF);    /* Q(m) */
        state_add3(&expect, &p, &q, &h);                    /* P(h+m)+Q(m)+h */
        matrix_compression(&got, &h, &m0, rounds, MDS, AFF);
        uint8_t be[64], bg[64];
        matrix_state_to_bytes_512(be,&expect);
        matrix_state_to_bytes_512(bg,&got);
        DBG_CHECK("compression f=P(h+m)+Q(m)+h", memcmp(be,bg,64)==0,
                  "manual recomputation matches matrix_compression");
        state_clear(&m0); state_clear(&p); state_clear(&q);
        state_clear(&hm); state_clear(&expect); state_clear(&got);
    }

    for(size_t b = 0; b < num_blocks; b++){
        const uint8_t *block = padded + b * block_bytes;
        matrix_bytes_to_state(&m, block);
        matrix_compression(&h_new, &h, &m, rounds, MDS, AFF);
        state_copy(&h, &h_new);
    }

    //全ブロックのラウンド処理が終わって出てきた状態行列を使って、最終的な出力を求める
    int ok = matrix_output_transform_digest(digest, digest_len, &h, rounds, MDS, AFF);

    /* --- TEST 6: 出力変換 31ビット詰めにダイジェスト範囲の定数ビットが無い --- */
    {
        uint8_t oracc[64]; memset(oracc,0,64);
        for(int t=0;t<2000;t++){
            state_t rs; state_init(&rs); state_random(&rs);
            uint8_t rb[64]; matrix_state_to_bytes_512(rb,&rs);
            for(int k=0;k<64;k++) oracc[k]|=rb[k];
            state_clear(&rs);
        }
        int head_const=0;
        for(int bit=0; bit<(int)(digest_len*8); bit++)
            if(!((oracc[bit>>3]>>(7-(bit&7)))&1)) head_const++;
        DBG_CHECK("output packing no constant bit", head_const==0,
                  "%d constant bits in first %zu output bits", head_const, digest_len*8);
    }

    /* --- TEST 7: ハッシュ全体の決定性 --- */
    {
        uint8_t d2[MATRIX_STATE_BYTES_MAX];
        /* 同じ手順を再実行(検算用に独立計算) */
        state_t hh, mm, hn; state_init(&hh); state_init(&mm); state_init(&hn);
        state_set_zero(&hh); fp_set_ui(&hh.m[3][3],256);
        for(size_t b=0;b<num_blocks;b++){
            matrix_bytes_to_state(&mm, padded + b * block_bytes);
            matrix_compression(&hn,&hh,&mm,rounds,MDS,AFF);
            state_copy(&hh,&hn);
        }
        matrix_output_transform_digest(d2, digest_len, &hh, rounds, MDS, AFF);
        DBG_CHECK("hash deterministic", memcmp(digest,d2,digest_len)==0,
                  "recomputed digest matches");
        state_clear(&hh); state_clear(&mm); state_clear(&hn);
    }

    printf("---------------------------------------------------------------\n");
    printf("  self-test result: %d passed, %d failed\n", dbg_pass, dbg_fail);
    printf("================================================================\n");
    #undef DBG_CHECK

    state_clear(&h);
    state_clear(&m);
    state_clear(&h_new);

    free(padded);

    return ok;
}


// void test_matrix_hash_one_block(void)
// {
//     uint8_t block[MATRIX_STATE_BYTES] = {0};
//     uint8_t digest[32];

//     /*
//       分かりやすい入力を入れる
//       block = 0,1,2,...,63
//     */
//     for (int i = 0; i < MATRIX_STATE_BYTES; i++) {
//         block[i] = (uint8_t)i;
//     }

//     int ok = matrix_hash_one_block(
//         digest,
//         32,              // 256 bit digest
//         block,
//         MATRIX_ROUNDS,
//         &MDS
//     );

//     if (!ok) {
//         printf("matrix_hash_one_block failed\n");
//         return;
//     }

//     printf("digest = ");
//     matrix_print_digest_hex(digest, 32);
// }

















void test_state_bytes_roundtrip(void){
    state_t S, T;
    uint8_t buf[64];

    state_init(&S);
    state_init(&T);

    state_random(&S);

    printf("S:\n");
    state_print(&S);

    matrix_state_to_bytes_512(buf, &S);

    printf("buf :\n");
    print_bytes_hex(buf, 64);

    matrix_bytes_to_state(&T, buf);

    printf("T:\n");
    state_print(&T);

    if (state_equal(&S, &T)) {
        printf("state -> bytes -> state: OK\n");
    } else {
        printf("state -> bytes -> state: NG\n");
        printf("S:\n");
        state_print(&S);
        printf("T:\n");
        state_print(&T);
    }

    state_clear(&S);
    state_clear(&T);
}



// static void matrix_subbytes_with_inv(state_t *S, fp16_inv_func inv){
//     fp16_t a, out;
//     state_to_fp16(&a, S);
//     //もしaが0なら，逆元計算できないので先に見ておく
//     if (fp4_is_zero(&a.x0) && fp4_is_zero(&a.x1) &&
//         fp4_is_zero(&a.x2) && fp4_is_zero(&a.x3)) {
//         state_set_zero(S);
//         return;
//     }
//     inv(&out, &a);
//     state_from_fp16(S, &out);
// }

// void matrix_subbytes_inv(state_t *S){
//     matrix_subbytes_with_inv(S, fp16_inv);
// }

// void matrix_subbytes_inv_new(state_t *S){
//     matrix_subbytes_with_inv(S, fp16_inv_new);
// }

// void matrix_subbytes_inv_karatsuba(state_t *S){
//     matrix_subbytes_with_inv(S, fp16_inv_karatsuba);
// }

// void matrix_add_round_constant_p(state_t *S, uint8_t round){
//     static const uint8_t base[4] = {0x00, 0x10, 0x20, 0x30};
//     fp_t c;
//     for (int j = 0; j < 4; j++) {
//         c.x0 = (uint32_t)(base[j] ^ round);
//         fp_add(&S->m[0][j], &S->m[0][j], &c);
//     }
// }

// void matrix_add_round_constant_q(state_t *S, uint8_t round){
//     static const uint8_t base[4] = {0xff, 0xef, 0xdf, 0xcf};
//     fp_t c;
//     for (int i = 0; i < 3; i++) {
//         for (int j = 0; j < 4; j++) {
//             c.x0 = 0xff;
//             fp_add(&S->m[i][j], &S->m[i][j], &c);
//         }
//     }
//     for (int j = 0; j < 4; j++) {
//         c.x0 = (uint32_t)(base[j] ^ round);
//         fp_add(&S->m[3][j], &S->m[3][j], &c);
//     }
// }

// void matrix_shiftbytes(state_t *S){
//     fp_t tmp[4];
//     for (int i = 1; i < 4; i++) {
//         for (int j = 0; j < 4; j++) tmp[j] = S->m[i][j];
//         for (int j = 0; j < 4; j++) {
//             S->m[i][j] = tmp[(j + i) % 4];
//         }
//     }
// }



// static fp16_inv_func matrix_inv_from_mode(int inv_mode){
//     switch (inv_mode) {
//         case 1: return fp16_inv_new;
//         case 2: return fp16_inv_karatsuba;
//         default: return fp16_inv;
//     }
// }

// void matrix_round_p(state_t *S, uint8_t round, int inv_mode){
//     matrix_add_round_constant_p(S, round);
//     matrix_subbytes_with_inv(S, matrix_inv_from_mode(inv_mode));
//     matrix_shiftbytes(S);
//     matrix_mixbytes(S);
// }

// void matrix_round_q(state_t *S, uint8_t round, int inv_mode){
//     matrix_add_round_constant_q(S, round);
//     matrix_subbytes_with_inv(S, matrix_inv_from_mode(inv_mode));
//     matrix_shiftbytes(S);
//     matrix_mixbytes(S);
// }



// static void matrix_subbytes_with_inv(state_t *S, fp16_inv_func inv){
//     fp16_t a, out;
//     state_to_fp16(&a, S);
//     //もしaが0なら，逆元計算できないので先に見ておく
//     if (fp4_is_zero(&a.x0) && fp4_is_zero(&a.x1) &&
//         fp4_is_zero(&a.x2) && fp4_is_zero(&a.x3)) {
//         state_set_zero(S);
//         return;
//     }
//     inv(&out, &a);
//     state_from_fp16(S, &out);
// }

// void matrix_subbytes_inv(state_t *S){
//     matrix_subbytes_with_inv(S, fp16_inv);
// }

// void matrix_subbytes_inv_new(state_t *S){
//     matrix_subbytes_with_inv(S, fp16_inv_new);
// }

// void matrix_subbytes_inv_karatsuba(state_t *S){
//     matrix_subbytes_with_inv(S, fp16_inv_karatsuba);
// }

// void matrix_add_round_constant_p(state_t *S, uint8_t round){
//     static const uint8_t base[4] = {0x00, 0x10, 0x20, 0x30};
//     fp_t c;
//     for (int j = 0; j < 4; j++) {
//         c.x0 = (uint32_t)(base[j] ^ round);
//         fp_add(&S->m[0][j], &S->m[0][j], &c);
//     }
// }

// void matrix_add_round_constant_q(state_t *S, uint8_t round){
//     static const uint8_t base[4] = {0xff, 0xef, 0xdf, 0xcf};
//     fp_t c;
//     for (int i = 0; i < 3; i++) {
//         for (int j = 0; j < 4; j++) {
//             c.x0 = 0xff;
//             fp_add(&S->m[i][j], &S->m[i][j], &c);
//         }
//     }
//     for (int j = 0; j < 4; j++) {
//         c.x0 = (uint32_t)(base[j] ^ round);
//         fp_add(&S->m[3][j], &S->m[3][j], &c);
//     }
// }

// void matrix_shiftbytes(state_t *S){
//     fp_t tmp[4];
//     for (int i = 1; i < 4; i++) {
//         for (int j = 0; j < 4; j++) tmp[j] = S->m[i][j];
//         for (int j = 0; j < 4; j++) {
//             S->m[i][j] = tmp[(j + i) % 4];
//         }
//     }
// }



// static fp16_inv_func matrix_inv_from_mode(int inv_mode){
//     switch (inv_mode) {
//         case 1: return fp16_inv_new;
//         case 2: return fp16_inv_karatsuba;
//         default: return fp16_inv;
//     }
// }

// void matrix_round_p(state_t *S, uint8_t round, int inv_mode){
//     matrix_add_round_constant_p(S, round);
//     matrix_subbytes_with_inv(S, matrix_inv_from_mode(inv_mode));
//     matrix_shiftbytes(S);
//     matrix_mixbytes(S);
// }

// void matrix_round_q(state_t *S, uint8_t round, int inv_mode){
//     matrix_add_round_constant_q(S, round);
//     matrix_subbytes_with_inv(S, matrix_inv_from_mode(inv_mode));
//     matrix_shiftbytes(S);
//     matrix_mixbytes(S);
// }

// #include "16_header.h"


// typedef void (*fp16_inv_func)(fp16_t *S, const fp16_t *X);

// void state_init(state_t *S){
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             fp_init(&S->m[i][j]);
//         }
//     }
// }

// void state_set_zero(state_t *S){
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             S->m[i][j].x0 = 0;
//         }
//     }
// }

// void state_random(state_t *S){
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             fp_random(&S->m[i][j]);
//         }
//     }
// }

// int state_is_zero(const state_t *S){
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             if (!fp_is_zero(&S->m[i][j])) return 0;
//         }
//     }
//     return 1;
// }

// void state_from_fp16(state_t *S, const fp16_t *X){
//     const fp4_t *rows[4] = {&X->x0, &X->x1, &X->x2, &X->x3};
//     for (int i = 0; i < 4; i++) {
//         S->m[i][0] = rows[i]->x0;
//         S->m[i][1] = rows[i]->x1;
//         S->m[i][2] = rows[i]->x2;
//         S->m[i][3] = rows[i]->x3;
//     }
// }

// void state_to_fp16(fp16_t *S, const state_t *X){
//     fp4_t *rows[4] = {&S->x0, &S->x1, &S->x2, &S->x3};
//     for (int i = 0; i < 4; i++) {
//         rows[i]->x0 = X->m[i][0];
//         rows[i]->x1 = X->m[i][1];
//         rows[i]->x2 = X->m[i][2];
//         rows[i]->x3 = X->m[i][3];
//     }
// }


// void state_clear(state_t *S){
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             fp_clear(&S->m[i][j]);
//         }
//     }
// }

// void state_copy(state_t *dst, const state_t *src){
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             fp_set(&dst->m[i][j], &src->m[i][j]);
//         }
//     }
// }

// void state_add(state_t *Z, const state_t *X, const state_t *Y){
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             fp_add(&Z->m[i][j], &X->m[i][j], &Y->m[i][j]);
//         }
//     }
// }

// void state_sub(state_t *Z, const state_t *X, const state_t *Y){
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             fp_sub(&Z->m[i][j], &X->m[i][j], &Y->m[i][j]);
//         }
//     }
// }

// void state_add3(state_t *Z, const state_t *A, const state_t *B, const state_t *C){
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             fp_t tmp;
//             fp_add(&tmp, &A->m[i][j], &B->m[i][j]);
//             fp_add(&Z->m[i][j], &tmp, &C->m[i][j]);
//         }
//     }
// }

// int  state_equal(const state_t *A, const state_t *B){
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             if (!fp_is_equal(&A->m[i][j], &B->m[i][j])) return 0;
//         }
//     }
//     return 1;
// }

// void state_print(const state_t *S){

//     for (int i = 0; i < 4; i++) {
//         printf("  [");
//         for (int j = 0; j < 4; j++) {
//             fp_printf(&S->m[i][j]);
//             if (j < 3) printf(", ");
//         }
//         printf("]\n");
//     }
// }

// //MDS行列×状態行列の各列
// void state_mix_column(fp_t y[4], const fp_t x[4], const state_t *M){
//     fp_t acc;
//     fp_t term;

//     fp_init(&acc);
//     fp_init(&term);

//     for (int i = 0; i < 4; i++) {
//         fp_set_zero(&acc);

//         for (int j = 0; j < 4; j++) {
//             // term = M[i][j] * x[j]
//             fp_mul(&term, &M->m[i][j], &x[j]);

//             // acc += term
//             fp_add(&acc, &acc, &term);
//         }

//         fp_set(&y[i], &acc);
//     }

//     fp_clear(&acc);
//     fp_clear(&term);
// }

// void matrix_mixbytes(state_t *S_new, const state_t *S, const state_t *M){
//     fp_t x[4];
//     fp_t y[4];

//     for (int i = 0; i < 4; i++) {
//         fp_init(&x[i]);
//         fp_init(&y[i]);
//     }

//     for (int col = 0; col < 4; col++) {
//         // col列を取り出す
//         for (int row = 0; row < 4; row++) {
//             fp_set(&x[row], &S->m[row][col]);
//         }

//         // y = M x
//         state_mix_column(y, x, M);

//         // 結果を書き戻す
//         for (int row = 0; row < 4; row++) {
//             fp_set(&S_new->m[row][col], &y[row]);
//         }
//     }

//     for (int i = 0; i < 4; i++) {
//         fp_clear(&x[i]);
//         fp_clear(&y[i]);
//     }
// }

// //shift[4]=[0,1,2,3]なら、0行目はシフトなし、1行目は左に1ビットシフト、2行目は左に2ビットシフト、3行目は左に3ビットシフトさせる関数
// void matrix_shiftbytes(state_t *S_new, const state_t *S, const int shift[4]){
//     for (int row = 0; row < 4; row++){
//         int s = shift[row] % 4;

//         for (int col = 0; col < 4; col++){
//             fp_set(&S_new->m[row][col], &S->m[row][(col + s) % 4]);
//         }
//     }
// }

// void matrix_shiftbytes_P(state_t *S_new, const state_t *S){
//     const int shift_P[4] = {0,1,2,3};
//     matrix_shiftbytes(S_new,S,shift_P);
// }

// void matrix_shiftbytes_Q(state_t *S_new, const state_t *S){
//     const int shift_Q[4] = {1,3,0,2};
//     matrix_shiftbytes(S_new,S,shift_Q);
// }

// void affine16_init(affine16_t *AFF){
//     for (int i = 0; i < 16; i++){
//         fp_init(&AFF->b[i]);
//         for (int j = 0; j < 16; j++){
//             fp_init(&AFF->A[i][j]);
//         }
//     }
// }

// void affine16_clear(affine16_t *AFF){
//     for (int i = 0; i < 16; i++){
//         fp_clear(&AFF->b[i]);
//         for (int j = 0; j < 16; j++){
//             fp_clear(&AFF->A[i][j]);
//         }
//     }
// }

// void affine16_print_A(const affine16_t *F){
//     for (int i = 0; i < 16; i++) {
//         for (int j = 0; j < 16; j++) {
//             printf("%u ", F->A[i][j].x0);
//         }
//         printf("\n");
//     }
// }

// void affine16_print_b(const affine16_t *F)
// {
//     for (int i = 0; i < 16; i++) {
//         printf("%u ", F->b[i].x0);
//     }

//     printf("\n");
// }

// //4*4を16*1に変換する関数
// void state_to_vec16(fp_t v[16], const state_t *S){
//     int idx = 0;

//     for (int i = 0; i < 4; i++){
//         for (int j = 0; j < 4; j++){
//             fp_set(&v[idx], &S->m[i][j]);
//             idx++;
//         }
//     }
// }

// //16*1を4*4に変換する関数
// void vec16_to_state(state_t *S, const fp_t v[16]){
//     int idx = 0;

//     for (int i = 0; i < 4; i++){
//         for (int j = 0; j < 4; j++){
//             fp_set(&S->m[i][j], &v[idx]);
//             idx++;
//         }
//     }
// }

// //Aに値をセットする関数
// void affine16_set_A(affine16_t *AFF)
// {
//     for (int i = 0; i < 16; i++) {
//         fp_set_zero(&AFF->b[i]);

//         for (int j = 0; j < 16; j++) {
//             fp_set_zero(&AFF->A[i][j]);
//         }
//     }

//     for (int i = 0; i < 16; i++) {
//         fp_set_ui(&AFF->A[i][i], 1);
//         fp_set_ui(&AFF->A[i][(i + 4) % 16], 1);
//         fp_set_ui(&AFF->A[i][(i + 5) % 16], 1);
//         fp_set_ui(&AFF->A[i][(i + 6) % 16], 1);
//         fp_set_ui(&AFF->A[i][(i + 7) % 16], 1);
//     }
// }

// void affine16_set_b(affine16_t *AFF)
// {
//     for (int i = 0; i < 16; i++) {
//         fp_set_ui(&AFF->b[i], i + 1);
//     }
// }

// //A,bをセットする関数(一旦Aは単位ベクトル、bは零ベクトルでセット)
// void affine16_set(affine16_t *AFF){
//     for (int i = 0; i < 16; i++){
//         fp_set_zero(&AFF->b[i]);
//         for (int j = 0; j < 16; j++){
//         fp_set_zero(&AFF->A[i][j]);
//         }
//     }

//     for (int i = 0; i < 16; i++){
//         fp_set_ui(&AFF->A[i][i], 1);
//     }
// }

// //16*16行列Aが正則行列(逆行列を持つ)かどうかを判定する関数
// //素体F_p上でガウスの消去法による前進消去を行い、全ての列でピボットが立てば正則
// //正則なら1、特異(正則でない)なら0を返す
// int affine16_is_regular(const affine16_t *AFF){
//     fp_t M[16][16];

//     //Aをコピー(消去法で破壊的に変更するため)
//     for (int i = 0; i < 16; i++){
//         for (int j = 0; j < 16; j++){
//             fp_set(&M[i][j], &AFF->A[i][j]);
//         }
//     }

//     fp_t inv, factor, term, tmp;
//     fp_init(&inv);
//     fp_init(&factor);
//     fp_init(&term);
//     fp_init(&tmp);

//     int regular = 1;

//     for (int col = 0; col < 16; col++){
//         //col列で非零成分(ピボット)を持つ行をcol行以降から探す
//         int pivot = -1;
//         for (int row = col; row < 16; row++){
//             if (!fp_is_zero(&M[row][col])){
//                 pivot = row;
//                 break;
//             }
//         }

//         //ピボットが見つからない→この列は消去後すべて0→正則ではない
//         if (pivot == -1){
//             regular = 0;
//             break;
//         }

//         //ピボット行を対角位置(col行)に移動
//         if (pivot != col){
//             for (int j = 0; j < 16; j++){
//                 fp_set(&tmp, &M[col][j]);
//                 fp_set(&M[col][j], &M[pivot][j]);
//                 fp_set(&M[pivot][j], &tmp);
//             }
//         }

//         //ピボットの逆元
//         fp_inv(&inv, &M[col][col]);

//         //col行より下の行のcol列成分を消去
//         for (int row = col + 1; row < 16; row++){
//             if (fp_is_zero(&M[row][col])) continue;

//             //factor = M[row][col] / M[col][col]
//             fp_mul(&factor, &M[row][col], &inv);

//             //row行 = row行 - factor * col行
//             for (int j = col; j < 16; j++){
//                 fp_mul(&term, &factor, &M[col][j]);
//                 fp_sub(&M[row][j], &M[row][j], &term);
//             }
//         }
//     }

//     fp_clear(&inv);
//     fp_clear(&factor);
//     fp_clear(&term);
//     fp_clear(&tmp);

//     return regular;
// }

// //xを16*1に変換した後に使う
// void affine16_apply_vec(fp_t y[16], const fp_t x[16], const affine16_t *AFF){
//     fp_t acc;
//     fp_t term;

//     fp_init(&acc);
//     fp_init(&term);

//     for (int i = 0; i <16; i++){
//         fp_set_zero(&acc);
//         for (int j = 0; j < 16; j++){
//             //term = A[i][j] * x[j]
//             fp_mul(&term, &AFF->A[i][j], &x[j]);

//             //acc = acc + term
//             fp_add(&acc, &acc, &term);
//         }
//         //y[i] = acc + b[i]
//         fp_add(&y[i], &acc, &AFF->b[i]);
//     }
//     fp_clear(&acc);
//     fp_clear(&term);
// }

// void matrix_affine(state_t *S_new, const state_t *S, const affine16_t *AFF){
//     fp_t x[16];
//     fp_t y[16];

//     for (int i = 0; i < 16; i++){
//         fp_init(&x[i]);
//         fp_init(&y[i]);
//     }

//     //逆元計算を終えたS(4*4行列)をx(16*1行列)に変換する
//     state_to_vec16(x, S);

//     //y=A*x+bを行う(yは16*1行列、Aは16*16行列、bは16*1行列)
//     affine16_apply_vec(y, x, AFF);

//     //yを16*1から4*4行列に戻す
//     vec16_to_state(S_new, y);

//     for (int i = 0; i < 16; i++){
//         fp_clear(&x[i]);
//         fp_clear(&y[i]);
//     }
// }



// //xにSをセットし、y=1/x→yをS_newにセット
// void matrix_subbytes(state_t *S_new, const state_t *S, const affine16_t *AFF){
//     fp16_t x,y;
//     state_t S_inv;
//     fp16_init(&x);
//     fp16_init(&y);
//     state_init(&S_inv);

//     state_to_fp16(&x,S);

//     if(fp16_is_zero(&x)){
//         fp16_set_zero(&y);
//     } else{
//         //逆元計算
//         fp16_inv(&y,&x);
//     }

//     state_from_fp16(&S_inv,&y);

//     //affine変換
//     matrix_affine(S_new, &S_inv, AFF);

//     state_clear(&S_inv);
//     fp16_clear(&x);
//     fp16_clear(&y);
// }

// //Pでは1行目にラウンド固定の値を入れる
// void matrix_add_round_constant_P(state_t *S_new, const state_t *S, int r){
//     state_copy(S_new,S);
//     fp_t c;
//     fp_init(&c);

//     fp_set_ui(&c, (0x00 ^ r));
//     fp_add(&S_new->m[0][0], &S_new->m[0][0], &c);

//     fp_set_ui(&c, (0x10 ^ r));
//     fp_add(&S_new->m[0][1], &S_new->m[0][1], &c);

//     fp_set_ui(&c, (0x20 ^ r));
//     fp_add(&S_new->m[0][2], &S_new->m[0][2], &c);

//     fp_set_ui(&c, (0x30 ^ r));
//     fp_add(&S_new->m[0][3], &S_new->m[0][3], &c);

//     fp_clear(&c);
// }

// void matrix_add_round_constant_Q(state_t *S_new, const state_t *S, int r){
//     state_copy(S_new,S);
//     fp_t c;
//     fp_init(&c);

//     for(int i = 0; i < 4; i++){
//         for(int j = 0; j < 4; j++){
//             unsigned int val = 0xff;

//             //4行目にラウンド固定の値を入れる
//             if(i == 3){
//                 if(j == 0) val = 0xff ^ r;
//                 if(j == 1) val = 0xef ^ r;
//                 if(j == 2) val = 0xdf ^ r;
//                 if(j == 3) val = 0xcf ^ r;
//             }

//             //4行目以外はここでffが加算され、4行目はさっき計算した値が加算される
//             fp_set_ui(&c, val);
//             fp_add(&S_new->m[i][j], &S_new->m[i][j], &c);
//         }
//     }
//     fp_clear(&c);
// }

// //1ラウンド分の処理
// void matrix_round_P(state_t *S_new, const state_t *S, int r, const state_t *MDS, const affine16_t *AFF){
//     state_t T1, T2, T3;

//     state_init(&T1);
//     state_init(&T2);
//     state_init(&T3);

//     //add round constant
//     matrix_add_round_constant_P(&T1,S,r);

//     // printf("after add round constant:\n");
//     // state_print(&T1);

//     //subbytes
//     matrix_subbytes(&T2,&T1, AFF);

//     // printf("after subbytes:\n");
//     // state_print(&T2);

//     //shiftbytes
//     matrix_shiftbytes_P(&T3,&T2);

//     // printf("after shiftbytes:\n");
//     // state_print(&T3);

//     //mixbytes
//     matrix_mixbytes(S_new,&T3,MDS);

//     // printf("after mixbytes:\n");
//     // state_print(S_new);

//     state_clear(&T1);
//     state_clear(&T2);
//     state_clear(&T3);
// }

// void matrix_round_Q(state_t *S_new, const state_t *S, int r, const state_t *MDS, const affine16_t *AFF){
//     state_t T1, T2, T3;
//     state_init(&T1);
//     state_init(&T2);
//     state_init(&T3);

//     //add round constant
//     matrix_add_round_constant_Q(&T1,S,r);

//     //subbytes
//     matrix_subbytes(&T2,&T1, AFF);

//     //shiftbytes
//     matrix_shiftbytes_Q(&T3,&T2);

//     //mixbytes
//     matrix_mixbytes(S_new, &T3, MDS);

//     state_clear(&T1);
//     state_clear(&T2);
//     state_clear(&T3);
// }

// //matrix_round_Pをround回分行う関数
// void matrix_permutation_P(state_t *S_new, const state_t *S, int rounds, const state_t *MDS, const affine16_t *AFF){
//     state_t cur, next;

//     state_init(&cur);
//     state_init(&next);

//     state_copy(&cur,S);

//     for(int r = 0; r < rounds; r++){
//         matrix_round_P(&next, &cur, r, MDS, AFF);
//         state_copy(&cur, &next);
//     }
//     state_copy(S_new, &cur);
//     state_clear(&cur);
//     state_clear(&next);
// }

// //matrix_round_Qをround回分行う関数
// void matrix_permutation_Q(state_t *S_new, const state_t *S, int rounds, const state_t *MDS, const affine16_t *AFF){
//     state_t cur, next;

//     state_init(&cur);
//     state_init(&next);

//     state_copy(&cur,S);

//     for(int r = 0; r < rounds; r++){
//         matrix_round_Q(&next, &cur, r, MDS, AFF);
//         state_copy(&cur, &next);
//     }
//     state_copy(S_new, &cur);
//     state_clear(&cur);
//     state_clear(&next);
// }

// //圧縮関数(out:次の状態行列、h:現在の状態行列、m:圧縮したいメッセージブロック)
// void matrix_compression(state_t *out, const state_t *h, const state_t *m, int rounds, const state_t *MDS, const affine16_t *AFF){
//     state_t hm;
//     state_t p_out;
//     state_t q_out;

//     state_init(&hm);
//     state_init(&p_out);
//     state_init(&q_out);

//     //hm = h + m
//     state_add(&hm, h, m);

//     //p_out = P(h + m)
//     matrix_permutation_P(&p_out, &hm, rounds, MDS, AFF);

//     //q_out = Q(m)
//     matrix_permutation_Q(&q_out, m, rounds, MDS, AFF);

//     // out = p_out + q_out + h
//     state_add3(out, &p_out, &q_out, h);

//     state_clear(&hm);
//     state_clear(&p_out);
//     state_clear(&q_out);
// }

// //P(h)+hを計算←まだここでは出力は行列のまま
// void matrix_output_transform(state_t *out, const state_t *h, int rounds, const state_t *MDS, const affine16_t *AFF){
//     state_t p_out;
//     state_init(&p_out);

//     //p_out = P(h)
//     matrix_permutation_P(&p_out, h, rounds, MDS, AFF);

//     //out = P(h) + h
//     state_add(out, &p_out, h);

//     state_clear(&p_out);
// }

// //fp4の値をout[16]に格納→out[i]はそれぞれ1byteの大きさになる8bit(1byte)*16=128bit(16byte)
// static void fp4_to_bytes_128(uint8_t *out, const fp4_t *x){
//     uint32_t v;

//     v = x->x0.x0;
//     out[0] = (uint8_t)((v >> 24) & 0xff);
//     out[1] = (uint8_t)((v >> 16) & 0xff);
//     out[2] = (uint8_t)((v >> 8) & 0xff);  //v >> 8では、32-8=24で上位24ビット取り出してしまうので、その中の下位8ビットだけが欲しい→&0xffする(0xffは0~7bitすべて1)
//     out[3] = (uint8_t)(v & 0xff);

//     v = x->x1.x0;
//     out[4] = (uint8_t)((v >> 24) & 0xff);
//     out[5] = (uint8_t)((v >> 16) & 0xff);
//     out[6] = (uint8_t)((v >> 8) & 0xff);
//     out[7] = (uint8_t)(v & 0xff);

//     v = x->x2.x0;
//     out[8] = (uint8_t)((v >> 24) & 0xff);
//     out[9] = (uint8_t)((v >> 16) & 0xff);
//     out[10] = (uint8_t)((v >> 8) & 0xff);
//     out[11] = (uint8_t)(v & 0xff);

//     v = x->x3.x0;
//     out[12] = (uint8_t)((v >> 24) & 0xff);
//     out[13] = (uint8_t)((v >> 16) & 0xff);
//     out[14] = (uint8_t)((v >> 8) & 0xff);
//     out[15] = (uint8_t)(v & 0xff);
// }

// //fp16の値をout[64]に格納→out[i]はそれぞれ1byteの大きさになる 8bit(1byte)*64=512bit(64byte)
// // void fp16_to_bytes_512(uint8_t out[MATRIX_STATE_BYTES], const fp16_t *x){
// //     fp4_to_bytes_128(out +  0, &x->x0);
// //     fp4_to_bytes_128(out + 16, &x->x1);
// //     fp4_to_bytes_128(out + 32, &x->x2);
// //     fp4_to_bytes_128(out + 48, &x->x3);
// // }

// void fp16_to_bytes_512(uint8_t out[MATRIX_STATE_BYTES], const fp16_t *x){
//     /* 16個の31ビット係数を所定の順序で集める */
//     const fp4_t *blk[4] = { &x->x0, &x->x1, &x->x2, &x->x3 };
//     uint32_t vals[16];
//     int idx = 0;
//     for(int i = 0; i < 4; i++){
//         vals[idx++] = blk[i]->x0.x0 & 0x7fffffffu;
//         vals[idx++] = blk[i]->x1.x0 & 0x7fffffffu;
//         vals[idx++] = blk[i]->x2.x0 & 0x7fffffffu;
//         vals[idx++] = blk[i]->x3.x0 & 0x7fffffffu;
//     }
 
//     /* MSBファーストで31ビットずつ詰める */
//     memset(out, 0, MATRIX_STATE_BYTES);
//     size_t bitpos = 0;
//     for(int e = 0; e < 16; e++){
//         uint32_t v = vals[e];
//         for(int b = 30; b >= 0; b--){                 /* 31ビット, MSBから */
//             if((v >> b) & 1u)
//                 out[bitpos >> 3] |= (uint8_t)(0x80u >> (bitpos & 7));
//             bitpos++;
//         }
//     }
//     /* bitpos == 496。残り out[62],out[63] は memset により 0 のまま。 */
// }

// //行列からビット列(512bit)に変換
// void matrix_state_to_bytes_512(uint8_t out[MATRIX_STATE_BYTES], const state_t *S){
//     fp16_t x;
//     fp16_init(&x);

//     state_to_fp16(&x,S);
//     fp16_to_bytes_512(out, &x);

//     fp16_clear(&x);
// }

// //ビット列から任意のbyte数だけ取り出す関数
// int matrix_trunc_head_bytes(uint8_t *digest, size_t digest_len, const uint8_t full_state[MATRIX_STATE_BYTES]){
//     if(digest == NULL || full_state == NULL){  //なぜdigest == NULLを確認？→NULLアドレスに書き込むとクラッシュするから
//         return 0;
//     }

//     if(digest_len > MATRIX_STATE_BYTES){
//         return 0;
//     }

//     // size_t start = MATRIX_STATE_BYTES - digest_len; //先頭からdigest_len分切り出す仕様にしたので、これは必要ない
//     memcpy(digest, full_state, digest_len); //digest_len =16なら、start = 48なので、full_state[16]~[63]を取り出す[0]が先頭、[MATRIX_STATE_BYTES]が末尾

//     return 1;
// }

// //Ω(h) = trunc_n(P(h) + h)
// int matrix_output_transform_digest(uint8_t *digest, size_t digest_len, const state_t *h, int rounds, const state_t *MDS, const affine16_t *AFF){
//     state_t state_final;
//     uint8_t full_state[MATRIX_STATE_BYTES];

//     if(digest == NULL || h == NULL || MDS == NULL){
//         return 0;
//     }

//     if(digest_len > MATRIX_STATE_BYTES){
//         return 0;
//     }

//     state_init(&state_final);

//     //state_final = P(h) + h
//     matrix_output_transform(&state_final, h, rounds, MDS, AFF);

//     //tを512bit列に変換
//     matrix_state_to_bytes_512(full_state, &state_final);

//     int ok = matrix_trunc_head_bytes(digest, digest_len, full_state);

//     state_clear(&state_final);

//     return ok;
// }

// //4byte配列を32bitに変換
// static uint32_t load_u32_be(const uint8_t in[4]){
//     return ((uint32_t)in[0] << 24)
//          | ((uint32_t)in[1] << 16)
//          | ((uint32_t)in[2] << 8)
//          | ((uint32_t)in[3]);
// }

// //メッセージブロック(64byte)を4*4行列に変換
// void matrix_bytes_to_state(state_t *S, const uint8_t block[MATRIX_STATE_BYTES]){ //block[i]には1byte
//     size_t pos = 0;

//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             uint32_t v = load_u32_be(block + pos); //block[0]~block[3]の値をvに格納→S->m[0][0]へ代入→block[4]~block[7]の値をvに格納→S->m[0][1]に...
//             //mod pしてから行列に格納
//             v %= P_MERSENNE;
//             fp_set_ui(&S->m[i][j], v);

//             pos += 4;
//         }
//     }
// }

// //16進数を出力
// void print_bytes_hex(const uint8_t *buf, size_t len){
//     for (size_t i = 0; i < len; i++) {
//         printf("%02x", buf[i]);

//         // 4 byteごとに空白を入れる
//         if ((i + 1) % 4 == 0) {
//             printf(" ");
//         }

//         // 16 byteごとに改行する
//         if ((i + 1) % 16 == 0) {
//             printf("\n");
//         }
//     }

//     if (len % 16 != 0) {
//         printf("\n");
//     }
// }

// //1ブロック分のハッシュ処理をまとめる関数
// int matrix_hash_one_block(uint8_t *digest, size_t digest_len, const uint8_t block[MATRIX_STATE_BYTES], int rounds, const state_t *MDS, const affine16_t *AFF){
//     state_t h;
//     state_t m;
//     state_t h_new;

//     if(digest == NULL || block == NULL || MDS == NULL){
//         return 0;
//     }

//     state_init(&h);
//     state_init(&m);
//     state_init(&h_new);

//     //初期値 h = IV
//     state_set_zero(&h);

//     // printf("h0 :\n");
//     // state_print(&h);

//     //64byteのブロック→状態行列 m
//     matrix_bytes_to_state(&m, block);

//     // printf("m :\n");
//     // state_print(&m);

//     //h_new = 圧縮関数(h,m)
//     matrix_compression(&h_new, &h, &m, rounds, MDS, AFF);

//     int ok = matrix_output_transform_digest(digest, digest_len, &h_new, rounds, MDS, AFF);

//     state_clear(&h);
//     state_clear(&m);
//     state_clear(&h_new);

//     return ok;
// }

// size_t matrix_padded_length(size_t msg_len){
//     size_t len = msg_len + 1 + 8;  //1:0x80、 8:メッセージ長を入れる領域
//     size_t rem = len % MATRIX_STATE_BYTES;

//     if(rem == 0){
//         return len;
//     }
//     return len + (MATRIX_STATE_BYTES - rem);
// }

// //メッセージ長の最大値を超えないかの確認
// int matrix_check_msg_len(size_t msg_len){
//     if (msg_len > UINT64_MAX / 8) {
//         return 0; // 長すぎる
//     }
//     return 1;
// }

// void matrix_pad(uint8_t *out, size_t padded_len, const uint8_t *msg, size_t msg_len){
//     //まず全体を0で埋める
//     memset(out, 0, padded_len);

//     //元のメッセージを先頭にコピー
//     memcpy(out, msg, msg_len);

//     //メッセージの末尾の隣に0x80を追加
//     out[msg_len] = 0x80;

//     //最後の8byteにメッセージ長をbit単位で入れる
//     uint64_t bit_len = (uint64_t)msg_len * 8;

//     out[padded_len -8] = (uint8_t)(bit_len >> 56);
//     out[padded_len -7] = (uint8_t)(bit_len >> 48);
//     out[padded_len -6] = (uint8_t)(bit_len >> 40);
//     out[padded_len -5] = (uint8_t)(bit_len >> 32);
//     out[padded_len -4] = (uint8_t)(bit_len >> 24);
//     out[padded_len -3] = (uint8_t)(bit_len >> 16);
//     out[padded_len -2] = (uint8_t)(bit_len >> 8);
//     out[padded_len -1] = (uint8_t)(bit_len);
// }



// // void test_matrix_hash_one_block(void)
// // {
// //     uint8_t block[MATRIX_STATE_BYTES] = {0};
// //     uint8_t digest[32];

// //     /*
// //       分かりやすい入力を入れる
// //       block = 0,1,2,...,63
// //     */
// //     for (int i = 0; i < MATRIX_STATE_BYTES; i++) {
// //         block[i] = (uint8_t)i;
// //     }

// //     int ok = matrix_hash_one_block(
// //         digest,
// //         32,              // 256 bit digest
// //         block,
// //         MATRIX_ROUNDS,
// //         &MDS
// //     );

// //     if (!ok) {
// //         printf("matrix_hash_one_block failed\n");
// //         return;
// //     }

// //     printf("digest = ");
// //     matrix_print_digest_hex(digest, 32);
// // }

















// void test_state_bytes_roundtrip(void){
//     state_t S, T;
//     uint8_t buf[64];

//     state_init(&S);
//     state_init(&T);

//     state_random(&S);

//     printf("S:\n");
//     state_print(&S);

//     matrix_state_to_bytes_512(buf, &S);

//     printf("buf :\n");
//     print_bytes_hex(buf, 64);

//     matrix_bytes_to_state(&T, buf);

//     printf("T:\n");
//     state_print(&T);

//     if (state_equal(&S, &T)) {
//         printf("state -> bytes -> state: OK\n");
//     } else {
//         printf("state -> bytes -> state: NG\n");
//         printf("S:\n");
//         state_print(&S);
//         printf("T:\n");
//         state_print(&T);
//     }

//     state_clear(&S);
//     state_clear(&T);
// }



// // static void matrix_subbytes_with_inv(state_t *S, fp16_inv_func inv){
// //     fp16_t a, out;
// //     state_to_fp16(&a, S);
// //     //もしaが0なら，逆元計算できないので先に見ておく
// //     if (fp4_is_zero(&a.x0) && fp4_is_zero(&a.x1) &&
// //         fp4_is_zero(&a.x2) && fp4_is_zero(&a.x3)) {
// //         state_set_zero(S);
// //         return;
// //     }
// //     inv(&out, &a);
// //     state_from_fp16(S, &out);
// // }

// // void matrix_subbytes_inv(state_t *S){
// //     matrix_subbytes_with_inv(S, fp16_inv);
// // }

// // void matrix_subbytes_inv_new(state_t *S){
// //     matrix_subbytes_with_inv(S, fp16_inv_new);
// // }

// // void matrix_subbytes_inv_karatsuba(state_t *S){
// //     matrix_subbytes_with_inv(S, fp16_inv_karatsuba);
// // }

// // void matrix_add_round_constant_p(state_t *S, uint8_t round){
// //     static const uint8_t base[4] = {0x00, 0x10, 0x20, 0x30};
// //     fp_t c;
// //     for (int j = 0; j < 4; j++) {
// //         c.x0 = (uint32_t)(base[j] ^ round);
// //         fp_add(&S->m[0][j], &S->m[0][j], &c);
// //     }
// // }

// // void matrix_add_round_constant_q(state_t *S, uint8_t round){
// //     static const uint8_t base[4] = {0xff, 0xef, 0xdf, 0xcf};
// //     fp_t c;
// //     for (int i = 0; i < 3; i++) {
// //         for (int j = 0; j < 4; j++) {
// //             c.x0 = 0xff;
// //             fp_add(&S->m[i][j], &S->m[i][j], &c);
// //         }
// //     }
// //     for (int j = 0; j < 4; j++) {
// //         c.x0 = (uint32_t)(base[j] ^ round);
// //         fp_add(&S->m[3][j], &S->m[3][j], &c);
// //     }
// // }

// // void matrix_shiftbytes(state_t *S){
// //     fp_t tmp[4];
// //     for (int i = 1; i < 4; i++) {
// //         for (int j = 0; j < 4; j++) tmp[j] = S->m[i][j];
// //         for (int j = 0; j < 4; j++) {
// //             S->m[i][j] = tmp[(j + i) % 4];
// //         }
// //     }
// // }



// // static fp16_inv_func matrix_inv_from_mode(int inv_mode){
// //     switch (inv_mode) {
// //         case 1: return fp16_inv_new;
// //         case 2: return fp16_inv_karatsuba;
// //         default: return fp16_inv;
// //     }
// // }

// // void matrix_round_p(state_t *S, uint8_t round, int inv_mode){
// //     matrix_add_round_constant_p(S, round);
// //     matrix_subbytes_with_inv(S, matrix_inv_from_mode(inv_mode));
// //     matrix_shiftbytes(S);
// //     matrix_mixbytes(S);
// // }

// // void matrix_round_q(state_t *S, uint8_t round, int inv_mode){
// //     matrix_add_round_constant_q(S, round);
// //     matrix_subbytes_with_inv(S, matrix_inv_from_mode(inv_mode));
// //     matrix_shiftbytes(S);
// //     matrix_mixbytes(S);
// // }
