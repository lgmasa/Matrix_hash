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

//fp16の値をout[64]に格納→out[i]はそれぞれ1byteの大きさになる 8bit(1byte)*64=512bit(64byte)
void fp16_to_bytes_512(uint8_t out[MATRIX_STATE_BYTES], const fp16_t *x){
    fp4_to_bytes_128(out +  0, &x->x0);
    fp4_to_bytes_128(out + 16, &x->x1);
    fp4_to_bytes_128(out + 32, &x->x2);
    fp4_to_bytes_128(out + 48, &x->x3);
}

//行列からビット列(512bit)に変換
void matrix_state_to_bytes_512(uint8_t out[MATRIX_STATE_BYTES], const state_t *S){
    fp16_t x;
    fp16_init(&x);

    state_to_fp16(&x,S);
    fp16_to_bytes_512(out, &x);

    fp16_clear(&x);
}

//ビット列から任意のbyte数だけ取り出す関数
int matrix_trunc_tail_bytes(uint8_t *digest, size_t digest_len, const uint8_t full_state[MATRIX_STATE_BYTES]){
    if(digest == NULL || full_state == NULL){  //なぜdigest == NULLを確認？→NULLアドレスに書き込むとクラッシュするから
        return 0;
    }

    if(digest_len > MATRIX_STATE_BYTES){
        return 0;
    }

    size_t start = MATRIX_STATE_BYTES - digest_len;
    memcpy(digest, full_state + start, digest_len); //digest_len =16なら、start = 48なので、full_state[16]~[63]を取り出す[0]が先頭、[MATRIX_STATE_BYTES]が末尾

    return 1;
}

//Ω(h) = trunc_n(P(h) + h)
int matrix_output_transform_digest(uint8_t *digest, size_t digest_len, const state_t *h, int rounds, const state_t *MDS, const affine16_t *AFF){
    state_t state_final;
    uint8_t full_state[MATRIX_STATE_BYTES];

    if(digest == NULL || h == NULL || MDS == NULL){
        return 0;
    }

    if(digest_len > MATRIX_STATE_BYTES){
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

//4byte配列を32bitに変換
static uint32_t load_u32_be(const uint8_t in[4]){
    return ((uint32_t)in[0] << 24)
         | ((uint32_t)in[1] << 16)
         | ((uint32_t)in[2] << 8)
         | ((uint32_t)in[3]);
}

//メッセージブロック(64byte)を4*4行列に変換
void matrix_bytes_to_state(state_t *S, const uint8_t block[MATRIX_STATE_BYTES]){ //block[i]には1byte
    size_t pos = 0;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            uint32_t v = load_u32_be(block + pos); //block[0]~block[3]の値をvに格納→S->m[0][0]へ代入→block[4]~block[7]の値をvに格納→S->m[0][1]に...
            //mod pしてから行列に格納
            v %= P_MERSENNE;
            fp_set_ui(&S->m[i][j], v);

            pos += 4;
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
int matrix_hash_one_block(uint8_t *digest, size_t digest_len, const uint8_t block[MATRIX_STATE_BYTES], int rounds, const state_t *MDS, const affine16_t *AFF){
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
    size_t rem = len % MATRIX_STATE_BYTES;

    if(rem == 0){
        return len;
    }
    return len + (MATRIX_STATE_BYTES - rem);
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
    
    //ここでエラーの原因になりそうなことをチェックしておく
    if(digest == NULL || msg == NULL || MDS == NULL){
        return 0;
    }
    if(digest_len > MATRIX_STATE_BYTES){
        return 0;
    }
    if(!matrix_check_msg_len(msg_len)){
        return 0;
    }

    //パディングを入れた後のバイト長を求める（実際には長さを確保しただけでまだ入れてない）
    size_t padded_len = matrix_padded_length(msg_len);

    uint8_t *padded = malloc(padded_len); //メモリ確保
    if(padded == NULL){
        return 0;
    }

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

    size_t num_blocks = padded_len / MATRIX_BLOCK_BYTES;  //もしpadded_len = 128なら、64byteのブロックが128/64=2つある=num_blocks

    for(size_t b = 0; b < num_blocks; b++){
        const uint8_t *block = padded + b * MATRIX_BLOCK_BYTES; //&padded[b * MATRIX_BLOCK_BYTES]と同義

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
