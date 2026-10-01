#include "16_header.h"
void test_pack_127(void){
    field_select_for_output(512);   // FP_BITS=127

    state_t S; state_init(&S);
    u128 full = (((u128)1) << FP_BITS) - 1;   // 2^127 - 1

    // ★注意: full == P_MERSENNE なので fp_set_ui で % P_MERSENNE すると 0 になる。
    //   それを避けるため、全ビット1ではなく「p-1(=2^127-2)」を使って確認する。
    u128 val = P_MERSENNE-1;                // = 2^127 - 2 (p 未満で最大, 上位ビットまで1)
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) fp_set_ui(&S.m[i][j], val);

    // --- 直前: S の各要素を hi/lo で表示 ---
    printf("=== S (各要素の x0 を hi:lo) ===\n");
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            u128 v = S.m[i][j].x0;
            printf("  m[%d][%d] hi=%016llx lo=%016llx\n", i, j,
                   (unsigned long long)(v>>64),
                   (unsigned long long)(v & 0xFFFFFFFFFFFFFFFFULL));
        }
    }

    uint8_t buf[MATRIX_STATE_BYTES_MAX];
    matrix_state_to_bytes_512(buf, &S);

    // --- 直後: buf を16進ダンプ ---
    printf("=== buf (state_bytes=%zu) ===\n", state_bytes);
    for(size_t i=0;i<state_bytes;i++){
        printf("%02x", buf[i]);
        if((i+1)%16==0) printf("\n");
        else if((i+1)%4==0) printf(" ");
    }
    printf("\n");

    int nonff=0;
    for(size_t i=0;i<state_bytes;i++) if(buf[i]!=0xff) nonff++;
    printf("0xff以外のバイト数=%d\n", nonff);
}

int main(void){
    uint32_t p_mer = P_MERSENNE;
    uint32_t p_pri = P_PRIME;
    // printf("p_mer : %u\n",p_mer); //この段階では規定値が入っている
    // printf("p_pri : %u\n",p_pri);
    printf("α : "); fp4_printf(&alpha);

    //MDS行列をセット
    state_t MDS;
    setup_MDS(&MDS);

    affine16_t AFF;
    affine16_set(&AFF);
    printf("A:\n");
    affine16_print_A(&AFF);
    printf("b:\n");
    affine16_print_b(&AFF);

    int A = affine16_is_regular(&AFF);

    if (A){
        printf("A is regular\n");
    }else{
        printf("A is not regular\n");
    }



    // affine16_set(&AFF);

    uint8_t block[MATRIX_BLOCK_BYTES_MAX] = {0};
    uint8_t digest[MATRIX_DIGEST_BYTES_MAX];
    uint8_t msg[] = "asahi"; //文字列リテラル "Hello World" の各文字コードが uint8_t 配列に格納される
    size_t msg_len = sizeof(msg) -1; //末尾のヌル文字\0も数えてしまい、1文字多くなってしまうから1を引く
    size_t digest_len = 32; //ここで出力長を決定（これはバイト長）, digest_len <= 8 → p = 2^7-1,  9 <= digest_len <= 32 → p = 2^31-1, 33 <= digest_len <= 127 → p = 2^127-1

    printf("MDS :\n");
    state_print(&MDS);

    printf("msg : ");
    printf("%s\n",msg);
    // print_bytes_hex(msg, 11);

    //throughput_matrix_hash(msg, msg_len, 100, &MDS, &AFF);

    matrix_hash(digest, digest_len, msg, msg_len, MATRIX_ROUNDS, &MDS, &AFF);

    printf("p_mer : "); print_u128_dec(P_MERSENNE); printf("\n");
    printf("digest :\n");
    print_bytes_hex(digest, digest_len);

    state_clear(&MDS);
    affine16_clear(&AFF);
    return 0;
}

