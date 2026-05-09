#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

typedef struct {
    uint64_t step;
    uint64_t live_word;
    uint64_t carry_event;
    uint64_t A;
    uint64_t theta_ticks;
    uint64_t kappa;
    uint64_t u;
    uint64_t v;
    uint64_t uv;
    uint64_t floor_den;
    uint64_t window_ready;
    uint64_t r_num;
    uint64_t r_den;
    uint64_t cL_num;
    uint64_t cR_num;
    uint64_t c_den;
    uint64_t lowbit;
} XiState;

extern void xi_step(XiState *s);

static void write_trace(const char *path, XiState *hist, size_t n) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(1); }
    fprintf(f, "step,live_word,lowbit,carry_event,A,theta_ticks,kappa,u,v,uv,floor_den,window_ready,r_num,r_den,cL_num,cR_num,c_den\n");
    for (size_t i = 0; i < n; ++i) {
        XiState *s = &hist[i];
        fprintf(f,
            "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
            s->step, s->live_word, s->lowbit, s->carry_event, s->A, s->theta_ticks, s->kappa,
            s->u, s->v, s->uv, s->floor_den, s->window_ready, s->r_num, s->r_den,
            s->cL_num, s->cR_num, s->c_den);
    }
    fclose(f);
}

int main(void) {
    XiState s = {0};
    s.live_word = 1;
    s.A = 0;
    s.theta_ticks = 0;
    s.kappa = 0;
    s.u = 1;
    s.v = 1;
    s.uv = 1;
    s.floor_den = 4096;
    s.window_ready = 0;
    s.r_num = 1;
    s.r_den = 1;
    s.cL_num = (uint64_t)-2;
    s.cR_num = 2;
    s.c_den = 2;
    s.lowbit = 1;

    const size_t N = 192;
    XiState *hist = calloc(N, sizeof(XiState));
    if (!hist) { perror("calloc"); return 1; }

    for (size_t i = 0; i < N; ++i) {
        xi_step(&s);
        hist[i] = s;
    }

    write_trace("/mnt/data/xi_full_engine_trace_patched.csv", hist, N);

    FILE *txt = fopen("/mnt/data/xi_full_engine_summary_patched.txt", "w");
    if (!txt) { perror("summary"); return 1; }

    fprintf(txt, "First 32 rows:\n");
    for (size_t i = 0; i < 32 && i < N; ++i) {
        XiState *x = &hist[i];
        fprintf(txt,
            "step=%" PRIu64 " live=%" PRIu64 " carry=%" PRIu64 " A=%" PRIu64 " theta=%" PRIu64 " kappa=%" PRIu64
            " u=%" PRIu64 " v=%" PRIu64 " uv=%" PRIu64 " ready=%" PRIu64 " r=1/%" PRIu64
            " c=[(%lld)/(%llu) pi, (%lld)/(%llu) pi]\n",
            x->step, x->live_word, x->carry_event, x->A, x->theta_ticks, x->kappa,
            x->u, x->v, x->uv, x->window_ready, x->r_den,
            (long long)x->cL_num, (unsigned long long)x->c_den,
            (long long)x->cR_num, (unsigned long long)x->c_den);
    }

    fprintf(txt, "\nCarry articulation steps:\n");
    for (size_t i = 0; i < N; ++i) {
        if (hist[i].carry_event) {
            fprintf(txt, "step=%" PRIu64 " A=%" PRIu64 " live=%" PRIu64 " u=%" PRIu64 " v=%" PRIu64 " uv=%" PRIu64 "\n",
                    hist[i].step, hist[i].A, hist[i].live_word, hist[i].u, hist[i].v, hist[i].uv);
        }
    }

    fprintf(txt, "\nFirst window per articulation class:\n");
    uint64_t last_A = UINT64_MAX;
    for (size_t i = 0; i < N; ++i) {
        if (hist[i].window_ready && hist[i].A != last_A) {
            fprintf(txt, "A=%" PRIu64 " step=%" PRIu64 " u=%" PRIu64 " v=%" PRIu64 " uv=%" PRIu64 " r=1/%" PRIu64 "\n",
                    hist[i].A, hist[i].step, hist[i].u, hist[i].v, hist[i].uv, hist[i].r_den);
            last_A = hist[i].A;
        }
    }

    fclose(txt);
    free(hist);
    return 0;
}
