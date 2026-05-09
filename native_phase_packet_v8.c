#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

typedef long mpfr_prec_t;
typedef long mpfr_exp_t;
typedef unsigned long mp_limb_t;
typedef struct {
    mpfr_prec_t _mpfr_prec;
    int _mpfr_sign;
    mpfr_exp_t _mpfr_exp;
    mp_limb_t *_mpfr_d;
} __mpfr_struct;
typedef __mpfr_struct mpfr_t[1];
typedef __mpfr_struct * mpfr_ptr;
typedef const __mpfr_struct * mpfr_srcptr;
typedef int mpfr_rnd_t;

#define MPFR_RNDN 0
#define MPFR_RNDD 3
#define EXPORT __attribute__((visibility("default")))

void mpfr_init2(mpfr_ptr x, mpfr_prec_t prec);
void mpfr_clear(mpfr_ptr x);
int mpfr_set_d(mpfr_ptr rop, double op, mpfr_rnd_t rnd);
int mpfr_set_ui(mpfr_ptr rop, unsigned long op, mpfr_rnd_t rnd);
int mpfr_set(mpfr_ptr rop, mpfr_srcptr op, mpfr_rnd_t rnd);
int mpfr_add(mpfr_ptr rop, mpfr_srcptr op1, mpfr_srcptr op2, mpfr_rnd_t rnd);
int mpfr_sub(mpfr_ptr rop, mpfr_srcptr op1, mpfr_srcptr op2, mpfr_rnd_t rnd);
int mpfr_mul(mpfr_ptr rop, mpfr_srcptr op1, mpfr_srcptr op2, mpfr_rnd_t rnd);
int mpfr_mul_ui(mpfr_ptr rop, mpfr_srcptr op1, unsigned long op2, mpfr_rnd_t rnd);
int mpfr_div(mpfr_ptr rop, mpfr_srcptr op1, mpfr_srcptr op2, mpfr_rnd_t rnd);
int mpfr_div_ui(mpfr_ptr rop, mpfr_srcptr op1, unsigned long op2, mpfr_rnd_t rnd);
int mpfr_sub_ui(mpfr_ptr rop, mpfr_srcptr op1, unsigned long op2, mpfr_rnd_t rnd);
int mpfr_ui_div(mpfr_ptr rop, unsigned long op1, mpfr_srcptr op2, mpfr_rnd_t rnd);
int mpfr_sqr(mpfr_ptr rop, mpfr_srcptr op, mpfr_rnd_t rnd);
int mpfr_sqrt(mpfr_ptr rop, mpfr_srcptr op, mpfr_rnd_t rnd);
unsigned long mpfr_get_ui(mpfr_srcptr op, mpfr_rnd_t rnd);
char * mpfr_get_str(char *str, mpfr_exp_t *expptr, int base, size_t n, mpfr_srcptr op, mpfr_rnd_t rnd);

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static mpfr_prec_t dps_to_bits(unsigned dps) {
    return (mpfr_prec_t)ceil(((double)dps + 8.0) * 3.32192809488736234787) + 48;
}

static double coarse_bound_log10(unsigned iters) {
    const double LOG10_2 = 0.30102999566398119521;
    const double INV_LN10 = 0.43429448190325182765;
    return ((double)iters + 8.0) * LOG10_2 - 3.0 * (double)(1u << (iters + 1u)) * INV_LN10;
}

static unsigned agm_iterations_for_digits(unsigned target_digits, unsigned probe_digits) {
    const unsigned required = target_digits + probe_digits + 4u;
    for (unsigned iters = 1u; iters < 32u; ++iters) {
        double digits = -coarse_bound_log10(iters);
        if (digits > (double)required) return iters;
    }
    return 31u;
}

typedef struct {
    int initialized;
    unsigned target_digits;
    unsigned probe_digits;
    unsigned iters;
    mpfr_prec_t bits;
    mpfr_t sqrt2, a, b, t, p, an, bn, tn, tmp1, tmp2, x;
} AgmCache;

static AgmCache g_agm = {0};

static void agm_clear(AgmCache *c) {
    if (!c->initialized) return;
    mpfr_clear(c->sqrt2);
    mpfr_clear(c->a);
    mpfr_clear(c->b);
    mpfr_clear(c->t);
    mpfr_clear(c->p);
    mpfr_clear(c->an);
    mpfr_clear(c->bn);
    mpfr_clear(c->tn);
    mpfr_clear(c->tmp1);
    mpfr_clear(c->tmp2);
    mpfr_clear(c->x);
    memset(c, 0, sizeof(*c));
}

static void agm_prepare(AgmCache *c, unsigned target_digits, unsigned probe_digits) {
    if (c->initialized && c->target_digits == target_digits && c->probe_digits == probe_digits) return;
    agm_clear(c);
    c->target_digits = target_digits;
    c->probe_digits = probe_digits;
    c->iters = agm_iterations_for_digits(target_digits, probe_digits);
    c->bits = dps_to_bits(target_digits + 24u);
    mpfr_init2(c->sqrt2, c->bits);
    mpfr_init2(c->a, c->bits);
    mpfr_init2(c->b, c->bits);
    mpfr_init2(c->t, c->bits);
    mpfr_init2(c->p, c->bits);
    mpfr_init2(c->an, c->bits);
    mpfr_init2(c->bn, c->bits);
    mpfr_init2(c->tn, c->bits);
    mpfr_init2(c->tmp1, c->bits);
    mpfr_init2(c->tmp2, c->bits);
    mpfr_init2(c->x, c->bits);
    mpfr_set_ui(c->tmp1, 2u, MPFR_RNDN);
    mpfr_sqrt(c->sqrt2, c->tmp1, MPFR_RNDN);
    c->initialized = 1;
}

static void agm_compute_pi(AgmCache *c) {
    mpfr_set_ui(c->a, 1u, MPFR_RNDN);
    mpfr_ui_div(c->b, 1u, c->sqrt2, MPFR_RNDN);
    mpfr_set_d(c->t, 0.25, MPFR_RNDN);
    mpfr_set_ui(c->p, 1u, MPFR_RNDN);
    for (unsigned i = 0; i < c->iters; ++i) {
        mpfr_add(c->an, c->a, c->b, MPFR_RNDN);
        mpfr_div_ui(c->an, c->an, 2u, MPFR_RNDN);
        mpfr_mul(c->bn, c->a, c->b, MPFR_RNDN);
        mpfr_sqrt(c->bn, c->bn, MPFR_RNDN);
        mpfr_sub(c->tmp1, c->a, c->an, MPFR_RNDN);
        mpfr_sqr(c->tmp1, c->tmp1, MPFR_RNDN);
        mpfr_mul(c->tmp1, c->p, c->tmp1, MPFR_RNDN);
        mpfr_sub(c->tn, c->t, c->tmp1, MPFR_RNDN);
        mpfr_mul_ui(c->p, c->p, 2u, MPFR_RNDN);
        mpfr_set(c->a, c->an, MPFR_RNDN);
        mpfr_set(c->b, c->bn, MPFR_RNDN);
        mpfr_set(c->t, c->tn, MPFR_RNDN);
    }
    mpfr_add(c->tmp1, c->a, c->b, MPFR_RNDN);
    mpfr_sqr(c->tmp1, c->tmp1, MPFR_RNDN);
    mpfr_mul_ui(c->tmp2, c->t, 4u, MPFR_RNDN);
    mpfr_div(c->x, c->tmp1, c->tmp2, MPFR_RNDN);
}

static int mpfr_emit_decimal_and_probe(mpfr_ptr x,
                                       unsigned digits,
                                       unsigned probe_digits,
                                       char *outbuf,
                                       size_t outlen,
                                       char *probebuf,
                                       size_t probelen,
                                       unsigned *first_non9_pos_out) {
    const size_t sig_digits = (size_t)digits + (size_t)probe_digits + 1u;
    if (outlen < (size_t)digits + 4u) return 5;
    if (probe_digits > 0u && probelen < (size_t)probe_digits + 1u) return 6;
    char *mant = (char *)malloc(sig_digits + 8u);
    if (!mant) return 7;
    mpfr_exp_t expo = 0;
    if (mpfr_get_str(mant, &expo, 10, sig_digits, x, MPFR_RNDD) == NULL) {
        free(mant);
        return 8;
    }
    if (expo != 1) {
        free(mant);
        return 9;
    }
    outbuf[0] = mant[0];
    outbuf[1] = '.';
    memcpy(outbuf + 2, mant + 1, digits);
    outbuf[digits + 2] = '\0';
    unsigned first_non9 = 0u;
    for (unsigned i = 0u; i < probe_digits; ++i) {
        char ch = mant[1u + digits + i];
        probebuf[i] = ch;
        if (first_non9 == 0u && ch < '9') first_non9 = i + 1u;
    }
    if (probe_digits > 0u) probebuf[probe_digits] = '\0';
    if (first_non9_pos_out) *first_non9_pos_out = first_non9;
    free(mant);
    return 0;
}

static int certify_target_digits(unsigned target_digits,
                                 unsigned iters,
                                 unsigned first_non9_pos,
                                 double *bound_log10_out,
                                 unsigned *safe_digits_lower_bound_out) {
    double log10B = coarse_bound_log10(iters);
    if (bound_log10_out) *bound_log10_out = log10B;
    if (first_non9_pos == 0u) {
        if (safe_digits_lower_bound_out) *safe_digits_lower_bound_out = 0u;
        return 0;
    }
    /* If the first non-9 appears at position j, carry margin exceeds 10^{-j}. */
    double lhs = log10B + (double)target_digits;
    double rhs = -(double)first_non9_pos;
    if (safe_digits_lower_bound_out) {
        double val = -log10B - (double)first_non9_pos;
        *safe_digits_lower_bound_out = (val > 0.0) ? (unsigned)floor(val) : 0u;
    }
    return (lhs < rhs) ? 1 : 0;
}

EXPORT int phase_native_pi_hot_v8(unsigned target_digits,
                                  unsigned probe_digits,
                                  char *outbuf,
                                  size_t outlen,
                                  char *probebuf,
                                  size_t probelen,
                                  double *seconds_out,
                                  unsigned *iters_out,
                                  unsigned *first_non9_pos_out) {
    agm_prepare(&g_agm, target_digits, probe_digits);
    double t0 = now_sec();
    agm_compute_pi(&g_agm);
    int rc = mpfr_emit_decimal_and_probe(g_agm.x, target_digits, probe_digits,
                                         outbuf, outlen, probebuf, probelen, first_non9_pos_out);
    if (seconds_out) *seconds_out = now_sec() - t0;
    if (iters_out) *iters_out = g_agm.iters;
    return rc;
}

EXPORT int phase_native_pi_full_v8(unsigned target_digits,
                                   unsigned probe_digits,
                                   char *outbuf,
                                   size_t outlen,
                                   char *probebuf,
                                   size_t probelen,
                                   double *seconds_out,
                                   unsigned *iters_out,
                                   int *cert_ok_out,
                                   double *bound_log10_out,
                                   unsigned *safe_digits_lower_bound_out,
                                   unsigned *first_non9_pos_out) {
    unsigned first_non9 = 0u;
    int rc = phase_native_pi_hot_v8(target_digits, probe_digits, outbuf, outlen, probebuf, probelen,
                                    seconds_out, iters_out, &first_non9);
    if (rc != 0) return rc;
    unsigned iters = (iters_out != NULL) ? *iters_out : g_agm.iters;
    unsigned safe_lb = 0u;
    int ok = certify_target_digits(target_digits, iters, first_non9, bound_log10_out, &safe_lb);
    if (cert_ok_out) *cert_ok_out = ok;
    if (safe_digits_lower_bound_out) *safe_digits_lower_bound_out = safe_lb;
    if (first_non9_pos_out) *first_non9_pos_out = first_non9;
    return 0;
}

EXPORT int phase_native_pi_hot_benchmark_v8(unsigned target_digits,
                                            unsigned probe_digits,
                                            unsigned reps,
                                            double *min_seconds,
                                            double *mean_seconds) {
    if (reps == 0u) return 3;
    agm_prepare(&g_agm, target_digits, probe_digits);
    char *buf = (char *)malloc((size_t)target_digits + 32u);
    char *probe = (char *)malloc((size_t)probe_digits + 1u);
    if (!buf || !probe) {
        free(buf); free(probe);
        return 4;
    }
    phase_native_pi_hot_v8(target_digits, probe_digits, buf, (size_t)target_digits + 32u,
                           probe, (size_t)probe_digits + 1u, NULL, NULL, NULL);
    double best = 1e300, sum = 0.0;
    for (unsigned i = 0u; i < reps; ++i) {
        double secs = 0.0;
        phase_native_pi_hot_v8(target_digits, probe_digits, buf, (size_t)target_digits + 32u,
                               probe, (size_t)probe_digits + 1u, &secs, NULL, NULL);
        if (secs < best) best = secs;
        sum += secs;
    }
    free(buf); free(probe);
    if (min_seconds) *min_seconds = best;
    if (mean_seconds) *mean_seconds = sum / (double)reps;
    return 0;
}

EXPORT int phase_native_pi_full_benchmark_v8(unsigned target_digits,
                                             unsigned probe_digits,
                                             unsigned reps,
                                             double *min_seconds,
                                             double *mean_seconds) {
    if (reps == 0u) return 3;
    agm_prepare(&g_agm, target_digits, probe_digits);
    char *buf = (char *)malloc((size_t)target_digits + 32u);
    char *probe = (char *)malloc((size_t)probe_digits + 1u);
    if (!buf || !probe) {
        free(buf); free(probe);
        return 4;
    }
    phase_native_pi_full_v8(target_digits, probe_digits, buf, (size_t)target_digits + 32u,
                            probe, (size_t)probe_digits + 1u, NULL, NULL, NULL, NULL, NULL, NULL);
    double best = 1e300, sum = 0.0;
    for (unsigned i = 0u; i < reps; ++i) {
        double t0 = now_sec();
        phase_native_pi_full_v8(target_digits, probe_digits, buf, (size_t)target_digits + 32u,
                                probe, (size_t)probe_digits + 1u, NULL, NULL, NULL, NULL, NULL, NULL);
        double secs = now_sec() - t0;
        if (secs < best) best = secs;
        sum += secs;
    }
    free(buf); free(probe);
    if (min_seconds) *min_seconds = best;
    if (mean_seconds) *mean_seconds = sum / (double)reps;
    return 0;
}

EXPORT void phase_native_reset_v8(void) {
    agm_clear(&g_agm);
}
