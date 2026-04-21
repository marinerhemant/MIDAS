/*
 * MapHeader.h — Parameter-hash header for Map.bin / nMap.bin
 *
 * Provides a 64-byte header written before the pixel mapping data.
 * The header contains a SHA-256 hash of all geometry/binning parameters
 * so that downstream consumers can detect stale mapping files.
 *
 * Usage in writer (DetectorMapper/DetectorMapperZarr):
 *   struct MapHeader hdr;
 *   map_header_compute(&hdr, Lsd, yCen, zCen, pxY, pxZ, tx, ty, tz,
 *                      p0, p1, p2, p3, p4, p6, RhoD,
 *                      RBinSize, EtaBinSize, RMin, RMax, EtaMin, EtaMax,
 *                      NrPixelsY, NrPixelsZ);
 *   map_header_write(mapfile, &hdr);
 *   fwrite(pxListStore, ...);
 *
 * Usage in reader (IntegratorZarrOMP, GPU):
 *   struct MapHeader file_hdr;
 *   int has_header = map_header_read(fd_or_file, &file_hdr);
 *   if (has_header) {
 *     struct MapHeader expected;
 *     map_header_compute(&expected, ...);
 *     if (!map_header_validate(&file_hdr, &expected)) {
 *       fprintf(stderr, "ERROR: Map.bin stale\n"); exit(1);
 *     }
 *   } else {
 *     fprintf(stderr, "WARNING: Map.bin has no header (legacy)\n");
 *   }
 */

#ifndef MAP_HEADER_H
#define MAP_HEADER_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Magic = "MAP0" in little-endian */
#define MAP_HEADER_MAGIC 0x3050414D
#define MAP_HEADER_VERSION 3
#define MAP_HEADER_SIZE 64

#pragma pack(push, 1)
struct MapHeader {
  uint32_t magic;         /*  4 bytes: 0x3050414D = "MAP0" */
  uint32_t version;       /*  4 bytes: 3 */
  uint8_t param_hash[32]; /* 32 bytes: SHA-256 of parameters */
  uint8_t  q_mode;        /*  1 byte:  0=R-mode, 1=Q-mode */
  uint8_t  gradient_mode; /*  1 byte:  0=no deltaR, 1=deltaR populated (v3+) */
  uint8_t  reserved_pad[6]; /* 6 bytes: alignment padding */
  double   wavelength;    /*  8 bytes: wavelength in Å (0 if R-mode) */
  uint8_t  reserved[8];   /*  8 bytes: zero-filled, future use */
};
#pragma pack(pop)

/* ──────────────────────────────────────────────────────────────────
 * Minimal SHA-256 implementation (public domain, no dependencies)
 * Based on Brad Conte's implementation
 * ────────────────────────────────────────────────────────────────── */

typedef struct {
  uint8_t data[64];
  uint32_t datalen;
  uint64_t bitlen;
  uint32_t state[8];
} MH_SHA256_CTX;

static const uint32_t mh_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define MH_ROTR(a, b) (((a) >> (b)) | ((a) << (32 - (b))))
#define MH_CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MH_MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define MH_EP0(x) (MH_ROTR(x, 2) ^ MH_ROTR(x, 13) ^ MH_ROTR(x, 22))
#define MH_EP1(x) (MH_ROTR(x, 6) ^ MH_ROTR(x, 11) ^ MH_ROTR(x, 25))
#define MH_SIG0(x) (MH_ROTR(x, 7) ^ MH_ROTR(x, 18) ^ ((x) >> 3))
#define MH_SIG1(x) (MH_ROTR(x, 17) ^ MH_ROTR(x, 19) ^ ((x) >> 10))

static void mh_sha256_transform(MH_SHA256_CTX *ctx, const uint8_t data[]) {
  uint32_t a, b, c, d, e, f, g, h, t1, t2, m[64];
  int i;
  for (i = 0; i < 16; ++i)
    m[i] = ((uint32_t)data[i * 4] << 24) | ((uint32_t)data[i * 4 + 1] << 16) |
           ((uint32_t)data[i * 4 + 2] << 8) | ((uint32_t)data[i * 4 + 3]);
  for (; i < 64; ++i)
    m[i] = MH_SIG1(m[i - 2]) + m[i - 7] + MH_SIG0(m[i - 15]) + m[i - 16];
  a = ctx->state[0];
  b = ctx->state[1];
  c = ctx->state[2];
  d = ctx->state[3];
  e = ctx->state[4];
  f = ctx->state[5];
  g = ctx->state[6];
  h = ctx->state[7];
  for (i = 0; i < 64; ++i) {
    t1 = h + MH_EP1(e) + MH_CH(e, f, g) + mh_sha256_k[i] + m[i];
    t2 = MH_EP0(a) + MH_MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
  ctx->state[5] += f;
  ctx->state[6] += g;
  ctx->state[7] += h;
}

static void mh_sha256_init(MH_SHA256_CTX *ctx) {
  ctx->datalen = 0;
  ctx->bitlen = 0;
  ctx->state[0] = 0x6a09e667;
  ctx->state[1] = 0xbb67ae85;
  ctx->state[2] = 0x3c6ef372;
  ctx->state[3] = 0xa54ff53a;
  ctx->state[4] = 0x510e527f;
  ctx->state[5] = 0x9b05688c;
  ctx->state[6] = 0x1f83d9ab;
  ctx->state[7] = 0x5be0cd19;
}

static void mh_sha256_update(MH_SHA256_CTX *ctx, const uint8_t *data,
                             size_t len) {
  size_t i;
  for (i = 0; i < len; ++i) {
    ctx->data[ctx->datalen] = data[i];
    ctx->datalen++;
    if (ctx->datalen == 64) {
      mh_sha256_transform(ctx, ctx->data);
      ctx->bitlen += 512;
      ctx->datalen = 0;
    }
  }
}

static void mh_sha256_final(MH_SHA256_CTX *ctx, uint8_t hash[32]) {
  uint32_t i = ctx->datalen;
  if (i < 56) {
    ctx->data[i++] = 0x80;
    while (i < 56)
      ctx->data[i++] = 0x00;
  } else {
    ctx->data[i++] = 0x80;
    while (i < 64)
      ctx->data[i++] = 0x00;
    mh_sha256_transform(ctx, ctx->data);
    memset(ctx->data, 0, 56);
  }
  ctx->bitlen += ctx->datalen * 8;
  ctx->data[63] = (uint8_t)(ctx->bitlen);
  ctx->data[62] = (uint8_t)(ctx->bitlen >> 8);
  ctx->data[61] = (uint8_t)(ctx->bitlen >> 16);
  ctx->data[60] = (uint8_t)(ctx->bitlen >> 24);
  ctx->data[59] = (uint8_t)(ctx->bitlen >> 32);
  ctx->data[58] = (uint8_t)(ctx->bitlen >> 40);
  ctx->data[57] = (uint8_t)(ctx->bitlen >> 48);
  ctx->data[56] = (uint8_t)(ctx->bitlen >> 56);
  mh_sha256_transform(ctx, ctx->data);
  for (i = 0; i < 4; ++i) {
    hash[i] = (ctx->state[0] >> (24 - i * 8)) & 0xFF;
    hash[i + 4] = (ctx->state[1] >> (24 - i * 8)) & 0xFF;
    hash[i + 8] = (ctx->state[2] >> (24 - i * 8)) & 0xFF;
    hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0xFF;
    hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0xFF;
    hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0xFF;
    hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0xFF;
    hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0xFF;
  }
}

/* ──────────────────────────────────────────────────────────────────
 * MapHeader API
 * ────────────────────────────────────────────────────────────────── */

/**
 * Compute the parameter hash and populate a MapHeader.
 * All geometry/binning parameters that affect the mapping are included.
 * qMode: 0 = equal-R bins, 1 = equal-Q bins.
 * Wavelength: in Å (only used and hashed when qMode=1).
 */
static void map_header_compute(struct MapHeader *hdr, double Lsd, double yCen,
                               double zCen, double pxY, double pxZ, double tx,
                               double ty, double tz, double p0, double p1,
                               double p2, double p3, double p4, double p6, double RhoD,
                               double RBinSize, double EtaBinSize, double RMin,
                               double RMax, double EtaMin, double EtaMax,
                               int NrPixelsY, int NrPixelsZ, int NrTransOpt,
                               const int TransOpt[10], int qMode,
                               double Wavelength) {
  memset(hdr, 0, sizeof(*hdr));
  hdr->magic = MAP_HEADER_MAGIC;
  hdr->version = MAP_HEADER_VERSION;
  hdr->q_mode = (uint8_t)qMode;
  hdr->wavelength = Wavelength;

  /* Build canonical parameter string (alphabetical keys) */
  char buf[2048];
  int n = snprintf(buf, sizeof(buf),
                   "BC=%.6f,%.6f|EtaBinSize=%.6f|EtaMax=%.6f|EtaMin=%.6f|"
                   "Lsd=%.6f|NrPixelsY=%d|NrPixelsZ=%d|"
                   "RBinSize=%.6f|RMax=%.6f|RMin=%.6f|RhoD=%.6f|"
                   "TransOpt=%d",
                   yCen, zCen, EtaBinSize, EtaMax, EtaMin, Lsd, NrPixelsY,
                   NrPixelsZ, RBinSize, RMax, RMin, RhoD, NrTransOpt);
  for (int i = 0; i < NrTransOpt && i < 10; i++) {
    n += snprintf(buf + n, sizeof(buf) - n, ",%d", TransOpt[i]);
  }
  n += snprintf(buf + n, sizeof(buf) - n,
                "|p0=%.6f|p1=%.6f|p2=%.6f|p3=%.6f|p4=%.6f|p6=%.6f|"
                "pxY=%.6f|pxZ=%.6f|tx=%.6f|ty=%.6f|tz=%.6f",
                p0, p1, p2, p3, p4, p6, pxY, pxZ, tx, ty, tz);
  if (qMode) {
    n += snprintf(buf + n, sizeof(buf) - n,
                  "|qMode=1|Wavelength=%.8f", Wavelength);
  }

  MH_SHA256_CTX ctx;
  mh_sha256_init(&ctx);
  mh_sha256_update(&ctx, (const uint8_t *)buf, (size_t)n);
  mh_sha256_final(&ctx, hdr->param_hash);
}

/**
 * Write header to an open FILE* (must be called before writing data).
 * Returns 0 on success, -1 on error.
 */
static int map_header_write(FILE *f, const struct MapHeader *hdr) {
  if (fwrite(hdr, MAP_HEADER_SIZE, 1, f) != 1)
    return -1;
  return 0;
}

/**
 * Check if an open file descriptor has a valid MapHeader.
 * Reads the first 64 bytes and checks for the magic number.
 * On success, populates hdr and returns 1.
 * On failure (no header / legacy file), returns 0 and seeks back.
 *
 * Note: this uses pread so the file offset is unchanged.
 */
static int map_header_read_fd(int fd, struct MapHeader *hdr) {
  ssize_t n = pread(fd, hdr, MAP_HEADER_SIZE, 0);
  if (n != MAP_HEADER_SIZE)
    return 0;
  if (hdr->magic != MAP_HEADER_MAGIC)
    return 0;
  return 1;
}

/**
 * Validate that two headers have matching parameter hashes.
 * Returns 1 if match, 0 if mismatch.
 */
static int map_header_validate(const struct MapHeader *file_hdr,
                               const struct MapHeader *expected) {
  return memcmp(file_hdr->param_hash, expected->param_hash, 32) == 0;
}

/**
 * Print header info to stdout for diagnostics.
 */
static void map_header_print(const char *filename,
                             const struct MapHeader *hdr) {
  printf("  %s header: magic=0x%08X ver=%u hash=", filename, hdr->magic,
         hdr->version);
  for (int i = 0; i < 8; i++)
    printf("%02x", hdr->param_hash[i]);
  printf("...\n");
}

#endif /* MAP_HEADER_H */
