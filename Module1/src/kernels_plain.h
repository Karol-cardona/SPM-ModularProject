#pragma once

#include "common.h"

// MASK: key & (P-1)
static void kernel_mask(const pkey_t* __restrict__ keys, partid_t* __restrict__ out, size_t N, uint32_t P)
{
    const uint64_t mask = P - 1;
    for (size_t i = 0; i < N; i++) {
        out[i] = static_cast<partid_t>(keys[i] & mask);
    }
}


// MULTIPLY_SHIFT: Dietzfelbinger universal hashing
static void kernel_multiply_shift(const pkey_t* __restrict__ keys, partid_t* __restrict__ out, size_t N, uint32_t P)
{
    const int shift = 64 - log2_pot(P);
    for (size_t i = 0; i < N; i++) {
        out[i] = static_cast<partid_t>((keys[i] * MULT_PRIME) >> shift);
    }
}


// MURMUR fmix64
static void kernel_murmur(const pkey_t* __restrict__ keys, partid_t* __restrict__ out, size_t N, uint32_t P)
{
    const uint64_t mask = P - 1;
    for (size_t i = 0; i < N; i++) {
        uint64_t k = keys[i];
        k ^= k >> 33; k *= MURMUR_C1;
        k ^= k >> 33; k *= MURMUR_C2;
        k ^= k >> 33;
        out[i] = static_cast<partid_t>(k & mask);
    }
}


// XOR_SHIFT: lightweight bit mixing
static void kernel_xor_shift(const pkey_t* __restrict__ keys, partid_t* __restrict__ out, size_t N, uint32_t P)
{
    const uint64_t mask = P - 1;
    for (size_t i = 0; i < N; i++) {
        uint64_t k = keys[i];
        k ^= k >> 33;
        out[i] = static_cast<partid_t>(k & mask);
    }
}

// FOLD32_MULT
static void kernel_fold32_mult(const pkey_t* __restrict__ keys, partid_t* __restrict__ out, size_t N, uint32_t P)
{
    const int shift = 32 - log2_pot(P);
    for (size_t i = 0; i < N; i++) {
        uint32_t k = static_cast<uint32_t>(keys[i] ^ (keys[i] >> 32));
        out[i] = static_cast<partid_t>((k * MULT_PRIME32) >> shift);
    }
}


// Dispatch by HashType
typedef void (*kernel_fn_t)(const pkey_t*, partid_t*, size_t, uint32_t);

static kernel_fn_t get_kernel(HashType h) {
    switch (h) {
        case HashType::MASK:            return kernel_mask;
        case HashType::MULTIPLY_SHIFT:  return kernel_multiply_shift;
        case HashType::MURMUR_FMIX64:   return kernel_murmur;
        case HashType::XOR_SHIFT:       return kernel_xor_shift;
        case HashType::FOLD32_MULT:     return kernel_fold32_mult;
        default:                        return nullptr;
    }
}
