/**
 * MIT License
 *
 * Copyright 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*!\file
 * \brief tensile_xfloat32.h provides struct for tensile_xfloat32 typedef
 */

#pragma once
#ifndef _TENSILE_XFLOAT32_H_
#define _TENSILE_XFLOAT32_H_

// If this is a C compiler, C++ compiler below C++11, or a host-only compiler,
// we only include a minimal definition of tensile_xfloat32
#if __cplusplus < 201103L || !defined(__HIPCC__)

#include <stdint.h>
typedef struct
{
    uint32_t data;
} tensile_xfloat32;

#else // __cplusplus < 201103L || !defined(__HIPCC__)

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <ostream>
#include <type_traits>

struct tensile_xfloat32
{
    uint32_t data;

    // Don't initialize `data` in purpose so that it could be used with
    // `__shared__`, which forbid any initializer.
    __host__ __device__ tensile_xfloat32() {}

    // round upper 18 bits of IEEE float to convert to xfloat32
    explicit __host__ __device__ tensile_xfloat32(float f)
        : data(float_to_xfloat32(f))
    {
    }

    __host__ __device__ operator float() const
    {
        union
        {
            uint32_t int32;
            float    fp32;
        } u = {data};
        return u.fp32;
    }

private:
    static __host__ __device__ uint32_t float_to_xfloat32(float f)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {f};
        if(~u.int32 & 0x7f800000)
        {
            // When the exponent bits are not all 1s, then the value is zero, normal,
            // or subnormal. We round the xfloat32 mantissa up by adding 0x1FFF, plus
            // 1 if the least significant bit of the xfloat32 mantissa is 1 (odd).
            // This causes the xfloat32's mantissa to be incremented by 1 if the 14
            // least significant bits of the float mantissa are greater than 0x2000,
            // or if they are equal to 0x2000 and the least significant bit of the
            // xfloat32 mantissa is 1 (odd). This causes it to be rounded to even when
            // the lower 14 bits are exactly 0x2000. If the xfloat32 mantissa already
            // has the value 0x1ff, then incrementing it causes it to become 0x00 and
            // the exponent is incremented by one, which is the next higher FP value
            // to the unrounded xfloat32 value. When the xfloat32 value is subnormal
            // with an exponent of 0x00 and a mantissa of 0x1FF, it may be rounded up
            // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
            // When the xfloat32 value has an exponent of 0xFE and a mantissa of 0x1FF,
            // incrementing it causes it to become an exponent of 0xFF and a mantissa
            // of 0x00, which is Inf, the next higher value to the unrounded value.
            u.int32 += 0x1fff + ((u.int32 >> 14) & 1); // Round to nearest, round to even
        }
        else if(u.int32 & 0x3fff)
        {
            // When all of the exponent bits are 1, the value is Inf or NaN.
            // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
            // mantissa bit. Quiet NaN is indicated by the most significant mantissa
            // bit being 1. Signaling NaN is indicated by the most significant
            // mantissa bit being 0 but some other bit(s) being 1. If any of the
            // lower 14 bits of the mantissa are 1, we set the least significant bit
            // of the xfloat32 mantissa, in order to preserve signaling NaN in case
            // the xfloat32's mantissa bits are all 0.
            u.int32 |= 0x4000; // Preserve signaling NaN
        }
        return u.int32 & 0xffffc000;
    }
};

static_assert(std::is_standard_layout<tensile_xfloat32>{},
              "tensile_xfloat32 is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivially_copyable<tensile_xfloat32>{},
              "tensile_xfloat32 is not trivially copyable, and thus is "
              "incompatible with C.");

inline std::ostream& operator<<(std::ostream& os, const tensile_xfloat32& xf32)
{
    return os << float(xf32);
}
inline __host__ __device__ tensile_xfloat32 operator+(tensile_xfloat32 a)
{
    return a;
}
inline __host__ __device__ tensile_xfloat32 operator-(tensile_xfloat32 a)
{
    a.data ^= 0x80000000;
    return a;
}
inline __host__ __device__ tensile_xfloat32 operator+(tensile_xfloat32 a, tensile_xfloat32 b)
{
    return tensile_xfloat32(float(a) + float(b));
}
inline __host__ __device__ tensile_xfloat32 operator+(int a, tensile_xfloat32 b)
{
    return static_cast<tensile_xfloat32>(static_cast<float>(a) + static_cast<float>(b));
}
inline __host__ __device__ tensile_xfloat32 operator+(tensile_xfloat32 a, int b)
{
    return static_cast<tensile_xfloat32>(static_cast<float>(a) + static_cast<float>(b));
}
inline __host__ __device__ tensile_xfloat32 operator-(tensile_xfloat32 a, tensile_xfloat32 b)
{
    return tensile_xfloat32(float(a) - float(b));
}
inline __host__ __device__ tensile_xfloat32 operator*(tensile_xfloat32 a, tensile_xfloat32 b)
{
    return tensile_xfloat32(float(a) * float(b));
}
inline __host__ __device__ tensile_xfloat32 operator/(tensile_xfloat32 a, tensile_xfloat32 b)
{
    return tensile_xfloat32(float(a) / float(b));
}
inline __host__ __device__ bool operator<(tensile_xfloat32 a, tensile_xfloat32 b)
{
    return float(a) < float(b);
}
inline __host__ __device__ bool operator==(tensile_xfloat32 a, tensile_xfloat32 b)
{
    return float(a) == float(b);
}
inline __host__ __device__ bool operator>(tensile_xfloat32 a, tensile_xfloat32 b)
{
    return b < a;
}
inline __host__ __device__ bool operator<=(tensile_xfloat32 a, tensile_xfloat32 b)
{
    return !(a > b);
}
inline __host__ __device__ bool operator!=(tensile_xfloat32 a, tensile_xfloat32 b)
{
    return !(a == b);
}
inline __host__ __device__ bool operator>=(tensile_xfloat32 a, tensile_xfloat32 b)
{
    return !(a < b);
}
inline __host__ __device__ tensile_xfloat32& operator+=(tensile_xfloat32& a, tensile_xfloat32 b)
{
    return a = a + b;
}
inline __host__ __device__ tensile_xfloat32& operator-=(tensile_xfloat32& a, tensile_xfloat32 b)
{
    return a = a - b;
}
inline __host__ __device__ tensile_xfloat32& operator*=(tensile_xfloat32& a, tensile_xfloat32 b)
{
    return a = a * b;
}
inline __host__ __device__ tensile_xfloat32& operator/=(tensile_xfloat32& a, tensile_xfloat32 b)
{
    return a = a / b;
}
inline __host__ __device__ tensile_xfloat32& operator++(tensile_xfloat32& a)
{
    return a += tensile_xfloat32(1.0f);
}
inline __host__ __device__ tensile_xfloat32& operator--(tensile_xfloat32& a)
{
    return a -= tensile_xfloat32(1.0f);
}
inline __host__ __device__ tensile_xfloat32 operator++(tensile_xfloat32& a, int)
{
    tensile_xfloat32 orig = a;
    ++a;
    return orig;
}
inline __host__ __device__ tensile_xfloat32 operator--(tensile_xfloat32& a, int)
{
    tensile_xfloat32 orig = a;
    --a;
    return orig;
}
inline __host__ __device__ bool isinf(tensile_xfloat32 a)
{
    return !(~a.data & 0x7f800000) && !(a.data & 0x7fc000);
}
inline __host__ __device__ bool isnan(tensile_xfloat32 a)
{
    return !(~a.data & 0x7f800000) && +(a.data & 0x7fc000);
}
inline __host__ __device__ bool iszero(tensile_xfloat32 a)
{
    return !(a.data & 0x7fffc000);
}
inline tensile_xfloat32 sin(tensile_xfloat32 a)
{
    return tensile_xfloat32(sinf(float(a)));
}
inline tensile_xfloat32 cos(tensile_xfloat32 a)
{
    return tensile_xfloat32(cosf(float(a)));
}

#endif // __cplusplus < 201103L || !defined(__HIPCC__)

#endif // _TENSILE_XFLOAT32_H_
