#pragma once
#include <iostream>
#include <emmintrin.h>
#include <immintrin.h>


using namespace std;

/// <summary>
/// H�m t�nh L1 = |x| + |y| d�ng simd SSE (256 bit)
/// </summary>
/// <param name="xt"></param>
/// <param name="yt"></param>
/// <param name="x_i"></param>
/// <param name="y_i"></param>
/// <param name="length"></param>
/// <param name="min_distance"></param>
/// <returns></returns>
float L1DistSSE2(const float& xt, const float& yt, const float* x_i, const float* y_i, const int& length, float min_distance);

/// <summary>
/// H�m t�nh L2 d�ng simd SSE (256 bit)
/// </summary>
/// <param name="xt"></param>
/// <param name="yt"></param>
/// <param name="x_i"></param>
/// <param name="y_i"></param>
/// <param name="length"></param>
/// <param name="min_distance"></param>
/// <returns></returns>
float L2DistSSE2(const float& xt, const float& yt, const float* x_i, const float* y_i, const int& length, float min_distance);


/// <summary>
/// H�m t�nh L2 d�ng simd AVX (512 bit)
/// </summary>
/// <param name="xt"></param>
/// <param name="yt"></param>
/// <param name="x_i"></param>
/// <param name="y_i"></param>
/// <param name="length"></param>
/// <param name="min_distance"></param>
/// <returns></returns>
float L2DistAVX(const float& xt, const float& yt, const float* x_i, const float* y_i, const int& length, float min_distance);