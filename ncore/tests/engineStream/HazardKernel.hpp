#pragma once

extern "C" void novaTestNullWrite(int *dst, int value, void *stream);

extern "C" void novaTestWriteFloat(float *dst, int n, float value,
                                   void *stream);

extern "C" void novaTestScaleInto(const float *src, float *dst, int n,
                                  float scale, void *stream);
