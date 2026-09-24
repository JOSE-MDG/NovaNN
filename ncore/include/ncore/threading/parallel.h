/**
 * @file parallel.h
 * @brief Facade for CPU parallelization decisions.
 *
 * @details
 * Single include for call sites: pattern vocabulary, tunable
 * grains, per-family decisions, and tensor builders.
 *
 * @see parallel/pattern.h  Families, verdict, and work structs.
 * @see parallel/grains.h   Per-thread grains.
 * @see parallel/decide.h   Decision functions.
 * @see parallel/helpers.h  Tensor builders.
 */

#pragma once

#include <ncore/threading/parallel/decide.h>
#include <ncore/threading/parallel/grains.h>
#include <ncore/threading/parallel/helpers.h>
#include <ncore/threading/parallel/pattern.h>
