/**
 * @file engine.cpp
 * @brief Backward execution: the node scheduler and its fan-out limit.
 *
 * Nothing is implemented yet.
 *
 * The scheduler runs independent nodes in parallel on the threads assigned
 * to @ref ParallelAutogradGroup and decides, per node, whether that is
 * worth doing. A CPU node forks the compute pool, so it is fanned out only
 * while ready nodes times compute threads stays below the logical thread
 * count. The stratified budget bounds the sum of the pool budgets, never
 * that product. A node whose work runs on a device is exempt: it holds a
 * stream, not cores.
 */
