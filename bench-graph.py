import numpy as np
import matplotlib.pyplot as plt
import time
import mlx.core as mx
import random
import math

# Constants
K = 5  # Number of top elements to find
MAX_SIMILARITY = 1024  # Maximum similarity value (uint16 range)
TARGET_SIZE = 100_000_000  # Target total elements to process

# Define Metal kernel for histogram-based top-k
HISTOGRAM_TOPK_KERNEL_SOURCE = """
    // Get global thread ID
    uint tid = thread_position_in_grid.x;
    // Check if this thread is within bounds
    if (tid >= num_elements) {
        return;
    }
    
    // Get the similarity score for this thread
    uint16_t sim = similarities[tid];

    // Get the actual index this thread represents
    uint32_t index = tid;

    // Atomically increment the counter for this similarity score
    uint32_t position = atomic_fetch_add_explicit(&counters[sim], 1, memory_order_relaxed);

    // If position < k, insert at that position
    if (position < k) {
        // Store the thread's index in the histogram using atomic operations
        atomic_store_explicit(&hist_indices[sim * k + position], (uint32_t)index, memory_order_relaxed);
    }
"""

def run_benchmark_with_visualization():
    print("Benchmarking and visualizing top-k selection at different scales\n")
    
    # Compile the Metal kernel
    histogram_topk_kernel = mx.fast.metal_kernel(
        name="histogram_top_k",
        input_names=["similarities", "num_elements", "k"],
        output_names=["hist_indices", "counters"],
        source=HISTOGRAM_TOPK_KERNEL_SOURCE,
        atomic_outputs=True,
    )

    # Warmup kernel with small array first
    warmup_size = 10
    warmup_similarities = np.random.randint(0, MAX_SIMILARITY, size=warmup_size, dtype=np.uint16)
    mx_warmup_similarities = mx.array(warmup_similarities)
    
    # Execute warmup
    histogram_indices, counters = histogram_topk_kernel(
        inputs=[mx_warmup_similarities, warmup_size, K],
        grid=(warmup_size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(MAX_SIMILARITY + 1, K), (MAX_SIMILARITY + 1,)],
        output_dtypes=[mx.uint32, mx.uint32],
        init_value=0
    )
    mx.eval(histogram_indices, counters)  # Ensure kernel is compiled
    
    # Test different dataset sizes
    sizes = [100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 
             10_000_000, 20_000_000, 50_000_000, 100_000_000, 200_000_000, 
             500_000_000, 1_000_000_000]
    
    # Arrays to store timing results
    hist_times = []
    argpart_times = []
    speedup_factors = []
    effective_sizes = []  # To store the actual tested sizes
    
    for size in sizes:
        # For arrays larger than TARGET_SIZE, run at least 10 times
        # For smaller arrays, repeat to process ~TARGET_SIZE elements
        iterations = max(10, math.ceil(TARGET_SIZE / size))
        total_elements = size * iterations
        
        # Only add the size to our results if we're going to run the test
        # (skip extremely large sizes if needed)
        try:
            print(f"\n=== TESTING SIZE {size:,} with {iterations} iterations ===")
            print(f"Total elements processed: {total_elements:,}")
            
            # Track cumulative times
            total_hist_time = 0
            total_argpart_time = 0
            
            for iteration in range(iterations):
                if iterations > 1 and iteration % 10 == 0:
                    print(f"  Running iteration {iteration}/{iterations}...")
                
                # Generate array with values up to MAX_SIMILARITY - 1
                similarities = np.random.randint(0, MAX_SIMILARITY, size=size, dtype=np.uint16)
                
                # Insert a "needle" with MAX_SIMILARITY at a random position
                needle_index = random.randint(0, size - 1)
                similarities[needle_index] = MAX_SIMILARITY
                
                # Convert to MLX array
                mx_similarities = mx.array(similarities)
                
                # Time and verify Metal histogram-based method
                start_time = time.time()
                
                # Execute kernel
                histogram_indices, counters = histogram_topk_kernel(
                    inputs=[mx_similarities, size, K],
                    grid=(size, 1, 1),
                    threadgroup=(256, 1, 1),
                    output_shapes=[(MAX_SIMILARITY + 1, K), (MAX_SIMILARITY + 1,)],
                    output_dtypes=[mx.uint32, mx.uint32],
                    init_value=0
                )
                
                # Evaluate to complete computation
                mx.eval(histogram_indices, counters)
                
                # Extract top-k indices from histogram
                hist_top_indices = []
                needed = K
                
                # Scan histogram from highest score to lowest
                counters_np = np.array(counters)
                histogram_indices_np = np.array(histogram_indices)
                
                for score in range(MAX_SIMILARITY, -1, -1):
                    count = int(counters_np[score])
                    if count > 0:
                        # Get indices from this bucket
                        bucket_indices = histogram_indices_np[score, :min(count, K)].tolist()
                        hist_top_indices.extend(bucket_indices[:needed])
                        needed -= len(bucket_indices[:needed])
                        if needed <= 0:
                            break
                
                hist_time = (time.time() - start_time) * 1000
                total_hist_time += hist_time
                
                # Time mx.argpartition
                start_time = time.time()
                argpart_top_indices = mx.argpartition(mx_similarities, -K)[-K:].tolist()
                argpart_top_indices.reverse()  # To match order (highest first)
                mx.eval(argpart_top_indices)  # Ensure computation is complete
                argpart_time = (time.time() - start_time) * 1000
                total_argpart_time += argpart_time
            
            # Calculate averages
            avg_hist_time = total_hist_time / iterations
            avg_argpart_time = total_argpart_time / iterations
            avg_speedup = avg_argpart_time / avg_hist_time
                
            # Store results
            hist_times.append(avg_hist_time)
            argpart_times.append(avg_argpart_time)
            speedup_factors.append(avg_speedup)
            effective_sizes.append(size)
                
            # Print performance results
            print(f"Metal histogram method: {avg_hist_time:.2f} ms (average)")
            print(f"mx.argpartition: {avg_argpart_time:.2f} ms (average)")
            print(f"Speedup: {avg_speedup:.2f}x")
            
            # Clear cache to avoid memory issues
            mx.metal.clear_cache()
            
        except Exception as e:
            print(f"Error processing size {size}: {e}")
            # Skip this size and continue with the next
            continue
    
    # Create visualizations
    # Convert sizes to millions for better display
    sizes_in_millions = [size / 1_000_000 for size in effective_sizes]
    
    # 1. Line chart comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(sizes_in_millions, hist_times, 'b-o', label='Histogram Method')
    plt.plot(sizes_in_millions, argpart_times, 'r-o', label='mx.argpartition')
    plt.title('Performance Comparison: Execution Time')
    plt.xlabel('Array Size (millions)')
    plt.ylabel('Execution Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(sizes_in_millions, speedup_factors, color='green', alpha=0.7)
    plt.title('Speedup Factor (argpartition / histogram)')
    plt.xlabel('Array Size (millions)')
    plt.ylabel('Speedup Factor')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('topk_comparison_results.png', dpi=300)
    
    # 2. Log scale comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.loglog(effective_sizes, hist_times, 'b-o', label='Histogram Method')
    plt.loglog(effective_sizes, argpart_times, 'r-o', label='mx.argpartition')
    plt.title('Performance Comparison (Log Scale)')
    plt.xlabel('Array Size (log scale)')
    plt.ylabel('Execution Time (ms, log scale)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.semilogx(effective_sizes, speedup_factors, 'g-o')
    plt.title('Speedup Factor vs Array Size (Log Scale)')
    plt.xlabel('Array Size (log scale)')
    plt.ylabel('Speedup Factor')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('topk_comparison_log_scale.png', dpi=300)
    
    # Print final results summary
    print("\n=== RESULTS SUMMARY ===")
    print("Array Sizes (millions):", sizes_in_millions)
    print("Histogram Method Times (ms):", hist_times)
    print("mx.argpartition Times (ms):", argpart_times)
    print("Speedup Factors:", speedup_factors)
    
    return effective_sizes, hist_times, argpart_times, speedup_factors

if __name__ == "__main__":
    run_benchmark_with_visualization()