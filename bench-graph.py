import numpy as np
import matplotlib.pyplot as plt
import time
import mlx.core as mx
import random

# Constants from original code
K = 5  # Number of top elements to find
MAX_SIMILARITY = 1024  # Maximum similarity value (uint16 range)

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

    # Warmup kernel
    
    # Test different dataset sizes
    #sizes = [10, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]  # Scaled for performance testing
    sizes = [10, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000, 200_000_000, 500_000_000, 1_000_000_000]  # Scaled for performance testing
    # First size is 1 to ensure the kernel is compiled and ready - its a warmup
    # Arrays to store timing results
    hist_times = []
    argpart_times = []
    
    for size in sizes:
        print(f"\n=== TESTING SIZE {size} ===")
        
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
        hist_times.append(hist_time)
        
        # Time mx.argpartition
        start_time = time.time()
        argpart_top_indices = mx.argpartition(mx_similarities, -K)[-K:].tolist()
        argpart_top_indices.reverse()  # To match order (highest first)
        mx.eval(argpart_top_indices)  # Ensure computation is complete
        argpart_time = (time.time() - start_time) * 1000
        argpart_times.append(argpart_time)
        
        # Print performance results
        print(f"Metal histogram method: {hist_time:.2f} ms")
        print(f"mx.argpartition: {argpart_time:.2f} ms")
        print(f"Speedup: {argpart_time/hist_time:.2f}x")
        mx.metal.clear_cache()  # Clear cache to avoid memory issues
    
    # Get rid of first size 1 results
    hist_times = hist_times[1:]
    argpart_times = argpart_times[1:]
    sizes = sizes[1:]  # Remove the warmup size
    # Create visualizations


    
    # Convert sizes to millions for better display
    sizes_in_millions = [size / 1_000_000 for size in sizes]
    
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
    
    # Calculate speedup
    speedup = [argpart / hist for hist, argpart in zip(hist_times, argpart_times)]
    
    plt.subplot(1, 2, 2)
    plt.bar(sizes_in_millions, speedup, color='green', alpha=0.7)
    plt.title('Speedup Factor (argpartition / histogram)')
    plt.xlabel('Array Size (millions)')
    plt.ylabel('Speedup Factor')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('topk_comparison_results.png', dpi=300)
    
    # 2. Log scale comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.loglog(sizes, hist_times, 'b-o', label='Histogram Method')
    plt.loglog(sizes, argpart_times, 'r-o', label='mx.argpartition')
    plt.title('Performance Comparison (Log Scale)')
    plt.xlabel('Array Size (log scale)')
    plt.ylabel('Execution Time (ms, log scale)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.semilogx(sizes, speedup, 'g-o')
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
    print("Speedup Factors:", speedup)
    
    return sizes, hist_times, argpart_times, speedup

if __name__ == "__main__":
    run_benchmark_with_visualization()