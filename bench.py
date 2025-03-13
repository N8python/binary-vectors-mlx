import numpy as np
import time
import mlx.core as mx
import random

# Constants
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

def run_verified_scaling_benchmark():
    print("Benchmarking and verifying top-k selection at different scales\n")
    
    # Compile the Metal kernel
    histogram_topk_kernel = mx.fast.metal_kernel(
        name="histogram_top_k",
        input_names=["similarities", "num_elements", "k"],
        output_names=["hist_indices", "counters"],
        source=HISTOGRAM_TOPK_KERNEL_SOURCE,
        atomic_outputs=True,
    )
    
    # Test different dataset sizes
    sizes = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
    
    for size in sizes:
        print(f"\n=== TESTING SIZE {size} ===")
        
        # Generate array with values up to MAX_SIMILARITY - 1
        similarities = np.random.randint(0, MAX_SIMILARITY, size=size, dtype=np.uint16)
        
        # Insert a "needle" with MAX_SIMILARITY at a random position
        needle_index = random.randint(0, size - 1)
        similarities[needle_index] = MAX_SIMILARITY
        
        print(f"Inserted MAX_SIMILARITY at index {needle_index}")
        
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
        
        # Time and verify mx.argpartition
        start_time = time.time()
        argpart_top_indices = mx.argpartition(mx_similarities, -K)[-K:].tolist()
        argpart_top_indices.reverse()  # To match order (highest first)
        mx.eval(argpart_top_indices)  # Ensure computation is complete
        argpart_time = (time.time() - start_time) * 1000
        
        # Get top indices and scores for verification
        hist_top_index = hist_top_indices[0] if hist_top_indices else -1
        argpart_top_index = argpart_top_indices[0] if argpart_top_indices else -1
        
        # Verify both methods found the needle
        hist_correct = hist_top_index == needle_index
        argpart_correct = argpart_top_index == needle_index
        
        # Print performance results
        print(f"Metal histogram method: {hist_time:.2f} ms")
        print(f"mx.argpartition: {argpart_time:.2f} ms")
        print(f"Speedup: {argpart_time/hist_time:.2f}x")
        
        # Print verification results
        print("\nVerification:")
        print(f"Histogram top index: {hist_top_index}, Expected: {needle_index}, Correct: {hist_correct}")
        print(f"Argpartition top index: {argpart_top_index}, Expected: {needle_index}, Correct: {argpart_correct}")
        
        if hist_correct and argpart_correct:
            print("✅ BOTH METHODS VERIFIED CORRECT")
        else:
            print("❌ VERIFICATION FAILED")
            
            # Print more details on failure
            if not hist_correct:
                hist_top_score = similarities[hist_top_index] if hist_top_index >= 0 else -1
                needle_score = similarities[needle_index]
                print(f"Histogram failure details:")
                print(f"  Top index score: {hist_top_score}, Needle score: {needle_score}")
                print(f"  Histogram top indices: {hist_top_indices[:5]}")
                
            if not argpart_correct:
                argpart_top_score = similarities[argpart_top_index] if argpart_top_index >= 0 else -1
                needle_score = similarities[needle_index]
                print(f"Argpartition failure details:")
                print(f"  Top index score: {argpart_top_score}, Needle score: {needle_score}")
                print(f"  Argpartition top indices: {argpart_top_indices[:5]}")
    
    print("\n=== MULTIPLE NEEDLES TEST WITH LARGE ARRAY ===")
    # Test with a large array and multiple maximum values
    size = 10_000_000
    similarities = np.random.randint(0, MAX_SIMILARITY, size=size, dtype=np.uint16)
    
    # Insert multiple needles with MAX_SIMILARITY at random positions
    num_needles = K
    needle_indices = random.sample(range(size), num_needles)
    for idx in needle_indices:
        similarities[idx] = MAX_SIMILARITY
    
    print(f"Array size: {size}")
    print(f"Inserted {num_needles} needles with MAX_SIMILARITY at random positions")
    
    # Convert to MLX array
    mx_similarities = mx.array(similarities)
    
    # Time Metal histogram-based method
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
    
    # Time mx.argpartition
    start_time = time.time()
    argpart_top_indices = mx.argpartition(mx_similarities, -K)[-K:].tolist()
    argpart_top_indices.reverse()  # To match order (highest first)
    mx.eval(argpart_top_indices)  # Ensure computation is complete
    argpart_time = (time.time() - start_time) * 1000
    
    # Verify both methods return only MAX_SIMILARITY values
    hist_top_scores = similarities[hist_top_indices[:K]]
    argpart_top_scores = similarities[argpart_top_indices]
    
    # Check if all returned indices have MAX_SIMILARITY
    hist_all_max = all(score == MAX_SIMILARITY for score in hist_top_scores)
    argpart_all_max = all(score == MAX_SIMILARITY for score in argpart_top_scores)
    
    # Check if returned indices are among the needle indices
    hist_in_needles = all(idx in needle_indices for idx in hist_top_indices[:K])
    argpart_in_needles = all(idx in needle_indices for idx in argpart_top_indices)
    
    # Print performance results
    print(f"\nMetal histogram method: {hist_time:.2f} ms")
    print(f"mx.argpartition: {argpart_time:.2f} ms")
    print(f"Speedup: {argpart_time/hist_time:.2f}x")
    
    # Print verification results
    print("\nVerification:")
    print(f"Histogram all MAX_SIMILARITY: {hist_all_max}")
    print(f"Argpartition all MAX_SIMILARITY: {argpart_all_max}")
    print(f"Histogram indices all in needle set: {hist_in_needles}")
    print(f"Argpartition indices all in needle set: {argpart_in_needles}")
    
    if hist_all_max and argpart_all_max and hist_in_needles and argpart_in_needles:
        print("✅ BOTH METHODS VERIFIED CORRECT WITH MULTIPLE NEEDLES")
    else:
        print("❌ VERIFICATION FAILED WITH MULTIPLE NEEDLES")

if __name__ == "__main__":
    run_verified_scaling_benchmark()