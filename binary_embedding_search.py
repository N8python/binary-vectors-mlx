import numpy as np
import json
import time
import mlx.core as mx
from mlx_embeddings.utils import load

K = 5
class BinaryEmbeddingSearch:
    """
    A class for binary embedding search operations using Metal acceleration.
    """
    
    # Define Metal kernel for binary similarity computation
    BINARY_SIMILARITY_KERNEL_SOURCE = """
        // Get global thread ID
        uint idx = thread_position_in_grid.x;
        
        // Early return if thread index is out of bounds
        if (idx >= corpus_shape[0]) {
            return;
        }
        
        // Number of uint64 components per embedding
        uint num_components = corpus_shape[1];
        
        // Initialize similarity counter
        uint total_bits = 0;
        
        // Process each component of the embedding vectors
        for (uint i = 0; i < num_components; i++) {
            // Get corpus value (row idx, column i)
            uint64_t corpus_val = corpus[idx * num_components + i];
            
            // Get corresponding query value
            uint64_t query_val = query[i];
            
            // Compute bitwise AND
            uint64_t and_result = corpus_val & query_val;
            
            // Count bits using popcount
            total_bits += metal::popcount(and_result);
        }
        
        // Store the result
        similarities[idx] = (uint16_t)total_bits;
    """

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

    // If position < 5, insert at that position, otherwise overwrite position 4
    //position = min(position, (uint32_t) (k - 1));
    if (position < k) {
    // Store the thread's index in the histogram using atomic operations
        atomic_store_explicit(&hi[sim * k + position], (uint32_t)index, memory_order_relaxed);
    }

    """
    
    def __init__(self, embeddings_path=None, chunks_path=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the BinaryEmbeddingSearch with embeddings, chunks, and model.
        
        Args:
            embeddings_path (str): Path to the embeddings numpy file
            chunks_path (str): Path to the chunks JSON file
            model_name (str): Name of the model to load
        """
        self.binary_similarity_kernel = mx.fast.metal_kernel(
            name="binary_similarity",
            input_names=["corpus", "query"],
            output_names=["similarities"],
            source=self.BINARY_SIMILARITY_KERNEL_SOURCE,
        )

        self.histogram_topk_kernel = mx.fast.metal_kernel(
            name="histogram_top_k",
            input_names=["similarities", "num_elements", "k"],
            output_names=["hi", "counters"],
            source=self.HISTOGRAM_TOPK_KERNEL_SOURCE,
            atomic_outputs=True,
        )
        
        self.embeddings = None
        self.chunks = None
        self.model = None
        self.tokenizer = None
        
        # Load resources if paths are provided
        if embeddings_path and chunks_path:
            self.load_resources(embeddings_path, chunks_path, model_name)
    
    def load_resources(self, embeddings_path, chunks_path, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Load embeddings, chunks, and model resources.
        
        Args:
            embeddings_path (str): Path to the embeddings numpy file
            chunks_path (str): Path to the chunks JSON file
            model_name (str): Name of the model to load
        
        Returns:
            self: For method chaining
        """
        self.embeddings = mx.array(np.load(embeddings_path))
        with open(chunks_path, "r") as f:
            self.chunks = json.load(f)
            
        self.model, self.tokenizer = load(model_name)
        return self
    
    def get_query_embedding(self, query):
        """
        Generate binary embedding for a query string.
        
        Args:
            query (str): The query text
            
        Returns:
            mx.array: The binary embedding as uint64 array
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded before generating embeddings")
            
        inputs = self.tokenizer.batch_encode_plus(
            [query], 
            return_tensors="mlx", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        outputs = self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        # Compute weighted average of token embeddings
        weighted = inputs["attention_mask"][..., None] * outputs[0]
        sum_weighted = mx.sum(weighted, axis=1)
        count = mx.sum(inputs["attention_mask"], axis=1, keepdims=True)
        avg = sum_weighted / count
        
        # Binarize embeddings
        avg = mx.where(avg <= 0, 0, 1)
        
        # Pack binary values into uint64 format
        avg_packed = mx.zeros((avg.shape[0], avg.shape[1]//64), dtype=mx.uint64)
        for i in range(64):
            avg_packed += (avg[:, i::64].astype(mx.uint64) << (64 - i))
            
        return avg_packed
    
    def compute_binary_similarity(self, query_embedding):
        """
        Compute similarity between query embedding and corpus embeddings.
        
        Args:
            query_embedding (mx.array): The query embedding
            
        Returns:
            mx.array: Similarity scores for each corpus embedding
        """
        if not self.embeddings.shape[0]:
            raise ValueError("Corpus embeddings must be loaded before computing similarity")
            
        if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding[0]
            
        num_corpus_embeddings = self.embeddings.shape[0]
        
        outputs = self.binary_similarity_kernel(
            inputs=[self.embeddings, query_embedding],
            grid=(num_corpus_embeddings, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(num_corpus_embeddings,)],
            output_dtypes=[mx.uint16],
        )
        
        return outputs[0]
    
    def search(self, query, k=K, detailed_timing=False):
        """
        Search for the top-k most similar chunks to the query.
        
        Args:
            query (str or mx.array): The query text or embedding
            k (int): Number of top results to return
            detailed_timing (bool): Whether to return detailed timing information
            
        Returns:
            dict: Search results and timing information if requested
        """
        # Process query to get embedding if string is provided
        t_start = time.time()
        
        if isinstance(query, str):
            query_embedding = self.get_query_embedding(query)
        else:
            query_embedding = query
            
        # Start timing for similarity calculation
        t1_start = time.time()
        similarities = self.compute_binary_similarity(query_embedding)
        mx.eval(similarities)
        t1_end = time.time()
        
        # Start timing for top-k selection
        t2_start = time.time()
        num_elements = similarities.shape[0]
        max_similarity = self.embeddings.shape[1] * 64  # Assuming each embedding is packed into uint64

        histogram_indices, counters = self.histogram_topk_kernel(
            inputs=[similarities, num_elements, k],
            grid=(num_elements, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(max_similarity + 1, k), (max_similarity + 1,)],
            output_dtypes=[mx.uint32, mx.uint32],
            init_value=0
        )

        mx.eval(histogram_indices)

        top_indices = []
        needed = k
        
        # Scan histogram from highest score to lowest
        counters = counters.tolist()
        histogram_indices = np.array(histogram_indices)
        for score in range(max_similarity, -1, -1):
            count = int(counters[score])

            if count > 0:
                # Get indices from this bucket
                bucket_indices = histogram_indices[score, :min(count, k)].tolist()
                top_indices.extend(bucket_indices[:needed])
                needed -= len(bucket_indices[:needed])
                if needed <= 0:
                    break
        top_similarities = similarities[mx.array(top_indices)].tolist()
        t2_end = time.time()
        
        # Build results
        t3_start = time.time()
        results = []
        for i, idx in zip(range(k), top_indices):
            results.append({
                "chunk": self.chunks[idx],
                "similarity": top_similarities[i],
                "index": idx
            })
        t3_end = time.time()
        
        # Calculate timings
        if detailed_timing:
            similarity_time = (t1_end - t1_start) * 1000
            topk_time = (t2_end - t2_start) * 1000
            results_time = (t3_end - t3_start) * 1000
            total_time = (t3_end - t_start) * 1000
            
            return {
                "results": results,
                "timings": {
                    "similarity_ms": similarity_time,
                    "topk_ms": topk_time,
                    "results_ms": results_time,
                    "total_ms": total_time
                }
            }
        
        return results
    
    def run_benchmark(self, query="aaaaaaaa", iterations=10):
        """
        Run a benchmark test to measure search performance.
        
        Args:
            query (str): The query text to use for benchmarking
            iterations (int): Number of benchmark iterations
            
        Returns:
            dict: Average timings across all iterations
        """
        print("Generating query embedding...")
        query_embedding = self.get_query_embedding(query)
        
        # Perform warmup
        print("Performing warmup...")
        _ = self.search(query_embedding, detailed_timing=True)
        
        # Run multiple tests and average results
        print("\nRunning benchmark tests...")
        all_timings = {
            "similarity_ms": [], 
            "topk_ms": [], 
            "results_ms": [], 
            "total_ms": []
        }
        
        for i in range(iterations):
            random_query = self.chunks[np.random.randint(0, len(self.chunks))]
            query_embedding = self.get_query_embedding(random_query)
            result = self.search(query_embedding, detailed_timing=True)
            timings = result["timings"]
            
            # Store all timings
            for key in all_timings:
                all_timings[key].append(timings[key])
            
            # Print breakdown for this iteration
            print(f"\nIteration {i+1}:")
            print(f"  Similarity calculation: {timings['similarity_ms']:.2f} ms")
            print(f"  Top-k selection:        {timings['topk_ms']:.2f} ms")
            print(f"  Results creation:       {timings['results_ms']:.2f} ms")
            print(f"  Total search time:      {timings['total_ms']:.2f} ms")

        
        # Calculate averages
        avg_timings = {key: sum(values)/len(values) for key, values in all_timings.items()}
        
        print("\n=== AVERAGE TIMINGS ===")
        print(f"Similarity calculation: {avg_timings['similarity_ms']:.2f} ms")
        print(f"Top-k selection:        {avg_timings['topk_ms']:.2f} ms")
        print(f"Results creation:       {avg_timings['results_ms']:.2f} ms")
        print(f"Total search time:      {avg_timings['total_ms']:.2f} ms")
        
        # Compute MEGACHUNKS searched per second
        avg_timings["megachunks_per_second"] = (len(self.chunks)) / (avg_timings['total_ms'] / 1000)

        print(f"MEGACHUNKS searched per second: {avg_timings['megachunks_per_second'] / 1e6:.2f}")
        return avg_timings


def main():
    """Example usage of the BinaryEmbeddingSearch class"""
    # Initialize and load resources
    searcher = BinaryEmbeddingSearch(
        embeddings_path="embeddings-enwik9.npy",
        chunks_path="chunks-enwik9.json"
    )
    
    # Run benchmark
    searcher.run_benchmark(iterations=100)


if __name__ == "__main__":
    main()