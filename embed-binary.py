from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import mlx.core as mx
from mlx_embeddings.utils import load
import numpy as np
import json
import tqdm

# Load the model and tokenizer
model, tokenizer = load("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding_and_pack(texts, model, tokenizer):
    """
    Get embeddings for texts, binarize them, and immediately pack into uint64 format
    to reduce peak memory consumption
    """
    # Get the embeddings
    inputs = tokenizer.batch_encode_plus(texts, return_tensors="mlx", padding=True, truncation=True, max_length=512)
    outputs = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    weighted = inputs["attention_mask"][..., None] * outputs[0]
    sum = mx.sum(weighted, axis=1)
    count = mx.sum(inputs["attention_mask"], axis=1, keepdims=True)
    avg = sum / count
    
    # Binarize the embeddings
    binary_avg = mx.where(avg <= 0, 0, 1)
    
    # Pack the binary embeddings
    packed_shape = (binary_avg.shape[0], binary_avg.shape[1] // 64)
    embeddings_packed = mx.zeros(packed_shape, dtype=mx.uint64)
    
    for i in range(64):
        embeddings_packed += (binary_avg[:, i::64].astype(mx.uint64) << (64 - i))
    
    # Convert to numpy for accumulation
    return np.array(embeddings_packed.tolist(), dtype=np.uint64)

# Read and chunk the corpus
with open("enwik8.txt", "r") as f:
    chunk_size = 512
    stride_size = 256
    text = f.read()
    chunks = []
    for start in range(0, len(text), stride_size):
        chunks.append(text[start:start+chunk_size])
    print(f"Number of chunks: {len(chunks)}")

# Process in batches and immediately pack
packed_embeddings_list = []
batch_size = 16

for i in tqdm.tqdm(range(0, len(chunks), batch_size)):
    batch = chunks[i:i+batch_size]
    
    # Get embeddings, binarize, and pack in one step
    packed_batch = get_embedding_and_pack(batch, model, tokenizer)
    mx.metal.clear_cache()
    # Append the packed batch to our list
    packed_embeddings_list.append(packed_batch)

# Concatenate the packed batches
packed_embeddings = np.vstack(packed_embeddings_list)
print(f"Packed embeddings shape: {packed_embeddings.shape}")

# Save the packed embeddings to a file
np.save("embeddings-enwik8.npy", packed_embeddings)

# Save the chunks to a file
with open("chunks-enwik8.json", "w") as f:
    json.dump(chunks, f)

print("Done! Saved packed embeddings to embeddings.npy and chunks to chunks.json")