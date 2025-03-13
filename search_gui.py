import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import os
from binary_embedding_search import BinaryEmbeddingSearch

class Enwik9SearchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Configure the main window
        self.title("Enwik9 Binary Embedding Search")
        self.geometry("900x700")
        
        # Initialize default paths
        self.embeddings_path = "embeddings-enwik9.npy"
        self.chunks_path = "chunks-enwik9.json"
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Initialize flags and searcher
        self.searching = False
        self.searcher = None
        
        # Setup GUI components
        self.create_widgets()
    
    def create_widgets(self):
        """Create and arrange all GUI elements"""
        # Create main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Configuration")
        config_frame.pack(fill=tk.X, pady=5)
        
        # Embeddings path
        ttk.Label(config_frame, text="Embeddings Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.embed_path_var = tk.StringVar(value=self.embeddings_path)
        ttk.Entry(config_frame, textvariable=self.embed_path_var, width=50).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        ttk.Button(config_frame, text="Browse", command=lambda: self.browse_file(self.embed_path_var)).grid(row=0, column=2, padx=5, pady=2)
        
        # Chunks path
        ttk.Label(config_frame, text="Chunks Path:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.chunks_path_var = tk.StringVar(value=self.chunks_path)
        ttk.Entry(config_frame, textvariable=self.chunks_path_var, width=50).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        ttk.Button(config_frame, text="Browse", command=lambda: self.browse_file(self.chunks_path_var)).grid(row=1, column=2, padx=5, pady=2)
        
        # Model name
        ttk.Label(config_frame, text="Model Name:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.model_name_var = tk.StringVar(value=self.model_name)
        ttk.Entry(config_frame, textvariable=self.model_name_var, width=50).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
        
        # Initialize button
        ttk.Button(config_frame, text="Initialize Searcher", command=self.initialize_searcher).grid(row=2, column=2, padx=5, pady=2)
        
        # Configure grid column weights
        config_frame.columnconfigure(1, weight=1)
        
        # Search input area
        input_frame = ttk.LabelFrame(main_frame, text="Search Query")
        input_frame.pack(fill=tk.X, pady=5)
        
        self.query_entry = ttk.Entry(input_frame, width=70, font=('Arial', 11))
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.query_entry.bind("<Return>", self.search)
        
        self.search_button = ttk.Button(input_frame, text="Search", command=self.search)
        self.search_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.search_button.config(state=tk.DISABLED)  # Disabled until searcher is initialized
        
        # Status area
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT, padx=5)
        self.status_var = tk.StringVar(value="Not initialized")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        self.progress.pack(side=tk.RIGHT, padx=5)
        
        # Timing information area
        timing_frame = ttk.LabelFrame(main_frame, text="Timing Information")
        timing_frame.pack(fill=tk.X, pady=5)
        
        # Timing labels
        self.timing_vars = {
            "similarity_ms": tk.StringVar(value="0.00 ms"),
            "topk_ms": tk.StringVar(value="0.00 ms"),
            "results_ms": tk.StringVar(value="0.00 ms"),
            "total_ms": tk.StringVar(value="0.00 ms")
        }
        
        # Create two columns for timing info
        col_frame1 = ttk.Frame(timing_frame)
        col_frame1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        col_frame2 = ttk.Frame(timing_frame)
        col_frame2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        ttk.Label(col_frame1, text="Similarity calculation:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(col_frame1, textvariable=self.timing_vars["similarity_ms"]).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(col_frame1, text="Top-k selection:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(col_frame1, textvariable=self.timing_vars["topk_ms"]).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(col_frame2, text="Results creation:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(col_frame2, textvariable=self.timing_vars["results_ms"]).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(col_frame2, text="Total search time:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(col_frame2, textvariable=self.timing_vars["total_ms"]).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Search Results (Top 5)")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a text widget to display the results with scrollbar
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=20, font=('Courier New', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text.config(state=tk.DISABLED)
    
    def browse_file(self, string_var):
        """Open a file browser dialog and set the selected path to a StringVar"""
        file_path = filedialog.askopenfilename()
        if file_path:
            string_var.set(file_path)
    
    def initialize_searcher(self):
        """Initialize the binary embedding searcher in a separate thread"""
        # Get path values from UI
        self.embeddings_path = self.embed_path_var.get()
        self.chunks_path = self.chunks_path_var.get()
        self.model_name = self.model_name_var.get()
        
        # Validate paths
        if not os.path.exists(self.embeddings_path):
            messagebox.showerror("Error", f"Embeddings file not found: {self.embeddings_path}")
            return
            
        if not os.path.exists(self.chunks_path):
            messagebox.showerror("Error", f"Chunks file not found: {self.chunks_path}")
            return
        
        # Update status and start progress
        self.update_status("Initializing searcher...", "blue")
        self.progress.start()
        
        def init_thread():
            try:
                # Create the searcher
                self.searcher = BinaryEmbeddingSearch(
                    embeddings_path=self.embeddings_path,
                    chunks_path=self.chunks_path,
                    model_name=self.model_name
                )
                self.update_status("Ready", "green")
                self.search_button.config(state=tk.NORMAL)
            except Exception as e:
                self.update_status(f"Error initializing searcher", "red")
                messagebox.showerror("Initialization Error", str(e))
                self.search_button.config(state=tk.DISABLED)
            finally:
                # Stop progress
                self.progress.stop()
        
        # Start the initialization in a separate thread
        threading.Thread(target=init_thread, daemon=True).start()
    
    def search(self, event=None):
        """Perform the search operation"""
        # Check if searcher is initialized
        if not self.searcher:
            messagebox.showwarning("Warning", "Searcher is not initialized. Please initialize it first.")
            return
        
        # If already searching, ignore
        if self.searching:
            return
        
        # Get the query
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a search query.")
            return
        
        # Update status and disable search
        self.searching = True
        self.search_button.config(state=tk.DISABLED)
        self.update_status("Searching...", "blue")
        self.progress.start()
        
        # Clear previous results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        
        # Reset timing information
        for key in self.timing_vars:
            self.timing_vars[key].set("0.00 ms")
        
        # Perform search in a separate thread
        def search_thread():
            try:
                # Perform the search with detailed timing
                results = self.searcher.search(query, k=5, detailed_timing=True)
                
                # Update the UI with results
                self.update_results(results)
                self.update_status("Search completed", "green")
            except Exception as e:
                self.update_status("Error during search", "red")
                messagebox.showerror("Search Error", str(e))
            finally:
                # Enable search again
                self.searching = False
                self.search_button.config(state=tk.NORMAL)
                self.progress.stop()
        
        # Start the search in a separate thread
        threading.Thread(target=search_thread, daemon=True).start()
    
    def update_results(self, results_data):
        """Update the results display with the search results"""
        # Check the structure of the results
        if isinstance(results_data, dict) and "results" in results_data:
            # Detailed timing format
            search_results = results_data["results"]
            timings = results_data["timings"]
            
            # Update timing information
            for key, value in timings.items():
                if key in self.timing_vars:
                    self.timing_vars[key].set(f"{value:.2f} ms")
        else:
            # Direct results format
            search_results = results_data
        
        # Format and display results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)  # Clear existing text
        
        for i, result in enumerate(search_results):
            # Format the result
            self.results_text.insert(tk.END, f"Result #{i+1}\n", "title")
            self.results_text.insert(tk.END, f"Similarity: {result['similarity']}, Index: {result['index']}\n", "info")
            self.results_text.insert(tk.END, "-" * 80 + "\n", "separator")
            self.results_text.insert(tk.END, f"{result['chunk']}\n\n", "content")
        
        # Configure tags for styling
        self.results_text.tag_configure("title", font=("Arial", 12, "bold"))
        self.results_text.tag_configure("info", font=("Arial", 10, "italic"), foreground="blue")
        self.results_text.tag_configure("separator", foreground="gray")
        self.results_text.tag_configure("content", font=("Courier New", 10))
        
        self.results_text.config(state=tk.DISABLED)
    
    def update_status(self, message, color="black"):
        """Update the status message with color"""
        self.status_var.set(message)
        
        # Map color names to tkinter color strings
        color_map = {
            "red": "#ff0000",
            "green": "#008000",
            "blue": "#0000ff",
            "black": "#000000"
        }
        
        # Update the label's foreground color
        self.status_label.config(foreground=color_map.get(color, "black"))

if __name__ == "__main__":
    app = Enwik9SearchGUI()
    app.mainloop()