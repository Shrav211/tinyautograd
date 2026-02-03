#!/bin/bash
# benchmarks/run_all.sh

echo "====================================="
echo "Running Complete Benchmark Suite"
echo "====================================="

# Create results directory
mkdir -p benchmarks/results

# Run PyTorch benchmark
echo ""
echo "1/3: Running PyTorch benchmark..."
python -m benchmarks.pytorch_baseline --device cuda --steps 2000 --batch-size 64

# Run TinyAutoGrad benchmark  
echo ""
echo "2/3: Running TinyAutoGrad benchmark..."
python -m benchmarks.tinyautograd_bench --device cuda --steps 2000 --batch-size 64

# Compare results
echo ""
echo "3/3: Generating comparison report..."
python -m benchmarks.compare

echo ""
echo "====================================="
echo "Benchmark suite complete!"
echo "Results saved in benchmarks/results/"
echo "====================================="