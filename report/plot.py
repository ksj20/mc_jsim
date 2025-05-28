import matplotlib.pyplot as plt

# 1. Sensitivity bar chart
categories = [
    "Primitive Edits\n(1–4%)",
    "Array Reorders\n(24–40%)",
    "Mixed Edits\nCompact (15%)",
    "Mixed Edits\nLarge (7.5%)",
    "Schema Swap\n(100%)"
]
drop_percent = [2.5, 32, 15, 7.5, 100]

plt.figure(figsize=(6, 4))
plt.bar(categories, drop_percent, color='skyblue')
plt.ylabel("Similarity Drop (%)")
plt.title("MC–JSim Sensitivity Across Edit Types")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig('mcjsim_sensitivity.png')
plt.close()

# 2. MinHash accuracy bar chart
labels = ["Measured Error", "Theoretical Bound"]
values = [0.010, 1/(64**0.5)]  # 1/sqrt(64) ≈ 0.125

plt.figure(figsize=(4, 3))
plt.bar(labels, values, color=['orange','gray'])
plt.ylabel("Error")
plt.title("MinHash Estimation Accuracy")
plt.ylim(0, max(values)*1.1)
plt.tight_layout()
plt.savefig('mcjsim_minhash_accuracy.png')
plt.close()

# # 3. Latency & Throughput chart with projection to 100k nodes
# nodes = [100, 1000, 10000, 100000]
# fingerprint_ms = [0.465, 3.5, 24.2, 242]         # projected for 100k
# sketch_ms      = [31.719, 155.0, 1481.5, 14815]  # projected for 100k
# compare_ms     = [35.088, 162.008, 1411.412, 14112]  # projected for 100k

# plt.figure(figsize=(6, 4))
# plt.plot(nodes, fingerprint_ms, marker='o', label='Fingerprint')
# plt.plot(nodes, sketch_ms, marker='s', label='Sketch')
# plt.plot(nodes, compare_ms, marker='^', label='Compare')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel("Document size (nodes)")
# plt.ylabel("Time (ms) [log scale]")
# plt.title("MC–JSim Latency & Throughput (Log–Log Scale)")
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.savefig('mcjsim_latency_throughput.png')
# plt.close()
