# Simulation: Real-Time Image Generation Based on Viewer Positions
**Project: The Most Polish Landscape**

### Key Findings

Based on the presented results, there is a clear performance difference between PNG and BMP formats:

#### PNG:
- **Total load time**: ~0.22–0.23s  
- **Merge time**: ~0.005–0.007s  
- **Save time**: ~0.044–0.046s  
- **Total processing time**: ~0.27–0.28s  

#### BMP:
- **Total load time**: ~0.063–0.066s  
- **Merge time**: ~0.003–0.004s  
- **Save time**: ~0.009–0.018s  
- **Total processing time**: ~0.077–0.088s  

---

### Main Insights:
1. **BMP is approximately 3.5x faster** in total processing time.
2. The most significant difference is observed in **load time**:  
   - PNG: ~0.22s  
   - BMP: ~0.065s  
   - **BMP is roughly 3.4x faster**.
3. **Save time** also highlights a notable difference:  
   - PNG: ~0.045s  
   - BMP: ~0.009–0.017s  
   - **BMP is 2.6–5x faster**.

---

### Conclusion
BMP demonstrates significantly better performance than PNG, particularly in load and save operations, making it a more efficient format for scenarios requiring high-speed processing.