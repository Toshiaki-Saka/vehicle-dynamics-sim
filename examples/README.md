# examples/

Sample input/output for the C++ build. These are provided as a reference
for sanity checks and for trying out `visualize_results.py`.

| File | Contents |
|---|---|
| `lqr_k.json` | Example LQR gain produced by `python/compute_lqr_gain.py`. Input to `vehicle_dynamics`. |
| `results.json` | Example simulation output produced by `vehicle_dynamics`. Input to `visualize_results.py`. |

These are generated artifacts. To regenerate them yourself:

```bash
python python/compute_lqr_gain.py examples/lqr_k.json
./build/vehicle_dynamics examples/lqr_k.json examples/results.json
python visualize_results.py examples/results.json
```
