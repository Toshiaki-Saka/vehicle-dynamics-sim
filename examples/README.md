# examples/

C++ 版の入出力サンプルです。動作確認や `visualize_results.py` を
試す際の参照用に置いてあります。

| ファイル | 内容 |
|---|---|
| `lqr_k.json` | `python/compute_lqr_gain.py` が出力する LQR ゲインの例。`vehicle_dynamics` の入力。 |
| `results.json` | `vehicle_dynamics` が出力するシミュレーション結果の例。`visualize_results.py` の入力。 |

これらは生成物です。自分で再生成する場合は次のようにします。

```bash
python python/compute_lqr_gain.py examples/lqr_k.json
./build/vehicle_dynamics examples/lqr_k.json examples/results.json
python visualize_results.py examples/results.json
```
