# tiny-mixtral




### Parameter Calculation for MoE Transformer (Mixtral-like)

### 📌 Model Hyperparameters
| Parameter       | Value    |
|----------------|----------|
| `vocab_size`   | 32000    |
| `d_model`      | 512      |
| `d_head`       | 64       |
| `n_heads`      | 8        |
| `n_kv_heads`   | 2        |
| `n_experts`    | 8        |
| `top_k`        | 2        |
| `n_layers`     | 8        |

---

#### 🧠 1. Token Embedding
Each token gets a `d_model`-dim vector.  
Embedding=vocab_size x d_model = 32000 x 512 = 16,384,000

---

#### 🧠 2. Attention Layer (per layer)

#### a. QKV Projections
- **Query**: `512 × 512 = 262,144`
- **Key**: `512 × 128 = 65,536`
- **Value**: `512 × 128 = 65,536`
- **Output Projection** (`W_o`): `512 × 512 = 262,144`  
  
Attention Total = 262,144 + 65,536 + 65,536 + 262,144 = 655,360

#### b. LayerNorms (x2)
Each LayerNorm: `2 × d_model = 1024`

---

#### 🧠 3. MoE FFN (per layer)

Each expert:
- `W1`: `512 × 2048 = 1,048,576`
- `W2`: `2048 × 512 = 1,048,576`
- Biases: `2048 + 512 = 2560`  
  
Per Expert Total = 2,099,712  
Total for 8 Experts = 8 × 2,099,712 = 16,797,696

---

#### 🧠 4. Total Parameters per Transformer Layer
Layer Total = Attention + MoE + LayerNorms  
= 655,360 + 16,797,696 + 1,024  
= 17,454,080  

---

#### 🧠 5. Total for All Transformer Layers
Total = 8 × 17,454,080 = 139,632,640

---

#### 🧠 6. Final Output Layer
Output Head = d_model × vocab_size  
= 512 × 32000  
= 16,384,000  

---

#### ✅ Final Total Parameter Count

| Component              | Count         |
|------------------------|---------------|
| Token Embedding        | 16,384,000    |
| Transformer Layers     | 139,632,640   |
| Output Projection Head | 16,384,000    |

---
Total = 16,384,000 + 139,632,640 + 16,384,000
= `172,400,640 parameters`
#### 🧾 Summary

- **Total Trainable Parameters**: **172.4M**
- **Optimizer**: AdamW (adds states, not parameters)
- **KV Cache & Sliding Window**: Runtime memory only