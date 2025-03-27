#**AI-Odyssey-Labs**
---
# Pre-training NanoGPT with Different Positional Encodings

This project explores the impact of various positional encoding techniques on the performance of a small-scale GPT-2 model (NanoGPT, ~16.03M parameters) when pre-trained on a subset of the OpenWebText dataset.

## Introduction

Positional encodings are crucial for sequence models like Transformers to understand the order of tokens in an input sequence. This project investigates the effectiveness of different positional encoding strategies within the context of a small GPT model. We pre-trained NanoGPT using:

* **Absolute Positional Embeddings (Standard GPT):** The traditional method of adding fixed vectors to token embeddings based on their position.
* **NoPE (No Positional Encoding):** A baseline to observe the model's performance without any explicit positional information.
* **RoPE (Rotary Positional Embeddings):** A technique that applies a rotation matrix to the query and key vectors based on their positional indices.
* **QK Norm:** A modification where the query and key vectors are normalized before attention calculation.
* **RNoPE (Hybrid Attention Strategy):** A combination of different attention mechanisms, the specifics of which are detailed in the code.

The goal is to analyze how these different approaches affect the model's ability to learn language representations, as reflected in the training and validation loss.

## Project Details

* **Model:** NanoGPT (approximately 16.03 million parameters)
* **Training Data:** 10% of the OpenWebText dataset (~900M tokens)
* **Validation Data:** 0.1% of the OpenWebText dataset (~9M tokens)
* **Training Ratio:** Trained on approximately ~56x the model size (~900M tokens / ~16.03M parameters). While the Chinchilla paper suggests training on 20x the model size, this experiment explores the effect of a higher training ratio on this smaller model.
* **Precision:** Used fp16 precision for training. bfloat16 could potentially offer better performance but was not available due to hardware limitations.
* **Optimization:** Utilized Flash Attention for improved training efficiency.
* **Compilation:** Employed `model.compile()`. While its impact is more pronounced for larger models, it was included in the training pipeline.

## Results

The following table summarizes the average training and validation loss achieved by each model:

| Model        | Train Loss | Validation Loss |
|--------------|------------|-----------------|
| GPT          | 4.2067     | 4.1138          |
| GPT NOPE     | 4.4082     | 4.2856          |
| GPT ROPE     | 4.0554     | 3.9634          |
| GPT QK Norm  | 4.3752     | 4.263           |
| GPT RNOPE    | 4.0745     | 3.9817          |

**Observations from the Loss Data:**

* **RoPE and RNoPE** achieved the lowest validation loss, suggesting that these positional encoding methods were most effective for this model and dataset.
* **NoPE** performed the worst, highlighting the importance of positional information for language modeling.
* **QK Norm** also showed a higher loss compared to the standard GPT.
* The standard **GPT** with absolute positional embeddings performed better than NoPE and QK Norm but not as well as RoPE and RNoPE.

## Conclusion

This project demonstrates the impact of different positional encoding techniques on a small GPT model. The results suggest that **Rotary Positional Embeddings (RoPE)** and the **RNoPE** hybrid strategy were particularly beneficial for this setup, leading to lower validation loss. The experiment with NoPE clearly underscores the necessity of providing positional information to the model. While the Multi-Headed Latent Attention (MHLA) experiment was not successful in this context, further investigation with larger models might yield different results.

## Further Exploration

* Experiment with different hyperparameters for each positional encoding method.
* Train on a larger portion of the OpenWebText dataset or a different dataset.
* Investigate the performance of MHLA on larger GPT models.
* Analyze the qualitative differences in the generated text by models trained with different positional encodings.
* Explore other positional encoding techniques not included in this study.


Feel free to contribute to this project by experimenting with different settings or adding new positional encoding implementations 🚀!
