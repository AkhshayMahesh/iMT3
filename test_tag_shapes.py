import torch
from transformers import T5Config
from models.t5_segmem_v2_with_prev import T5SegMemV2WithPrev
import sys

def test_tag():
    # Mock config
    config = T5Config(
        vocab_size=128,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    # Enable TAG
    config.use_tag = True
    config.num_instruments = 10
    
    # Initialize the model
    print("Initializing model...")
    model = T5SegMemV2WithPrev(config=config, segmem_num_layers=1, segmem_length=10)
    
    # Create dummy inputs
    B = 2
    T_enc = 20
    T_dec = 15
    
    # In MT3, inputs are typically spectrograms or embeddings, let's use the `proj` size which is `d_model`
    # However `self.proj` maps from `d_model` to `d_model`. So `inputs` must have `d_model` features.
    inputs = torch.randn(B, T_enc, config.d_model)
    labels = torch.randint(0, config.vocab_size, (B, T_dec))
    
    # Mock `num_insts`
    num_insts = torch.tensor([3, 7], dtype=torch.long)
    
    # Keep gradients for param
    model.train()
    
    print("Running forward pass...")
    outputs = model(
        inputs=inputs, 
        labels=labels, 
        num_insts=num_insts, 
        return_dict=True
    )
        
    print(f"Logits shape: {outputs.logits.shape}")
    assert outputs.logits.shape == (B, T_dec, config.vocab_size), "Unexpected logits shape!"
    
    print("Running backward pass...")
    loss = outputs.logits.sum()
    try:
        loss.backward()
    except Exception as e:
        print(f"Backward pass failed! Error: {e}")
        sys.exit(1)
        
    # Check if SaliencyHead got gradients
    grad = model.saliency_head.proj.weight.grad
    if grad is None:
        print("FAIL: SaliencyHead did not receive gradients. Gating mechanism detached from graph?")
        sys.exit(1)
    
    print(f"SaliencyHead grad norm: {grad.norm().item():.4f}")
    if grad.norm().item() == 0.0:
        print("FAIL: SaliencyHead gradient is perfectly zero.")
        sys.exit(1)
        
    print("SUCCESS: TAG Mechanism verification passed!")

if __name__ == "__main__":
    test_tag()
