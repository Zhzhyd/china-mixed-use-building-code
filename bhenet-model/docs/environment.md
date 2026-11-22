# Environment

| Component | Requirement |
| --- | --- |
| Python | 3.10 or newer |
| PyTorch | 2.6 ≤ version < 3; CUDA build for GPU training |
| NumPy and h5py | HDF5 sample loading |
| Transformers and SentencePiece | LLaMA model and tokenizer |
| PEFT | RS-LoRA support (`use_rslora=True`) |
| MobileSAM and timm | `vit_t` image encoder |
| pytest | Unit tests |

Install the package and its data/test dependencies:

```bash
python -m pip install -e ".[data,test]"
```

Install a CUDA-compatible PyTorch build and the backbone dependencies in the
same environment. Install MobileSAM from its maintained source. Provide local
Atom-7B-Chat model/tokenizer files and `mobile_sam.pt` in the run configuration.

The research GPU environment used Python 3.10.13, PyTorch 2.7.1+cu128,
Transformers 4.36.2, PEFT 0.11.2.dev0 and timm 1.0.7. CPU tests use small
synthetic inputs; full-backbone training uses a CUDA GPU. The model uses
floating-point backbones and RS-LoRA adapters.
