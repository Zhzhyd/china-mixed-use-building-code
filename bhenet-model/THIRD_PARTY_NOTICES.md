# Third-party components

The model and sample interfaces use the following components:

- PyTorch, for tensor operations and modules.
- Hugging Face Transformers, for the LLaMA causal language-model implementation.
- Hugging Face PEFT, for low-rank adapters including rank-stabilized LoRA.
- Atom-7B-Chat, as the research text-backbone checkpoint family.
- MobileSAM, as the research image-encoder implementation and checkpoint family.
- NumPy and h5py, for the per-building HDF5 sample interface.

The project code is distributed under the MIT License in `LICENSE`.
The components listed above retain their original licences and weight-use
terms. Obtain permitted local copies and follow the terms of the versions used.
