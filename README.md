# nano-llama31-mlx

MLX port of [Andrej Karpathy's nano-llama31](https://github.com/karpathy/nano-llama31). The original project is to Llama 3.1 what nanoGPT is to GPT-2 - a minimal, dependency-light implementation of the Llama 3.1 architecture. This MLX version aims to maintain that spirit while leveraging the capabilities of Apple's MLX framework.

Like the original, this code focuses on the 8B base model of Llama 3.1.

### Key Differences from Karpathy's Original nano-llama31

- Replaced PyTorch with MLX for all tensor operations and neural network modules
- Removed CUDA-specific optimizations (e.g., flash attention) as MLX handles optimizations differently

### Usage

1. Clone this forked repository:

    ```bash
    git clone https://github.com/your-username/nano-llama31-mlx.git
    cd nano-llama31-mlx
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements
    ```

3. Set up Hugging Face access:
    - Create a Hugging Face account if you don't have one: [Hugging Face](https://huggingface.co/join)
    - Generate an access token: [Hugging Face Tokens](https://huggingface.co/settings/tokens)
    - Set the token as an environment variable:

        ```bash
        export HF_JOSEF_TOKEN='your_token_here'
        ```

4. Request access to the Llama 3.1 model:
   - Go to the [Llama 3.1 8B model page on Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
   - Fill out the form to request access to Llama 3.1
   - Wait for approval (this may take some time)

5. Run the inference script:

    ```bash
    python llama31.py
    ```

6. Run the test script:

    ```bash
    python test_llama31.py
    ```

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request. When contributing, please keep in mind that this is a fork aimed at translating the original work to MLX, so major architectural changes should be carefully considered.

### Acknowledgements

This project is a fork of [Andrej Karpathy's nano-llama31](https://github.com/karpathy/nano-llama31). We are deeply grateful to Andrej Karpathy and all contributors to the original project for their absolutely fantastic work, which made this MLX adaptation possible.

### Disclaimer

This is an unofficial, community-driven fork. For the most up-to-date and original implementation, please refer to Karpathy's repository.
