import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from inference.generator import LamoGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate text with LamoLLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--prompt", type=str, default="Hello, I am")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, mps, or auto")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    print("Loading LamoLLM...")
    generator = LamoGenerator.from_checkpoint(args.checkpoint, device=args.device)

    if args.interactive:
        print("Interactive mode (type 'quit' to exit)")
        history = []
        while True:
            prompt = input("\nUser: ")
            if prompt.lower() == 'quit':
                break

            response = generator.chat(prompt, history)
            print(f"\nAssistant: {response}")
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": response})
    else:
        output = generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print(f"\n{args.prompt}{output}")


if __name__ == "__main__":
    main()
