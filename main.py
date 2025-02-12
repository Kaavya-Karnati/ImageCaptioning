import argparse
from train import train
from test import test

def main():
    parser = argparse.ArgumentParser(description="Image Captioning Task")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="train or test the model")
    args = parser.parse_args()

    if args.mode == "train":
        print("Starting training...")
        train()
    elif args.mode == "test":
        print("Starting testing...")
        test()
    else:
        print("Invalid mode. Use --mode train or --mode test.")

if __name__ == "__main__":
    main()
