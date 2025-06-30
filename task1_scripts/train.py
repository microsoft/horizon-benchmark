import sys
import csv
import argparse
import os
from recbole.quick_start.quick_start import run

# Increase the CSV field size limit to handle large input fields
csv.field_size_limit(sys.maxsize)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BERT4Rec", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="recbole_dataset_sampled", help="name of datasets"
    )
    parser.add_argument(
        "--port", "-p", type=str, default="54321", help="Port number"
    )
    parser.add_argument(
        "--nproc", "-np", type=int, default=4, help="NProc"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="./config/train_bert4rec.yaml",
        help="Path to YAML config file for RecBole (e.g. train_gru4rec.yaml)",
    )
    args = parser.parse_args()

    # Optional: avoid CUDA fragmentation (important for large models)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Launch multi-GPU training (nproc=4 uses 4 processes = 4 GPUs)
    run(
        model=args.model,
        dataset=args.dataset,
        config_file_list=[args.config],
        nproc=args.nproc,              # number of GPUs you want to use
        world_size=-1,        # RecBole handles this automatically
        ip="localhost",
        port=args.port,          # change if you have a port conflict
        group_offset=0,
    )

if __name__ == "__main__":
    main()
