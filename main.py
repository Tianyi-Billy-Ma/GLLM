from src import load_data, load_emb
from arguments import parse_args


if __name__ == "__main__":
    args = parse_args()
    dataset = load_data(args)
    print("")
