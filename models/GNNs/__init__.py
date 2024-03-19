# from .GCN import GCN


def parse_GNNs(args):
    model = None
    if args.GNNs_model == "GCN":
        # model = GCN(args)
        pass
    else:
        raise ValueError("Model not found")
    return model
