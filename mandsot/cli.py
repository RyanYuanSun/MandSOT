import argparse
from mandsot import features, dataloader, model, train
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", "--mode", type=str, required=True, choices=['train', 'eval', 'vis'], help="")
    parser.add_argument("-device", "--device", type=str, required=False, default='auto', choices=['cuda', 'mps', 'cpu', 'auto'], help="")
    parser.add_argument("-noise", "--static_noise", type=bool, required=False, default=False, help="")
    parser.add_argument("-w1", "--w1", type=float, required=False, default=1, help="")
    parser.add_argument("-w2", "--w2", type=float, required=False, default=1, help="")
    parser.add_argument("-w3", "--w3", type=float, required=False, default=1, help="")
    parser.add_argument("-noise_width", "--static_noise_width", type=int, required=False, default=0, help="")
    parser.add_argument("-audio", "--audio", type=str, required=False, help="")
    parser.add_argument("-audio_dir", "--audio_dir", type=str, required=False, help="")
    parser.add_argument("-csv", "--csv", type=str, required=False, help="")
    parser.add_argument("-test", "--test_ratio", type=float, required=False, default=0.2, help="")
    parser.add_argument("-esp", "--earlystop_patience", type=int, required=False, default=5, help="")
    parser.add_argument("-esd", "--earlystop_delta", type=float, required=False, default=0, help="")
    parser.add_argument("-batch", "--batch_size", type=int, required=False, default=64, help="")
    parser.add_argument("-keep", "--keep_all", type=bool, required=False, default=True, help="")
    parser.add_argument("-o", "--output", type=str, required=False, help="")
    parser.add_argument("-cont", "--continued_train", type=bool, required=False, help="")
    parser.add_argument("-lr", "--learning_rate", type=float, required=False, help="")
    parser.add_argument("-verbose", "--verbose", type=bool, required=False, default=True, help="")
    args = parser.parse_args()

    if args.mode == 'train':
        # prepare dataset
        dataset = dataloader.load_train_csv(args.csv, args.verbose)
        dataset = dataloader.load_features(dataset, args.static_noise, args.w1, args.w2, args.w3, args.verbose)
        dataset_train, dataset_test = dataloader.split_dataset(dataset, test_ratio=args.test_ratio)

        # load dataset for training
        batch_size = args.batch_size
        dataset_train = dataloader.VoiceDataset(dataset_train)
        dataset_test = dataloader.VoiceDataset(dataset_test)
        train_loader = dataloader.load_dataset(dataset_train, batch_size=batch_size, shuffle=True)
        test_loader = dataloader.load_dataset(dataset_test, batch_size=batch_size, shuffle=False)

        # config train parameters
        if args.device != 'auto':
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        early_stopping = train.EarlyStopping(patience=args.earlystop_patience, delta=args.earlystop_delta)
        sot_model = model.MandSOT().to(device)

        # start training
        train.start(sot_model, train_loader, test_loader, device, args.learning_rate, early_stopping, args.output, args.keep_all)
    elif args.mode == 'eval':
        pass
    elif args.mode == 'vis':
        features.view_features(args.audio, args.static_noise)
    else:
        pass
