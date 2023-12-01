import torch
from util.train_args import parse_args
from models.RDCNN import RDCNN
from models.CSRNet import CSRNet
from models.MCNN import MCNN
from data.dataset import crowddataset
from trainer import trainer

if __name__ == '__main__':
    args = parse_args()
    device = args.device

    ## dataset
    train_dataset = crowddataset(args.datadir, stage="train")
    test_dataset = crowddataset(args.datadir, stage="test")

    ## dataloader
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle = True,
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle = False,
    )

    ## model

    # model = RDCNN(pretained=True).to(device)
    # model = CSRNet().to(device)
    model = MCNN().to(device)
    net_trainer = trainer(model, train_data_loader, test_data_loader, args, device)
    net_trainer.start()