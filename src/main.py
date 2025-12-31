from src.trainer import Trainer
from src.config import Config


def main():
    parser = Config.new_parser()
    Config.add_argument(parser)
    args = parser.parse_args()

    trainer = Trainer(args=args)
    trainer.train()


if __name__ == "__main__":
    main()
