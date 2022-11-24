
from dataloader import load_data
from models import get_model

# import hydra
# from omegaconf import DictConfig

# https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b


def main(cfg: DictConfig):
    
    model = get_model(cfg)
    predicts = model(test_data)

    # SAVE

if __name__ == '__main__':
    main()
