from torchlight import config
from torchlight.entry import run
from torchlight.utils.helper import get_obj
from torchlight.utils.reproducibility import SeedDataLoader, setup_randomness

import source.module as module
import source.data.dataset as dataset
from source.data.collate import default_collate

if __name__ == '__main__':
    args, cfg = config.basic_args()
    setup_randomness(2021)
    
    train_dataset = get_obj(cfg.train.dataset, dataset)
    test_dataset = get_obj(cfg.test.dataset, dataset)
    
    train_loader = SeedDataLoader(train_dataset, collate_fn=default_collate, **cfg.train.loader)
    test_loader = SeedDataLoader(test_dataset, collate_fn=default_collate, **cfg.test.loader)
    
    module = get_obj(cfg.module, module)

    run(args, cfg, module, train_loader, test_loader, test_loader)
    