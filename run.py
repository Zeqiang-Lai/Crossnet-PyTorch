from torchlight import config
from torchlight.entry import run
from torchlight.utils.helper import get_obj

import source.module as module
from source.data.dataloader import RGBRefSRDataLoader

if __name__ == '__main__':
    args, cfg = config.basic_args()
  
    train_loader = RGBRefSRDataLoader(**cfg.loader.train)
    test_loader = RGBRefSRDataLoader(augment=False, **cfg.loader.test)
    
    module = get_obj(cfg.module, module)

    run(args, cfg, module, train_loader, test_loader, test_loader)
    