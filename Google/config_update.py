import omegaconf
with open('config.yaml') as f:
    conf = omegaconf.OmegaConf.load(f)
    print(conf)
    # update="trainer.max_steps=5"
    update="model.optim.schedule.warmup=5"
    parts = update.split("=")
    omegaconf.OmegaConf.update(conf, parts[0], parts[1], merge=True)
    print("After update:", conf)
    omegaconf.OmegaConf.save(conf, "updated_config.yaml")