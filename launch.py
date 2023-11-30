
import os,sys
from pytorch_lightning import loggers as pl_loggers

def wandbtrain(config=None,dir=None,devices=None,accelerator=None,Dataset=None,project="6DIMCLIPTOKSweepvfinal",entity="st7ma784"):
    import pytorch_lightning
    import wandb
    if config is not None:
        config=config.__dict__
        logdir=config.get("log_path",dir)
        wandb.login(key='9cf7e97e2460c18a89429deed624ec1cbfb537bc')
        if config.get("dims",6)==3:
            project+="3DIM"
        logtool= pytorch_lightning.loggers.WandbLogger( project=project,entity=entity, save_dir=logdir)

    else: 
        #We've got no config, so we'll just use the default, and hopefully a trainAgent has been passed
        import wandb
        print("here")
        wandb.login(key='9cf7e97e2460c18a89429deed624ec1cbfb537bc')
        #n=wandb.init(project="SPARC-VisGenome",entity="st7ma784",name="VRE-Vis",config=args)

        #logtool= pl.loggers.WandbLogger( project="SPARC-VisGenome",entity="st7ma784",name="VRE-Vis",experiment=run,save_dir=savepath,log_model=True)

        run=wandb.init(project=project,entity=entity,name=project,config=config)
        config=run.config.as_dict()
        if config.get("dims",6)==3:
            project+="3DIM"
        logtool= pytorch_lightning.loggers.WandbLogger( project=project,entity=entity,experiment=run, save_dir=config.get("log_path",dir))
    
    train(config,dir,devices,accelerator,Dataset,logtool)
def train(config={
        "batch_size":16,
        "learning_rate":2e-3,
        "precision":16,
        "embed_dim": 512,
        "codeversion":4,
        "transformer_width": 512,
        "transformer_heads": 32,
        "transformer_layers": 4,
        "JSE":False,
    },dir=None,devices=None,accelerator=None,Dataset=None,logtool=None):

    import pytorch_lightning
    version=int(config.get("codeversion",-1))
    
    from pytorch_lightning.callbacks import TQDMProgressBar,EarlyStopping
    if config.get("dims",6)==3:
        from model.trainclip_v533DIM import LightningCLIPModule
    elif config.get("dims",6)==3.5:
        from model.trainclip_v5335DIM import LightningCLIPModule
    elif config.get("dims",6)==4:
        from model.trainclip_v534DIM import LightningCLIPModule
    else:
        from model.trainclip_v53 import LightningCLIPModule
    # from pl_bolts.datamodules import ImagenetDataModule
    model=LightningCLIPModule( train_batch_size=config["batch_size"],
                                **config)
    logger=[]
    if logtool:
        #logtool.watch(model, log_freq=1000,log_graph=False)
        logger.append(logtool)
    #else:
        #logger.append(pytorch_lightning.loggers.TensorBoardLogger(dir, name="CLIPv{}".format(version)))
    if dir is None:
        dir=config.get("dir",".")
    if Dataset is None:
        from BuildSpainDataSet import COCODataModule
        
        #Dataset=LaionDataModule(Cache_dir=dir,batch_size=config["batch_size"])
        Dataset=COCODataModule(Cache_dir=dir,annotations=config.get("annotations",dir),batch_size=config["batch_size"])
        from BuildImagenet import ImagenetDataModule
        from pytorch_lightning.strategies import DDPStrategy as DDP

        TestLoader=ImagenetDataModule(
            data_dir=dir, 
            meta_dir=dir,
            num_imgs_per_val_class=50,
            image_size=224,
            num_workers=4, 
            batch_size=config["batch_size"], 
            shuffle=True,
            pin_memory=True,
            drop_last=True)
    if devices is None:
        devices=config.get("devices","auto")
    if accelerator is None:
        accelerator=config.get("acceleartor","auto")
    # print("Training with config: {}".format(config))
    Dataset.batch_size=config["batch_size"]
    callbacks=[
        TQDMProgressBar(),
        EarlyStopping(monitor="train_loss", mode="min",patience=10,check_finite=True,stopping_threshold=0.001),
    ]
    p=config['precision']
    if isinstance(p,str):
        p=16 if p=="bf16" else int(p)  ##needed for BEDE
    #for windows .... 
    if sys.platform == "win32":
       os.environ["PL_TORCH_DISTRIBUTED_BACKEND"]='gloo'
    print("Launching with precision",p)
    trainer=pytorch_lightning.Trainer(
            devices=devices,
            #auto_select_gpus=True,
            accelerator=accelerator,
            max_epochs=6,
            #profiler="advanced",
            logger=logger,
            strategy=DDP(find_unused_parameters=True),
            num_nodes=int(os.getenv("SLURM_NNODES",1)),
            callbacks=callbacks,
            gradient_clip_val=0.25,# Not supported for manual optimization
            accumulate_grad_batches=16,
            fast_dev_run=config["debug"],
            precision=p
    )
    if config["batch_size"] !=1:
        
        trainer.fit(model,Dataset)
        trainer.test(model,TestLoader)
    else:
        return 0 #No need to train if batch size is 1
    #do test
  
def SlurmRun(trialconfig):

    job_with_version = '{}v{}'.format("SINGLEGPUTESTLAUNCH", 0)

    sub_commands =['#!/bin/bash',
        '# Auto-generated by test-tube (https://github.com/williamFalcon/test-tube)',   
        '#SBATCH --time={}'.format( '24:00:00'),# Max run time
        '#SBATCH --job-name={}'.format(job_with_version), 
        '#SBATCH --nodes=1',  #Nodes per experiment
        '#SBATCH --ntasks-per-node=1',  #Tasks per node
        '#SBATCH --gres=gpu:1',  #{}'.format(per_experiment_nb_gpus),
        #'#SBATCH --gres=gpu:{}:{}'.format(self.gpu_type, self.per_experiment_nb_gpus),    If you want to specify a GPU type
        f'#SBATCH --signal=USR1@{5 * 60}',
        '#SBATCH --mail-type={}'.format(','.join(['END','FAIL'])),
        '#SBATCH --mail-user={}'.format('st7ma784@gmail.com'),
    ]
    comm="python"
    if str(os.getenv("HOSTNAME","localhost")).endswith("bede.dur.ac.uk"):
        sub_commands.extend([
        '#SBATCH --account=bdlan05',
        'export CONDADIR=/nobackup/projects/bdlan05/$USER/miniconda',])
        #slurm_commands={"account":"bdlan05"}#,"partition":"gpu"} Leaving this part out to run on non-bede slurm
        comm="python3"
    else: 
        sub_commands.extend(['export CONDADIR=/home/user/miniconda3',])
        #slurm_commands={}
    #sub_commands.extend([ '#SBATCH --{}={}\n'.format(cmd, value) for  (cmd, value) in slurm_commands.items()])
    sub_commands.extend([
        '#SBATCH --mem-per-node=62G',  #Memory per node
        'export SLURM_NNODES=$SLURM_JOB_NUM_NODES',
        'export wandb=9cf7e97e2460c18a89429deed624ec1cbfb537bc',
        'source $CONDADIR/etc/profile.d/conda.sh',
        'conda activate open-ce',# ...and activate the conda environment
    ])
    #sub_commands.append("srun python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr='
    script_name= os.path.realpath(sys.argv[0]) #Find this scripts name...
    trialArgs=__get_hopt_params(trialconfig)

    sub_commands.append('{} {} {}'.format(comm, script_name,trialArgs))
    #when launched, this script will be called with no trials, and so drop into the wandbtrain section, 
    sub_commands = [x.lstrip() for x in sub_commands]        

    full_command = '\n'.join(sub_commands)
    return full_command

def __get_hopt_params(trial):
    """
    Turns hopt trial into script params
    :param trial:
    :return:
    """
    params = []
    for k in trial.__dict__:
        v = trial.__dict__[k]
        if k == 'num_trials':
            v=0
        # don't add None params
        if v is None or v is False:
            continue

        # put everything in quotes except bools
        if __should_escape(v):
            cmd = '--{} \"{}\"'.format(k, v)
        else:
            cmd = '--{} {}'.format(k, v)
        params.append(cmd)

    # this arg lets the hyperparameter optimizer do its thin
    full_cmd = ' '.join(params)
    return full_cmd

def __should_escape(v):
    v = str(v)
    return '[' in v or ';' in v or ' ' in v


if __name__ == '__main__':
    from HOparser import parser
    from subprocess import call
    myparser=parser()
    hyperparams = myparser.parse_args()
    defaultConfig=hyperparams.__dict__
   
    NumTrials=hyperparams.num_trials
    #BEDE has Env var containing hostname  #HOSTNAME=login2.bede.dur.ac.uk check we arent launching on this node
    if NumTrials==-1:
        trial=hyperparams.generate_trials(1)[0]
        print("Running trial: {}".format(trial))
        wandbtrain(trial)

    elif NumTrials ==0 and not str(os.getenv("HOSTNAME","localhost")).startswith("login"): #We'll do a trial run...
        #means we've been launched from a BEDE script, so use config given in args///
        wandbtrain(hyperparams)

    #OR To run with Default Args
    else: 
        trials=hyperparams.generate_trials(NumTrials)
    
        for i,trial in enumerate(trials):             
            print("Running trial: {}".format(trial))
            command=SlurmRun(trial)
            slurm_cmd_script_path = os.path.join(defaultConfig.get("dir","."),"slurm_cmdtrial{}.sh".format(i))
            with open(slurm_cmd_script_path, "w") as f:
              f.write(command)
            print('\nlaunching exp...')
            result = call('{} {}'.format("sbatch", slurm_cmd_script_path), shell=True)
            if result == 0:
                print('launched exp ', slurm_cmd_script_path)
            else:
                print('launch failed...')  