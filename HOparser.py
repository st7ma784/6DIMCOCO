from test_tube import HyperOptArgumentParser

class parser(HyperOptArgumentParser):
    def __init__(self,*args,strategy="random_search",**kwargs):

        super().__init__( *args,strategy=strategy, add_help=False) # or random search
        #more info at https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/
        self.add_argument("--dir",default="/nobackup/projects/bdlan05/smander3/data",type=str)
        self.add_argument("--annotations",default="/nobackup/projects/bdlan05/smander3/data/annotations",type=str)
        self.add_argument("--log_path",default="/nobackup/projects/bdlan05/smander3/logs/",type=str)
        self.opt_list("--learning_rate", default=0.00001, type=float, options=[1e-3,1e-5, 1e-4,], tunable=True)
        self.opt_list("--batch_size", default=10, type=int, options=[6,8,10,12], tunable=True)
        self.opt_list("--JSE", default=0, type=int, options=[0], tunable=True)
        self.opt_list("--prune",default=False,type=bool,options=[True,False], tunable=True)
        self.opt_list("--projection",type=str,options=["None","inv","iinv"], tunable=True)
        self.opt_list("--normlogits",default=True,type=bool,options=[True,False], tunable=True)
        self.opt_list("--exactlabels",default=0,type=int,options=[1,0], tunable=True)
        self.opt_list("--meanloss",default=False,type=bool,options=[True,False], tunable=True)
        self.opt_list("--maskLosses",default=0,type=int,options=[0,1,2], tunable=True) #1 and 2 often result in nan in labels?
        self.opt_list("--debug",default=False,type=bool,options=[False], tunable=True)
        self.opt_list("--logitsversion",default=4,type=int,options=[0,1,2,3,4,5,6,7,8], tunable=True) #1 and 2 often result in nan in labels?
        self.opt_list("--precision", default=32, options=[16], type=int, tunable=False)
        self.opt_list("--codeversion", default=6, type=int, options=[6], tunable=False)
        self.opt_list("--transformer_layers", default=8, type=int, options=[3,4,5,6], tunable=True)
        self.opt_list("--transformer_heads", default=16, type=int, options=[16], tunable=True)
        self.opt_list("--embed_dim", default=512, type=int, options=[512], tunable=True)
        self.opt_list("--transformer_width", default=512, type=int, options=[128,512], tunable=True)
        self.opt_list("--devices", default=1, type=int, options=[1], tunable=False)
        self.opt_list("--accelerator", default='gpu', type=str, options=['gpu'], tunable=False)
        self.opt_list("--num_trials", default=0, type=int, tunable=False)
        #self.opt_range('--neurons', default=50, type=int, tunable=True, low=100, high=800, nb_samples=8, log_base=None)
        self.opt_list("--dims",default=3, type=float, options=[3.0,4.0,3.5,6.0,0], tunable=True)
        self.opt_list("--cn",default=False,type=bool,options=[True,False], tunable=True)


# Testing to check param outputs
if __name__== "__main__":
    myparser=parser()
    hyperparams = myparser.parse_args()
    print(hyperparams.__dict__)
    #we're going to check that all the options can be generated
    #step one make a dict of all the options
    #step two while loop over generate trials until we get all the options
    #step three print the options

    #step one
    options={}
    for arg,val in hyperparams.__dict__.items():
        options[arg]=set([getattr(hyperparams,arg)])
    for trial in hyperparams.generate_trials(100):
        #create parset and parse it
        print(trial)
        for arg,val in trial.__dict__.items():
            options[arg].add(val)

    #step three compare the options
    for arg,val in options.items():
        print(arg,val)

        
