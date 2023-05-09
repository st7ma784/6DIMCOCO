import wandb

if __name__=="__main__":
    sweep_config = {
        'name':"Final Deploy",
        'method': 'bayes',  # Randomly sample the hyperparameter space (alternatives: grid, bayes)
        'metric': {  # This is the metric we are interested in maximizing
            'name': 'train_loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'values':[5e-4]
            },
            'batch_size': {
                'values': [10]
            },
            'precision': {
                'values': ['32']
            },
            'maskLosses': {
                'values': [0,1,2]
            },
            'embed_dim':{
                'values': [512]
            }, 
            'transformer_width':{
                'values': [512]
            },
            'logitsversion':{
                'values':[16]
            },
            "prune":{
                'values':[False]
            },
            "meanloss":{
                'values':[True,False]
            },
            "projection":{
                'values':["None","inv","iinv", ""]
            },
            'transformer_heads':{
                'values': [16]
            },
            'transformer_layers':{
                'values': [24]
            },
        }
    }
    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="6DimCachespliteinSweep", entity="st7ma784")
    print(sweep_id)
