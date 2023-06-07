import torch
from flask import Flask, render_template, request, jsonify
from nargsLossCalculation import get_loss_fn
from functools import reduce

def mean(args):
    return reduce(torch.add,args/len(args))
def std(args):
    return torch.sqrt(mean([(a-mean(args))**2 for a in args]))
def variance(args):
    return mean([(a-mean(args))**2 for a in args])
def l2mean(args):
    return torch.sqrt(mean([a**2 for a in args]))
def lsqrtmean(args):
    return torch.pow(mean([torch.sqrt(a) for a in args]),2)
def l3mean(args):
    return torch.pow(mean([torch.pow(a,3) for a in args]),1/3)
def dynmean(args):
    return torch.pow(mean([torch.pow(a,len(args)) for a in args]),1/len(args))
if __name__ == "__main__":
    functions={i:get_loss_fn(i,norm=False) for i in range(1,17)}
    normedfunctions={i:get_loss_fn(i,norm=True) for i in range(1,17)}
    usefulpoints={"mean":mean,"variance":variance, "std":std,"l2mean":l2mean,"l3mean":l3mean,"lsqrtmean":lsqrtmean,"dynmean":dynmean}
    app = Flask(__name__,template_folder='.')
    @app.route("/demo") 
    def index():
        return render_template("./index.html")
    
    @app.route('/demo/data', methods=['GET','POST'])
    async def getS():
        data=request.get_json()
        wh=torch.tensor([[data['width'],data['height']]])/2
        x=[float(x[:-2]) for x in filter(lambda a: a != '',data['x'])]
        y=[float(y[:-2]) for y in filter(lambda a: a != '',data['y'])]
        xys=[(torch.tensor([[x,y]],requires_grad=False)-wh)/wh for x,y in zip(x,y)]
        
        with torch.no_grad():                   
            return jsonify([str(func(*xys).item()) for func in functions.values()])
    @app.route('/demo/norm/data', methods=['GET','POST'])
    async def getnormS():
        data=request.get_json()
        wh=torch.tensor([[data['width'],data['height']]])/2
        x=[float(x[:-2]) for x in filter(lambda a: a != '',data['x'])]
        y=[float(y[:-2]) for y in filter(lambda a: a != '',data['y'])]
        xys=[(torch.tensor([[x,y]],requires_grad=False)-wh)/wh for x,y in zip(x,y)]
        
        with torch.no_grad():                   
            return jsonify([str(func(*xys).item()) for func in normedfunctions.values()])
    # these 2 functions really could be combined into one, but I'm lazy and there may be future reasons to keep them seperate
    @app.route('/demo/points', methods=['GET','POST'])
    async def getmetricS():
        data=request.get_json()
        wh=torch.tensor([[data['width'],data['height']]])/2
        x=[float(x[:-2]) for x in filter(lambda a: a != '',data['x'])]
        y=[float(y[:-2]) for y in filter(lambda a: a != '',data['y'])]
        xys=[(torch.tensor([[x,y]],requires_grad=False)-wh)/wh for x,y in zip(x,y)]
        
        with torch.no_grad(): 
            out = {name:(torch.nan_to_num(func(xys))*wh).tolist() for name,func in usefulpoints.items()}
            #print(out)# these are all relative to the width and height of the image

            return jsonify(out)
    # run at /smander
    app.run(host="0.0.0.0", port=5000, debug=False)
  