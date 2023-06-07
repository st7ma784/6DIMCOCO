import torch
from flask import Flask, render_template, request, jsonify
from nargsLossCalculation import get_loss_fn
from functools import reduce

def mean(args):
    return reduce(torch.add,*args)/len(args)
def std(args):
    return torch.sqrt(mean([(a-mean(args))**2 for a in args]))
def l2mean(args):
    return torch.sqrt(mean([a**2 for a in args]))

if __name__ == "__main__":
    functions={i:get_loss_fn(i,norm=False) for i in range(1,17)}
    normedfunctions={i:get_loss_fn(i,norm=True) for i in range(1,17)}
    usefulpoints=
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
            return jsonify({k:str(func(*xys).item())+"<br>" for k,func in functions.items()})
    @app.route('/demo/norm/data', methods=['GET','POST'])
    async def getnormS():
        data=request.get_json()
        wh=torch.tensor([[data['width'],data['height']]])/2
        x=[float(x[:-2]) for x in filter(lambda a: a != '',data['x'])]
        y=[float(y[:-2]) for y in filter(lambda a: a != '',data['y'])]
        xys=[(torch.tensor([[x,y]],requires_grad=False)-wh)/wh for x,y in zip(x,y)]
        
        with torch.no_grad():                   
            return jsonify({k:str(func(*xys).item())+"<br>" for k,func in normedfunctions.items()})
    # run at /smander
    @app.route('/demo/points', methods=['GET','POST'])
    async def getmetricS():
        data=request.get_json()
        wh=torch.tensor([[data['width'],data['height']]])/2
        x=[float(x[:-2]) for x in filter(lambda a: a != '',data['x'])]
        y=[float(y[:-2]) for y in filter(lambda a: a != '',data['y'])]
        xys=[(torch.tensor([[x,y]],requires_grad=False)-wh)/wh for x,y in zip(x,y)]
        
        with torch.no_grad(): 
            mean, std, l2mean = mean(xys), std(xys), l2mean(xys)
            print(mean,std,l2mean)                  
            return jsonify([mean,std,l2mean])
    # run at /smander
    app.run(host="0.0.0.0", port=5000, debug=False)
  