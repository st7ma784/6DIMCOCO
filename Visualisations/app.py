import torch
from flask import Flask, render_template, request, jsonify
from nargsLossCalculation import get_loss_fn

if __name__ == "__main__":
    functions={i:get_loss_fn(i,norm=True) for i in range(1,17)}

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
    # run at /smander

    app.run(host="0.0.0.0", port=5000, debug=False)
  