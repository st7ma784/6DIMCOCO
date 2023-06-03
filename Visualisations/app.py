import torch
from flask import Flask, render_template, request, jsonify
from nargsLossCalculation import get_loss_fn

app = Flask(__name__,template_folder='.')

functions={i:get_loss_fn(i) for i in range(1,17)}
@app.route("/")
def index():
    return render_template("./index.html")

@app.route('/data', methods=['GET','POST'])
async def getS():
    global functions
    #data will be sent as a jsonified dict of x and y
    data=request.get_json()
    wh=torch.tensor([[data['width'],data['height']]])/2
    x=[float(x[:-2]) for x in filter(lambda a: a != '',data['x'])]
    y=[float(y[:-2]) for y in filter(lambda a: a != '',data['y'])]
    xys=[(torch.tensor([[x,y]])-wh)/wh for x,y in zip(x,y)]
    with torch.no_grad():
        outputs={k:str(func(*xys).item())+"<br>" for k,func in functions.items()}
    return jsonify(outputs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
