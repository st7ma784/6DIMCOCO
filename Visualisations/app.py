import torch
from flask import Flask, render_template, request, jsonify

#use this to append the path to the model folder
from nargsLossCalculation import get_loss_fn
#this is the index page,
app = Flask(__name__,template_folder='.')

def wrap(f): # wraps a function in async and await
    async def wrapped(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapped

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
    #convert ("numberpx") to number
    x=[float(x[:-2]) for x in filter(lambda a: a != '',data['x'])]
    y=[float(y[:-2]) for y in filter(lambda a: a != '',data['y'])]
   
    xys=[(torch.tensor([[x,y]])-wh)/wh for x,y in zip(x,y)]

    outputs={k:str(func(*xys).item()) for k,func in functions.items()}

    return jsonify(outputs)


     
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
