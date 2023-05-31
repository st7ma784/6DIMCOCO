import torch
from flask import Flask, render_template, request, jsonify
import json

#use this to append the path to the model folder
from nargsLossCalculation import get_loss_fn
#this is the index page,
app = Flask(__name__,template_folder='.')

#there is n orbs on the html page. 
# Each orb has a unique id and an x,y coordinate
# They are all the same size and have the same color
# The orbs are positioned in a circle

#when the user clicks on an orb, they may drag it to a new position

#when the user clicks on the submit button, the new positions of the orbs are sent to the server

#the server calculates the distance between each orb and every other orb

#the server updates S on the webpage with the distances
func=lambda x: x**2


@app.route("/")
def index():
    return render_template("./index.html")

@app.route('/', methods=['POST'])
def ImportLogitMethod():
    #read in the value of the dropdown
    global func
    method = request.values.get('method')
    #print(method)
    func=get_loss_fn(method)
    return render_template("./index.html")
@app.route('/data', methods=['GET','POST'])
def getS():
    #data will be sent as a jsonified dict of x and y
    global func
    data=request.get_json()
    #print(data)
    #un stringify the data
    x=filter(lambda a: a != '',data['x'])
    y=filter(lambda a: a != '',data['y'])
    
    #print(x,y)
    width=data['width']
    height=data['height']
    wh=torch.tensor([[width,height]])/2
    #convert ("numberpx") to number
    x=[float(x[:-2]) for x in x]
    y=[float(y[:-2]) for y in y]
    #print("x",x,"y",y)
    #the div container "graph" has all the orbs
    xys=[(torch.tensor([[x,y]])-wh)/wh for x,y in zip(x,y)]
    #y=[orb.cy for orb in request.form.get('graph')]
    #for xy in xys:
    #    print(xy)
    #print( xy.shape for xy in xys)
    return str(func(*xys).item())
     
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
