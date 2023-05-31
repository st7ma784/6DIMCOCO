import torch
from flask import Flask, render_template, request, jsonify


#use this to append the path to the model folder
from nargsLossCalculation import get_loss_fn
#this is the index page,
app = Flask(__name__)

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
    return render_template("index.html")

@app.route('/ra/connect', methods=['GET', 'POST'])
def connect_management():
    user = request.form.get('selected_class')
    print('user:', user)
    return str(user)
@app.route('/ra/updateMethod', methods=['GET', 'POST'])
def ImportLogitMethod():
    #read in the value of the dropdown
    global func
    method = request.form.get('Logitsversion')
    func=get_loss_fn(method)
    coords=request.form.get('coords')
    return func(*coords)

@app.route('/ra/getS', methods=['GET', 'POST'])
def getS():
    #read all the x and y from all orbs in graph
    #the div container "graph" has all the orbs
    xys=[torch.tensor([orb.cx,orb.cy]) for orb in request.form.get('graph')]
    #y=[orb.cy for orb in request.form.get('graph')]
    return func(*xys).item()
     


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
