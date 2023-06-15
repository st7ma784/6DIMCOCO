import torch
from flask import Flask, render_template, request, jsonify, send_file
from nargsLossCalculation import get_loss_fn
from functools import reduce
from glob import glob
from io import BytesIO
from zipfile import ZipFile
import numpy as np
from matplotlib import pyplot as plt
def mean(args):
    return reduce(torch.add,[a/len(args) for a in args])
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
def square(logits):
    #a function that takes numpy logits and plots them on an x and y axis
    plt.figure(figsize=(logits.shape[0],logits.shape[1]))
    plt.imshow(logits)
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    return img_buf    


def cubes(logits):
    # Defining the size of the axes
    x, y, z = np.indices(logits.shape)
    sidex,sidey,sidez=logits.shape
    # Defining the length of the sides of the cubes
    cube = (x < sidex) & (y < sidey) & (z < sidez)
    # Defining the shape of the figure to be a cube
    voxelarray = cube
    # Defining the colors for the cubes
    colors = np.empty((3,*voxelarray.shape), dtype=float)
    # Defining the color of the cube
    c=np.sqrt(np.sqrt(logits.flatten()).unflatten(0,(sidex,sidey,sidez)).cpu().numpy())
    #colors[cube] = c.astype(str)[cube]
    colors= np.stack([c,c,1- c],axis=-1)
    print(colors.shape)
    # Defining the axes and the figure object
    ax = plt.figure(figsize=(9, 9)).add_subplot(projection='3d')
    # Plotting the cube in the figure
    ax.voxels(voxelarray , facecolors=colors, edgecolor='k')
    # Defining the title of the graph
    #plt.title("Batch MSE loss between random values and perfect case in n=3 dimensions")
    # Displaying the graph
    #save graph to IO buffer and return 
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    return img_buf

def hsquare(logits):
    # Defining the size of the axes
    sidew,sidex,sidey,sidez=logits.shape
    x,y = np.indices((sidew*sidex,sidey*sidez))
    # Defining the length of the sides of the cubes
    square = (x < (sidew*sidex)) & (y < (sidey*sidez))
    # Defining the shape of the figure to be a cube
    voxelarray = square
    # Defining the colors for the cubes
    colors = np.empty((*voxelarray.shape,3), dtype=int)
    # Defining the color of the cube
    #input is shape (B, B, B,B) 
    #C needs to be (b*b, B*b) \
    c=torch.softmax(logits.flatten(),dim=0).unflatten(0,(sidew*sidex,sidey*sidez)).cpu().numpy()
    c=c/np.amax(c)

    #c=np.sqrt(np.sqrt(smax(MLoss(sqsqlogits,pfsqsqlogits).flatten()).unflatten(0,(B*B,B*B)).cpu().numpy()))
    colors= np.stack([c,c,c],axis=-1)
    # Defining the axes and the figure object
    ax = plt.imshow(colors[:, :, :])

    # Plotting the cube in the figure
    # Defining the title of the graph
    #plt.title("Perfect case logits in n=4 dimensions plotted on a (B^2, B^2) ")
    # Displaying the graph
    #save graph to IO buffer and return 
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    return img_buf
# Defining the main() function
def hypcubes(logits):
    # Defining the size of the axes
    side=logits.shape[0]
    u,v,w,x, y, z = np.indices(logits.shape)
    # Defining the length of the sides of the cubes
    cube =  (u < side) & (v < side) & (w < side)& (x < side) & (y < side) & (z < side)
    # Defining the shape of the figure to be a cube
    voxelarray = cube
    # Defining the colors for the cubes
    colors = np.empty(voxelarray.shape, dtype=object)
    # Defining the color of the cube
    c=np.sqrt(np.sqrt(torch.softmax(logits.flatten()).unflatten(0,(u,v,w,x, y, z)).cpu().numpy()))*2
    colors[cube] = c.astype(str)[cube]
    # Defining the axes and the figure object
    ax = plt.figure(figsize=(9, 9)).add_subplot(projection='3d')
    # Plotting the cube in the figure
    ax.voxels(voxelarray , facecolors=colors, edgecolor='k')
    # Defining the title of the graph
    #plt.title("Batch MSE loss between random values and perfect case in n=3 dimensions")
    # Displaying the graph
    #save graph to IO buffer and return 
    

    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    return img_buf
def draw(logits):
    # Defining the side of the cube
    sides = len(logits.shape) #(subtract to take slices)
    # Calling the cubes () function
    #cubes(sides)
    if sides==2:
      return square(logits)
    if sides==3:
      return cubes(logits)
    if sides==4:
      return hsquare(logits)
    if sides==6:
      return hypcubes(logits)


if __name__ == "__main__":
    functions={i:get_loss_fn(i,norm=False) for i in range(1,17)}
    normedfunctions={i:get_loss_fn(i,norm=True) for i in range(1,17)}
    usefulpoints={"mean":mean,"variance":variance, "std":std,"l2mean":l2mean,"l3mean":l3mean,"lsqrtmean":lsqrtmean,"dynmean":dynmean}
    app = Flask(__name__,template_folder='.')
    @app.route("/demo") 
    def index():
        return render_template("./index.html")
    @torch.no_grad()
    @app.route('/demo/data', methods=['GET','POST'])
    async def getS():
        data=request.get_json()
        wh=torch.tensor([[data['width'],data['height']]])/2
        x=[float(x[:-2]) for x in filter(lambda a: a != '',data['x'])]
        y=[float(y[:-2]) for y in filter(lambda a: a != '',data['y'])]
        xys=[(torch.tensor([[x,y]],requires_grad=False)-wh)/wh for x,y in zip(x,y)]
        stats=data['stats']
        out={}
        
        if stats:
            out={name:(torch.nan_to_num(func(xys))*wh).tolist() for name,func in usefulpoints.items()}
            #getting error that > not supported between instances of int and str??

        normed=data['norm']
        if normed:
            out.update({str(name):(torch.nan_to_num(func(*xys))).tolist() for name,func in normedfunctions.items()})
        else:
            out.update({str(name):(torch.nan_to_num(func(*xys))).tolist() for name,func in functions.items()})
        return jsonify(out)
    
    @torch.no_grad()
    @app.route('/demo/Plotfour', methods=['GET','POST'])
    async def getPlot4():
        data=request.get_json()
        wh=torch.tensor([[data['width'],data['height']]])/2
        x=[float(x[:-2]) for x in filter(lambda a: a != '',data['x'])]
        y=[float(y[:-2]) for y in filter(lambda a: a != '',data['y'])]
        xys=torch.stack([torch.tensor([[x,y]],requires_grad=False)for x,y in zip(x,y)])-wh
        xys=xys/wh         
        normed=data['norm']

        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:              
            if normed:
                map(lambda x: zip_file.writestr('4DNormedGraphMethod{}.png'.format(x[0]), draw(torch.nan_to_num(x[1](xys,xys,xys,xys)))),normedfunctions.items())
            else:
                map(lambda x: zip_file.writestr('4DNormedGraphMethod{}.png'.format(x[0]), draw(torch.nan_to_num(x[1](xys,xys,xys,xys)))),functions.items())

        zip_buffer.seek(0)
        return send_file(zip_buffer, attachment_filename='Graphs.zip', as_attachment=True)

    app.run(host="0.0.0.0", port=5000, debug=False)
  
