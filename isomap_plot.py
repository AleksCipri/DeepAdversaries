#https://pythonmatplotlibtips.blogspot.com/2018/01/rotate-azimuth-angle-animation-3d-python-matplotlib-pyplot.html
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.manifold import Isomap
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def isomap_plot(embed_network, embed_reg, embed_nois, embed_pert, base_network, img_reg, img_noise, num, three_dim=True, save=False, where=False): 
    np.random.seed(0)
    ind = np.random.randint(0,len(img_reg),num)
    num2 = 2 * num
    num3 = num2+ len(embed_reg)
    num4 = num3 + len(embed_nois)
    #print(num,num2,num3,num4)

    #Make embeddings for many images to build Isomap space  
    base_network.to('cpu')
    embedding_reg, logits_reg = base_network(img_reg[ind])
    embedding_noise, logits_noise = base_network(img_noise[ind])

    softmax_reg = nn.Softmax(dim=1)(1.0 * logits_reg)
    _, predict_reg = torch.max(softmax_reg, 1)
    softmax_noi = nn.Softmax(dim=1)(1.0 * logits_noise)
    _, predict_noi = torch.max(softmax_noi, 1)

    predict_reg = predict_reg.tolist()
    predict_noi = predict_noi.tolist()


    # Import embedings for selected attacked triplets
    embed_network.to('cpu')
    embed_reg = embed_reg.detach().squeeze().to('cpu')
    embed_nois = embed_nois.detach().squeeze().to('cpu')
    embed_pert = embed_pert.detach().squeeze().to('cpu')
    
    log_reg = embed_network(embed_reg)
    sm_reg = nn.Softmax(dim=1)(1.0*log_reg)
    _, pr_reg = torch.max(sm_reg, 1)
    
    log_noi = embed_network(embed_nois)
    sm_noi = nn.Softmax(dim=1)(1.0*log_noi)
    _, pr_noi = torch.max(sm_noi, 1)
    
    log_pert = embed_network(embed_pert)
    sm_pert = nn.Softmax(dim=1)(1.0*log_pert)
    _, pr_pert = torch.max(sm_pert, 1)


    total = []
    total.append(embedding_reg)
    total.append(embedding_noise)
    total.append(embed_reg)
    total.append(embed_nois)
    total.append(embed_pert)

    total = torch.cat(total)

    #color = np.array(['r', 'b', 'k'])
    color = np.array(['#ffa600','#bc5090','#003f5c'])

    if three_dim:
        model = Isomap(n_components=3)
        proj = model.fit_transform(total.detach().cpu().numpy())

        
        fig = plt.figure(figsize = (20, 20))
        ax = plt.gca()
        ax = plt.axes(projection='3d')
        ax.view_init(elev=10)
        plt.title('Isomap 3D \n Curves: Y10 (orange), Y1 noisy (blue) \n Point color - predicted class: spiral (red), elliptical (blue), merger (black) \n Perturbed images - Y10 image (+), Y1  noisy (*), perturbed image (^) \n', fontsize=25)

        ax.plot_trisurf(proj[:num, 0], proj[:num, 1], proj[:num, 2], cmap="Oranges", alpha = 0.5)
        ax.plot_trisurf(proj[num:num2, 0], proj[num:num2, 1], proj[num:num2, 2],cmap="Blues", alpha=0.5)
        ax.scatter(proj[:num, 0], proj[:num, 1], proj[:num, 2], c=color[predict_reg[:]],marker='o', s=80)
        ax.scatter(proj[num:num2, 0], proj[num:num2, 1], proj[num:num2, 2],c=color[predict_noi[:]],marker='o', s=20)

        ax.scatter(proj[num2:num3, 0], proj[num2:num3, 1], proj[num2:num3, 2],c=color[pr_reg[:]],marker='X', edgecolors='k', s=400)
        ax.scatter(proj[num3:num4, 0], proj[num3:num4, 1], proj[num3:num4, 2],c=color[pr_noi[:]],marker='*', edgecolors='k', s=400)
        ax.scatter(proj[num4:, 0], proj[num4:, 1], proj[num4:, 2],c=color[pr_pert[:]],marker='^', edgecolors='k', s=400)
         
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.zaxis.set_ticks_position('none')

    else:
        model = Isomap(n_components=2)
        proj = model.fit_transform(total.detach().cpu().numpy())

        fig = plt.figure(figsize = (20, 10))
        ax = plt.gca()
        plt.title('Isomap 2D \n Point color - predicted class: spiral (yellow), elliptical (violet), merger (navy) \n Test images: Y10 large, Y1 noisy small \n Perturbed images - Y10 image (+), Y1  noisy (*), perturbed image (^) \n', fontsize=25)

        ax.scatter(proj[:num, 0], proj[:num, 1], c=color[predict_reg[:]],marker='o', s=80)
        ax.scatter(proj[num:num2, 0], proj[num:num2, 1] ,c=color[predict_noi[:]],marker='o', s=20)

        ax.scatter(proj[num2:num3, 0], proj[num2:num3, 1],c=color[pr_reg[:]],marker='X',edgecolors='k', s=400)
        ax.scatter(proj[num3:num4, 0], proj[num3:num4, 1],c=color[pr_noi[:]],marker='*',edgecolors='k', s=400)
        ax.scatter(proj[num4:, 0], proj[num4:, 1],c=color[pr_pert[:]],marker='^',edgecolors='k', s=400)
        
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')


    if save:
        plt.savefig(where+'.png')
        
    fig.clear()
    plt.close(fig)
        
        
        
def isomap_video(embed_network, embed_reg, embed_nois, embed_pert, base_network, img_reg, img_noise, num, vert=False, save=False, where=False):
    np.random.seed(0)
    ind = np.random.randint(0,len(img_reg),num)
    num2 = 2 * num
    num3 = num2+ len(embed_reg)
    num4 = num3 + len(embed_nois)
    #print(num,num2,num3,num4)

    #Make embeddings for many images to build Isomap space  
    base_network.to('cpu')
    embedding_reg, logits_reg = base_network(img_reg[ind])
    embedding_noise, logits_noise = base_network(img_noise[ind])

    softmax_reg = nn.Softmax(dim=1)(1.0 * logits_reg)
    _, predict_reg = torch.max(softmax_reg, 1)
    softmax_noi = nn.Softmax(dim=1)(1.0 * logits_noise)
    _, predict_noi = torch.max(softmax_noi, 1)

    predict_reg = predict_reg.tolist()
    predict_noi = predict_noi.tolist()


    # Import embedings for selected attacked triplets
    embed_network.to('cpu')
    embed_reg = embed_reg.detach().squeeze().to('cpu')
    embed_nois = embed_nois.detach().squeeze().to('cpu')
    embed_pert = embed_pert.detach().squeeze().to('cpu')
    
    log_reg = embed_network(embed_reg)
    sm_reg = nn.Softmax(dim=1)(1.0*log_reg)
    _, pr_reg = torch.max(sm_reg, 1)
    
    log_noi = embed_network(embed_nois)
    sm_noi = nn.Softmax(dim=1)(1.0*log_noi)
    _, pr_noi = torch.max(sm_noi, 1)
    
    log_pert = embed_network(embed_pert)
    sm_pert = nn.Softmax(dim=1)(1.0*log_pert)
    _, pr_pert = torch.max(sm_pert, 1)


    total = []
    total.append(embedding_reg)
    total.append(embedding_noise)
    total.append(embed_reg)
    total.append(embed_nois)
    total.append(embed_pert)

    total = torch.cat(total)

    #color = np.array(['r', 'b', 'k'])
    color = np.array(['#ffa600','#bc5090','#003f5c'])

    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    fig = plt.figure(figsize = (20, 10))
    ax = plt.gca()
    ax = plt.axes(projection='3d')
    plt.title('Isomap 3D \n Curves: Y10 (orange), Y1 noisy (blue) \n Point color - predicted class: spiral (yellow), elliptical (violet), merger (navy) \n Perturbed images - Y10 image (+), Y1  noisy (*), perturbed image (^) \n', fontsize=25)

    def init():
        model = Isomap(n_components=3)
        proj = model.fit_transform(total.detach().numpy())

        ax.plot_trisurf(proj[:num, 0], proj[:num, 1], proj[:num, 2], cmap="Oranges", alpha = 0.5)
        ax.plot_trisurf(proj[num:num2, 0], proj[num:num2, 1], proj[num:num2, 2],cmap="Blues", alpha=0.5)
        ax.scatter(proj[:num, 0], proj[:num, 1], proj[:num, 2], c=color[predict_reg[:]],marker='o', s=80)
        ax.scatter(proj[num:num2, 0], proj[num:num2, 1], proj[num:num2, 2],c=color[predict_noi[:]],marker='o', s=20)

        ax.scatter(proj[num2:num3, 0], proj[num2:num3, 1], proj[num2:num3, 2],c=color[pr_reg[:]],marker='X',edgecolors='k', s=400)
        ax.scatter(proj[num3:num4, 0], proj[num3:num4, 1], proj[num3:num4, 2],c=color[pr_noi[:]],marker='*',edgecolors='k', s=400)
        ax.scatter(proj[num4:, 0], proj[num4:, 1], proj[num4:, 2],c=color[pr_pert[:]],marker='^', edgecolors='k', s=400)
        
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.zaxis.set_ticks_position('none')
        
        return fig,

    if vert:
        def animate(i):
          # elevation angle : -180 deg to 180 deg
          ax.view_init(elev=(i-45)*4, azim=10)
          return fig,
    else:
        def animate(i):
          # azimuth angle : 0 deg to 360 deg
          ax.view_init(elev=10, azim=i*4)
          return fig,

    # Animate
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                            frames=90, interval=50, blit=True)

    if save:
        fn = where
        ani.save(fn+'.mp4',writer='ffmpeg',fps=1000/50)
    
    fig.clear()
    plt.close(fig)
  
    return ani