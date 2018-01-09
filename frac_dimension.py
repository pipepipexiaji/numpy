import sys, os
import numpy as np

def frac_dimension(z, threshold=0.9):
    def pointcount(z,k):
        s=np.add.reduceat(np.add.reduceat(
            z, np.arange(0, z.shape[0], k), axis=0 ),
                         np.arange(0, z.shape[1], k), axis=1)
        return len(np.where( ( s>0 ) & (s<k*k) )[0])
    z=(z<threshold)
    p = min(z.shape)
    n=2**np.floor(np.log(p)/np.log(2))
    n=int(np.log(n)/np.log(2))
    sizes=2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(pointcount(z, size))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

if __name__=='__main__':
    from scipy import misc
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    z=1.0-misc.imread("https://shrtm.nu/zw04")/255
    
    print(frac_dimension(z, threshold=0.25))
    
    sizes = 128, 64, 32
    xmin, xmax = 0, z.shape[1]
    ymin, ymax = 0, z.shape[0]
    fig = plt.figure(figsize=(10, 5))

    for i, size in enumerate(sizes):
        ax = plt.subplot(1, len(sizes), i+1, frameon=False)
        ax.imshow(1-Z, plt.cm.gray, interpolation="bicubic", vmin=0, vmax=1,
                  extent=[xmin, xmax, ymin, ymax], origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        for y in range(z.shape[0]//size+1):
            for x in range(z.shape[1]//size+1):
                s = (z[y*size:(y+1)*size, x*size:(x+1)*size] > 0.25).sum()
                if s > 0 and s < size*size:
                    rect = patches.Rectangle(
                        (x*size, z.shape[0]-1-(y+1)*size),
                        width=size, height=size,
                        linewidth=.5, edgecolor='.25',
                        facecolor='.75', alpha=.5)
                    ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig("fractal-dimension.png")
    plt.show()